# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython shim for the linear Kalman filter hot loop.

Allocates the history output arrays, marshals the NumPy buffers into the C
kf_inputs/kf_outputs structs, runs kf_hot_loop under nogil, and translates the
status code back into the exceptions KalmanFilter.run already handles. The
public signature and return shape match the numba `_kalman_hot_loop` so the two
are interchangeable at the call site.
"""

import numpy as np
from numpy.linalg import LinAlgError

from libc.stdint cimport int64_t


cdef extern from "kalman.h":
    int KF_OK
    int KF_ERR_MATRIX_CONDITION
    int KF_ERR_ALLOC

    ctypedef struct kf_inputs:
        int64_t n
        int64_t m
        int64_t k
        int64_t T
        double *A
        double *B
        double *C
        double *d
        double *Q
        double *R
        double *y
        double *x0
        double *P0
        int symmetrize
        double jitter
        int return_shocks
        int store_history

    ctypedef struct kf_outputs:
        double *x_pred
        double *x_filt
        double *P_pred
        double *P_filt
        double *y_pred
        double *y_filt
        double *innov
        double *std_innov
        double *S
        double *eps_hat
        double *loglik

    int kf_hot_loop(const kf_inputs *inp, kf_outputs *outp) nogil

    ctypedef void (*meas_fn)(
        const double *x,
        const double *params,
        double *out,
    ) noexcept nogil

    ctypedef struct ukf_inputs:
        meas_fn meas
        double *hx
        double *gx
        double *bx
        double *hxx
        double *gxx
        double *hss
        double *gss
        double *steady_state
        double *params
        double *Q
        double *R
        double *obs
        double *z0
        double *P0
        int64_t T

        int64_t n_state
        int64_t n_ctrl
        int64_t n_exog
        int64_t n_obs
        int64_t n_params

        double alpha
        double beta
        double kappa

        double jitter
        int symmetrize
        int store_history

    ctypedef struct ukf_outputs:
        double *x1_pred
        double *x2_pred
        double *x1_filt
        double *x2_filt

        double *P_pred
        double *P_filt

        double *y_pred
        double *y_filt
        double *innov
        double *std_innov
        double *S

        double *loglik

    int64_t c_ukf_hot_loop "ukf_hot_loop"(
        const ukf_inputs *inp,
        ukf_outputs *outp,
    ) nogil


def kalman_hot_loop(
    int64_t T,
    nmk,
    double[:, ::1] A,
    double[:, ::1] B,
    double[:, ::1] C,
    double[::1] d,
    double[:, ::1] Q,
    double[:, ::1] R,
    double[:, ::1] y,
    double[::1] x0,
    double[:, ::1] P0,
    bint symmetrize,
    double jitter,
    bint return_shocks=False,
    bint store_history=True,
):
    """Run the linear Kalman filter; mirrors numba `_kalman_hot_loop`.

    Returns ``(KF_OK, (0.0, 0.0, 0.0), out)`` where ``out`` is the history tuple
    + loglik. Raises ``numpy.linalg.LinAlgError`` on a non-PD innovation
    covariance (so KalmanFilter.run's existing ``except`` maps it to
    ``MatrixConditionError``).
    """
    cdef int64_t n = nmk[0]
    cdef int64_t m = nmk[1]
    cdef int64_t k = nmk[2]

    cdef int64_t hist_T = T if store_history else 0
    cdef int64_t shock_T = T if (return_shocks and store_history) else 0

    x_pred = np.zeros((hist_T, n), dtype=np.float64)
    x_filt = np.zeros((hist_T, n), dtype=np.float64)
    P_pred = np.zeros((hist_T, n, n), dtype=np.float64)
    P_filt = np.zeros((hist_T, n, n), dtype=np.float64)
    y_pred = np.zeros((hist_T, m), dtype=np.float64)
    y_filt = np.zeros((hist_T, m), dtype=np.float64)
    innov = np.zeros((hist_T, m), dtype=np.float64)
    std_innov = np.zeros((hist_T, m), dtype=np.float64)
    S = np.zeros((hist_T, m, m), dtype=np.float64)
    eps_hat = np.zeros((shock_T, k), dtype=np.float64)
    cdef double loglik = 0.0

    cdef double[:, ::1] x_pred_mv = x_pred
    cdef double[:, ::1] x_filt_mv = x_filt
    cdef double[:, :, ::1] P_pred_mv = P_pred
    cdef double[:, :, ::1] P_filt_mv = P_filt
    cdef double[:, ::1] y_pred_mv = y_pred
    cdef double[:, ::1] y_filt_mv = y_filt
    cdef double[:, ::1] innov_mv = innov
    cdef double[:, ::1] std_innov_mv = std_innov
    cdef double[:, :, ::1] S_mv = S
    cdef double[:, ::1] eps_hat_mv = eps_hat

    cdef kf_inputs inp
    inp.n = n
    inp.m = m
    inp.k = k
    inp.T = T
    inp.A = &A[0, 0]
    inp.B = &B[0, 0]
    inp.C = &C[0, 0]
    inp.d = &d[0]
    inp.Q = &Q[0, 0]
    inp.R = &R[0, 0]
    inp.y = &y[0, 0] if T > 0 else NULL
    inp.x0 = &x0[0]
    inp.P0 = &P0[0, 0]
    inp.symmetrize = symmetrize
    inp.jitter = jitter
    inp.return_shocks = return_shocks
    inp.store_history = store_history

    cdef kf_outputs outp
    outp.x_pred = &x_pred_mv[0, 0] if hist_T > 0 else NULL
    outp.x_filt = &x_filt_mv[0, 0] if hist_T > 0 else NULL
    outp.P_pred = &P_pred_mv[0, 0, 0] if hist_T > 0 else NULL
    outp.P_filt = &P_filt_mv[0, 0, 0] if hist_T > 0 else NULL
    outp.y_pred = &y_pred_mv[0, 0] if hist_T > 0 else NULL
    outp.y_filt = &y_filt_mv[0, 0] if hist_T > 0 else NULL
    outp.innov = &innov_mv[0, 0] if hist_T > 0 else NULL
    outp.std_innov = &std_innov_mv[0, 0] if hist_T > 0 else NULL
    outp.S = &S_mv[0, 0, 0] if hist_T > 0 else NULL
    outp.eps_hat = &eps_hat_mv[0, 0] if shock_T > 0 else NULL
    outp.loglik = &loglik

    cdef int64_t status
    with nogil:
        status = kf_hot_loop(&inp, &outp)

    if status == KF_ERR_MATRIX_CONDITION:
        raise LinAlgError("Innovation covariance is not positive definite.")
    if status == KF_ERR_ALLOC:
        raise MemoryError("kf_hot_loop: scratch allocation failed.")

    return (
        KF_OK,
        (0.0, 0.0, 0.0),
        (
            x_pred,
            x_filt,
            P_pred,
            P_filt,
            y_pred,
            y_filt,
            innov,
            std_innov,
            S,
            eps_hat,
            loglik,
        ),
    )


def ukf_hot_loop(
    size_t meas_addr,
    double[:, ::1] hx,
    double[:, ::1] gx,
    double[:, ::1] bx,
    double[:, :, ::1] hxx,
    double[:, :, ::1] gxx,
    double[::1] hss,
    double[::1] gss,
    double[::1] steady_state,
    double[::1] params,
    double[:, ::1] Q,
    double[:, ::1] R,
    double[:, ::1] obs,
    double[::1] z0,
    double[:, ::1] P0,
    double alpha,
    double beta,
    double kappa,
    double jitter,
    bint symmetrize=True,
    bint store_history=True,
):
    """Run the native second-order UKF hot loop.

    ``meas_addr`` is a real-valued measurement ``@cfunc`` address with signature
    ``void(double* vars, double* params, double* out)``.
    """
    cdef int64_t n_state = hx.shape[0]
    cdef int64_t n_ctrl = gx.shape[0]
    cdef int64_t n_exog = bx.shape[1]
    cdef int64_t n_obs = obs.shape[1]
    cdef int64_t n_params = params.shape[0]
    cdef int64_t T = obs.shape[0]
    cdef int64_t nz = 2 * n_state
    cdef int64_t n_var = n_state + n_ctrl

    if meas_addr == 0:
        raise ValueError("ukf_hot_loop requires a nonzero "
                         "measurement function address.")
    if n_state <= 0:
        raise ValueError("hx must have at least one state.")
    if n_obs <= 0:
        raise ValueError("obs must have at least one observable column.")
    if hx.shape[1] != n_state:
        raise ValueError("hx must have shape (n_state, n_state).")
    if bx.shape[0] != n_state:
        raise ValueError("bx must have shape (n_state, n_exog).")
    if gx.shape[1] != n_state:
        raise ValueError("gx must have shape (n_ctrl, n_state).")
    if hxx.shape[0] != n_state or hxx.shape[1] != n_state or hxx.shape[2] != n_state:
        raise ValueError("hxx must have shape (n_state, n_state, n_state).")
    if gxx.shape[0] != n_ctrl or gxx.shape[1] != n_state or gxx.shape[2] != n_state:
        raise ValueError("gxx must have shape (n_ctrl, n_state, n_state).")
    if hss.shape[0] != n_state:
        raise ValueError("hss must have shape (n_state,).")
    if gss.shape[0] != n_ctrl:
        raise ValueError("gss must have shape (n_ctrl,).")
    if steady_state.shape[0] != n_var:
        raise ValueError("steady_state must have shape (n_state + n_ctrl,).")
    if Q.shape[0] != n_exog or Q.shape[1] != n_exog:
        raise ValueError("Q must have shape (n_exog, n_exog).")
    if R.shape[0] != n_obs or R.shape[1] != n_obs:
        raise ValueError("R must have shape (n_obs, n_obs).")
    if z0.shape[0] != nz:
        raise ValueError("z0 must have shape (2 * n_state,).")
    if P0.shape[0] != nz or P0.shape[1] != nz:
        raise ValueError("P0 must have shape (2 * n_state, 2 * n_state).")

    cdef int64_t hist_T = T if store_history else 0

    x1_pred = np.zeros((hist_T, n_state), dtype=np.float64)
    x2_pred = np.zeros((hist_T, n_state), dtype=np.float64)
    x1_filt = np.zeros((hist_T, n_state), dtype=np.float64)
    x2_filt = np.zeros((hist_T, n_state), dtype=np.float64)
    P_pred = np.zeros((hist_T, nz, nz), dtype=np.float64)
    P_filt = np.zeros((hist_T, nz, nz), dtype=np.float64)
    y_pred = np.zeros((hist_T, n_obs), dtype=np.float64)
    y_filt = np.zeros((hist_T, n_obs), dtype=np.float64)
    innov = np.zeros((hist_T, n_obs), dtype=np.float64)
    std_innov = np.zeros((hist_T, n_obs), dtype=np.float64)
    S = np.zeros((hist_T, n_obs, n_obs), dtype=np.float64)
    cdef double loglik = 0.0

    cdef double[:, ::1] x1_pred_mv = x1_pred
    cdef double[:, ::1] x2_pred_mv = x2_pred
    cdef double[:, ::1] x1_filt_mv = x1_filt
    cdef double[:, ::1] x2_filt_mv = x2_filt
    cdef double[:, :, ::1] P_pred_mv = P_pred
    cdef double[:, :, ::1] P_filt_mv = P_filt
    cdef double[:, ::1] y_pred_mv = y_pred
    cdef double[:, ::1] y_filt_mv = y_filt
    cdef double[:, ::1] innov_mv = innov
    cdef double[:, ::1] std_innov_mv = std_innov
    cdef double[:, :, ::1] S_mv = S

    cdef ukf_inputs inp
    inp.meas = <meas_fn><void*>meas_addr
    inp.hx = &hx[0, 0]
    inp.bx = &bx[0, 0] if n_exog > 0 else NULL
    inp.hxx = &hxx[0, 0, 0]
    inp.hss = &hss[0]
    inp.steady_state = &steady_state[0]
    inp.params = &params[0] if n_params > 0 else NULL
    inp.Q = &Q[0, 0] if n_exog > 0 else NULL
    inp.R = &R[0, 0]
    inp.obs = &obs[0, 0] if T > 0 else NULL
    inp.z0 = &z0[0]
    inp.P0 = &P0[0, 0]
    inp.T = T
    inp.n_state = n_state
    inp.n_ctrl = n_ctrl
    inp.n_exog = n_exog
    inp.n_obs = n_obs
    inp.n_params = n_params
    inp.alpha = alpha
    inp.beta = beta
    inp.kappa = kappa
    inp.jitter = jitter
    inp.symmetrize = symmetrize
    inp.store_history = store_history

    if n_ctrl > 0:
        inp.gx = &gx[0, 0]
        inp.gxx = &gxx[0, 0, 0]
        inp.gss = &gss[0]
    else:
        inp.gx = NULL
        inp.gxx = NULL
        inp.gss = NULL

    cdef ukf_outputs outp
    outp.x1_pred = &x1_pred_mv[0, 0] if hist_T > 0 else NULL
    outp.x2_pred = &x2_pred_mv[0, 0] if hist_T > 0 else NULL
    outp.x1_filt = &x1_filt_mv[0, 0] if hist_T > 0 else NULL
    outp.x2_filt = &x2_filt_mv[0, 0] if hist_T > 0 else NULL
    outp.P_pred = &P_pred_mv[0, 0, 0] if hist_T > 0 else NULL
    outp.P_filt = &P_filt_mv[0, 0, 0] if hist_T > 0 else NULL
    outp.y_pred = &y_pred_mv[0, 0] if hist_T > 0 else NULL
    outp.y_filt = &y_filt_mv[0, 0] if hist_T > 0 else NULL
    outp.innov = &innov_mv[0, 0] if hist_T > 0 else NULL
    outp.std_innov = &std_innov_mv[0, 0] if hist_T > 0 else NULL
    outp.S = &S_mv[0, 0, 0] if hist_T > 0 else NULL
    outp.loglik = &loglik

    cdef int64_t status
    with nogil:
        status = c_ukf_hot_loop(&inp, &outp)

    if status == KF_ERR_MATRIX_CONDITION:
        raise LinAlgError("UKF covariance is not positive definite.")
    if status == KF_ERR_ALLOC:
        raise MemoryError("ukf_hot_loop: scratch allocation failed.")
    if status != KF_OK:
        raise ValueError(f"ukf_hot_loop failed with status {status}.")

    return (
        KF_OK,
        (0.0, 0.0, 0.0),
        (
            x1_pred,
            x2_pred,
            x1_filt,
            x2_filt,
            P_pred,
            P_filt,
            y_pred,
            y_filt,
            innov,
            std_innov,
            S,
            loglik,
        ),
    )
