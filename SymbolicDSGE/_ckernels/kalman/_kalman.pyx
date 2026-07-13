# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython shim for the linear Kalman filter hot loop.

Allocates the history output arrays, marshals the NumPy buffers into the C
kf_inputs/kf_outputs structs, runs kf_hot_loop under nogil, and translates the
status code back into the exceptions KalmanFilter.run already handles. The
public signature and return shape match the numba `_kalman_hot_loop` so the two
are interchangeable at the call site.
"""

import numpy as np

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
        const double *y
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

    ctypedef struct ekf_inputs:
        meas_fn meas
        meas_fn jac
        double *A
        double *B
        double *calib_params
        double *Q
        double *R
        const double *y
        double *x0
        double *P0
        int64_t T
        int64_t n
        int64_t m
        int64_t k
        int64_t n_par
        double jitter
        int symmetrize
        int compute_y_filt
        int return_shocks
        int store_history

    ctypedef struct ekf_outputs:
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

    int c_ekf_hot_loop "ekf_hot_loop"(
        const ekf_inputs *inp,
        ekf_outputs *outp,
    ) nogil

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
        const double *obs
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

        double *x_pred
        double *x_filt

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
    A, B,
    C, d,
    Q, R,
    y,
    x0, P0,
    bint symmetrize,
    double jitter,
    bint return_shocks=False,
    bint store_history=True,
):
    """Run the linear Kalman filter.

    Array inputs are coerced to contiguous float64 here. Returns
    ``(status, out)`` where ``status`` is a ``kalman.errors`` code (KF_OK on
    success) and ``out`` is the history tuple + loglik; the caller maps a nonzero
    status to the matching in-house exception.
    """
    cdef int64_t n = nmk[0]
    cdef int64_t m = nmk[1]
    cdef int64_t k = nmk[2]

    cdef int64_t hist_T = T if store_history else 0
    cdef int64_t shock_T = T if (return_shocks and store_history) else 0

    cdef double[:, ::1] Av = np.ascontiguousarray(A, dtype=np.float64)
    cdef double[:, ::1] Bv = np.ascontiguousarray(B, dtype=np.float64)
    cdef double[:, ::1] Cv = np.ascontiguousarray(C, dtype=np.float64)
    cdef double[::1] dv = np.ascontiguousarray(d, dtype=np.float64)

    cdef double[:, ::1] Qv = np.ascontiguousarray(Q, dtype=np.float64)
    cdef double[:, ::1] Rv = np.ascontiguousarray(R, dtype=np.float64)
    cdef const double[:, ::1] yv = np.ascontiguousarray(y, dtype=np.float64)

    cdef double[::1] x0v = np.ascontiguousarray(x0, dtype=np.float64)
    cdef double[:, ::1] P0v = np.ascontiguousarray(P0, dtype=np.float64)

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
    inp.A = &Av[0, 0]
    inp.B = &Bv[0, 0]
    inp.C = &Cv[0, 0]
    inp.d = &dv[0]
    inp.Q = &Qv[0, 0]
    inp.R = &Rv[0, 0]
    inp.y = &yv[0, 0] if T > 0 else NULL
    inp.x0 = &x0v[0]
    inp.P0 = &P0v[0, 0]
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

    return (
        status,
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


def ekf_hot_loop(
    size_t meas_addr,
    size_t jac_addr,
    A, B,
    calib_params,
    Q, R,
    y,
    x0, P0,
    bint symmetrize,
    double jitter,
    bint compute_y_filt=True,
    bint return_shocks=False,
    bint store_history=True,
):
    """Run the native extended Kalman filter (linear transition, nonlinear
    measurement via the ``meas``/``jac`` @cfunc addresses).

    ``meas_addr``/``jac_addr`` are real-valued @cfunc addresses with signature
    ``void(double* x, double* params, double* out)`` -- ``meas`` writes
    ``out[m]``, ``jac`` writes the (m, n) row-major jacobian. Array inputs are
    coerced to contiguous float64 here. Returns ``(status, out)`` with the
    history tuple + loglik; the caller maps a nonzero status to the matching
    in-house exception.
    """
    cdef int64_t n = A.shape[0]
    cdef int64_t k = B.shape[1]
    cdef int64_t T = y.shape[0]
    cdef int64_t m = y.shape[1]
    cdef int64_t n_par = calib_params.shape[0]

    if meas_addr == 0 or jac_addr == 0:
        raise ValueError("ekf_hot_loop requires nonzero measurement and "
                         "jacobian function addresses.")

    cdef int64_t hist_T = T if store_history else 0
    cdef int64_t shock_T = T if (return_shocks and store_history) else 0

    cdef double[:, ::1] Av = np.ascontiguousarray(A, dtype=np.float64)
    cdef double[:, ::1] Bv = np.ascontiguousarray(B, dtype=np.float64)
    cdef double[:, ::1] Qv = np.ascontiguousarray(Q, dtype=np.float64)
    cdef double[:, ::1] Rv = np.ascontiguousarray(R, dtype=np.float64)

    cdef const double[:, ::1] yv = np.ascontiguousarray(y, dtype=np.float64)

    cdef double[::1] x0v = np.ascontiguousarray(x0, dtype=np.float64)
    cdef double[:, ::1] P0v = np.ascontiguousarray(P0, dtype=np.float64)

    cdef double[::1] calib_paramsv = np.ascontiguousarray(
            calib_params,
            dtype=np.float64,
    )

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

    cdef ekf_inputs inp
    inp.meas = <meas_fn><void*>meas_addr
    inp.jac = <meas_fn><void*>jac_addr
    inp.A = &Av[0, 0]
    inp.B = &Bv[0, 0]
    inp.calib_params = &calib_paramsv[0] if n_par > 0 else NULL
    inp.Q = &Qv[0, 0]
    inp.R = &Rv[0, 0]
    inp.y = &yv[0, 0] if T > 0 else NULL
    inp.x0 = &x0v[0]
    inp.P0 = &P0v[0, 0]
    inp.T = T
    inp.n = n
    inp.m = m
    inp.k = k
    inp.n_par = n_par
    inp.jitter = jitter
    inp.symmetrize = symmetrize
    inp.compute_y_filt = compute_y_filt
    inp.return_shocks = return_shocks
    inp.store_history = store_history

    cdef ekf_outputs outp
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

    cdef int status
    with nogil:
        status = c_ekf_hot_loop(&inp, &outp)

    return (
        status,
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
    hx,
    gx,
    bx,
    hxx,
    gxx,
    hss,
    gss,
    steady_state,
    params,
    Q,
    R,
    obs,
    z0,
    P0,
    double alpha,
    double beta,
    double kappa,
    double jitter,
    bint symmetrize=True,
    bint store_history=True,
):
    """Run the native second-order UKF hot loop.

    ``meas_addr`` is a real-valued measurement ``@cfunc`` address with signature
    ``void(double* vars, double* params, double* out)``. Array inputs are coerced
    to contiguous float64 here. Returns ``(status, out)``; the caller maps a
    nonzero status to the matching in-house exception.
    """
    cdef double[:, ::1] hxv = np.ascontiguousarray(hx, dtype=np.float64)
    cdef double[:, ::1] gxv = np.ascontiguousarray(gx, dtype=np.float64)
    cdef double[:, ::1] bxv = np.ascontiguousarray(bx, dtype=np.float64)
    cdef double[:, :, ::1] hxxv = np.ascontiguousarray(hxx, dtype=np.float64)
    cdef double[:, :, ::1] gxxv = np.ascontiguousarray(gxx, dtype=np.float64)
    cdef double[::1] hssv = np.ascontiguousarray(hss, dtype=np.float64)
    cdef double[::1] gssv = np.ascontiguousarray(gss, dtype=np.float64)
    cdef double[::1] steady_statev = np.ascontiguousarray(
        steady_state, dtype=np.float64)
    cdef double[::1] paramsv = np.ascontiguousarray(params, dtype=np.float64)
    cdef double[:, ::1] Qv = np.ascontiguousarray(Q, dtype=np.float64)
    cdef double[:, ::1] Rv = np.ascontiguousarray(R, dtype=np.float64)
    cdef const double[:, ::1] obsv = np.ascontiguousarray(obs, dtype=np.float64)
    cdef double[::1] z0v = np.ascontiguousarray(z0, dtype=np.float64)
    cdef double[:, ::1] P0v = np.ascontiguousarray(P0, dtype=np.float64)

    cdef int64_t n_state = hxv.shape[0]
    cdef int64_t n_ctrl = gxv.shape[0]
    cdef int64_t n_exog = bxv.shape[1]
    cdef int64_t n_obs = obsv.shape[1]
    cdef int64_t n_params = paramsv.shape[0]
    cdef int64_t T = obsv.shape[0]
    cdef int64_t nz = 2 * n_state
    cdef int64_t n_var = n_state + n_ctrl

    if meas_addr == 0:
        raise ValueError("ukf_hot_loop requires a nonzero "
                         "measurement function address.")
    if n_state <= 0:
        raise ValueError("hx must have at least one state.")
    if n_obs <= 0:
        raise ValueError("obs must have at least one observable column.")
    if hxv.shape[1] != n_state:
        raise ValueError("hx must have shape (n_state, n_state).")
    if bxv.shape[0] != n_state:
        raise ValueError("bx must have shape (n_state, n_exog).")
    if gxv.shape[1] != n_state:
        raise ValueError("gx must have shape (n_ctrl, n_state).")
    if hxxv.shape[0] != n_state or hxxv.shape[1] != n_state or hxxv.shape[2] != n_state:
        raise ValueError("hxx must have shape (n_state, n_state, n_state).")
    if gxxv.shape[0] != n_ctrl or gxxv.shape[1] != n_state or gxxv.shape[2] != n_state:
        raise ValueError("gxx must have shape (n_ctrl, n_state, n_state).")
    if hssv.shape[0] != n_state:
        raise ValueError("hss must have shape (n_state,).")
    if gssv.shape[0] != n_ctrl:
        raise ValueError("gss must have shape (n_ctrl,).")
    if steady_statev.shape[0] != n_var:
        raise ValueError("steady_state must have shape (n_state + n_ctrl,).")
    if Qv.shape[0] != n_exog or Qv.shape[1] != n_exog:
        raise ValueError("Q must have shape (n_exog, n_exog).")
    if Rv.shape[0] != n_obs or Rv.shape[1] != n_obs:
        raise ValueError("R must have shape (n_obs, n_obs).")
    if z0v.shape[0] != nz:
        raise ValueError("z0 must have shape (2 * n_state,).")
    if P0v.shape[0] != nz or P0v.shape[1] != nz:
        raise ValueError("P0 must have shape (2 * n_state, 2 * n_state).")

    cdef int64_t hist_T = T if store_history else 0

    x1_pred = np.zeros((hist_T, n_state), dtype=np.float64)
    x2_pred = np.zeros((hist_T, n_state), dtype=np.float64)
    x1_filt = np.zeros((hist_T, n_state), dtype=np.float64)
    x2_filt = np.zeros((hist_T, n_state), dtype=np.float64)
    x_pred = np.zeros((hist_T, n_var), dtype=np.float64)
    x_filt = np.zeros((hist_T, n_var), dtype=np.float64)
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
    cdef double[:, ::1] x_pred_mv = x_pred
    cdef double[:, ::1] x_filt_mv = x_filt
    cdef double[:, :, ::1] P_pred_mv = P_pred
    cdef double[:, :, ::1] P_filt_mv = P_filt
    cdef double[:, ::1] y_pred_mv = y_pred
    cdef double[:, ::1] y_filt_mv = y_filt
    cdef double[:, ::1] innov_mv = innov
    cdef double[:, ::1] std_innov_mv = std_innov
    cdef double[:, :, ::1] S_mv = S

    cdef ukf_inputs inp
    inp.meas = <meas_fn><void*>meas_addr
    inp.hx = &hxv[0, 0]
    inp.bx = &bxv[0, 0] if n_exog > 0 else NULL
    inp.hxx = &hxxv[0, 0, 0]
    inp.hss = &hssv[0]
    inp.steady_state = &steady_statev[0]
    inp.params = &paramsv[0] if n_params > 0 else NULL
    inp.Q = &Qv[0, 0] if n_exog > 0 else NULL
    inp.R = &Rv[0, 0]
    inp.obs = &obsv[0, 0] if T > 0 else NULL
    inp.z0 = &z0v[0]
    inp.P0 = &P0v[0, 0]
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
        inp.gx = &gxv[0, 0]
        inp.gxx = &gxxv[0, 0, 0]
        inp.gss = &gssv[0]
    else:
        inp.gx = NULL
        inp.gxx = NULL
        inp.gss = NULL

    cdef ukf_outputs outp
    outp.x1_pred = &x1_pred_mv[0, 0] if hist_T > 0 else NULL
    outp.x2_pred = &x2_pred_mv[0, 0] if hist_T > 0 else NULL
    outp.x1_filt = &x1_filt_mv[0, 0] if hist_T > 0 else NULL
    outp.x2_filt = &x2_filt_mv[0, 0] if hist_T > 0 else NULL
    outp.x_pred = &x_pred_mv[0, 0] if hist_T > 0 else NULL
    outp.x_filt = &x_filt_mv[0, 0] if hist_T > 0 else NULL
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

    return (
        status,
        (
            x1_pred,
            x2_pred,
            x1_filt,
            x2_filt,
            x_pred,
            x_filt,
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
