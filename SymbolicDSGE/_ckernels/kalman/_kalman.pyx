# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython shim for the filters in kalman.c"""

import numpy as np

from libc.stdint cimport int64_t

cdef extern from "sdsge_common.h":
    ctypedef struct arena_size:
        int64_t n_float
        int64_t n_int

cdef extern from "kalman.h":
    int c_KF_OK "KF_OK"
    int c_KF_ERR_SHAPE_MISMATCH "KF_ERR_SHAPE_MISMATCH"
    int c_KF_ERR_ALLOC "KF_ERR_ALLOC"
    int c_KF_ERR_LYAPUNOV_CONVERGENCE "KF_ERR_LYAPUNOV_CONVERGENCE"
    int c_KF_ERR_P_FACTORIZATION "KF_ERR_P_FACTORIZATION"
    int c_KF_ERR_Q_FACTORIZATION "KF_ERR_Q_FACTORIZATION"
    int c_KF_ERR_S_FACTORIZATION "KF_ERR_S_FACTORIZATION"
    int c_KF_ERR_BAD_HYPERPARAMS "KF_ERR_BAD_HYPERPARAMS"

    arena_size kf_stationary_covariance_arena_size(
        int64_t n,
        int64_t k,
    ) nogil

    int kf_stationary_covariance(
        const double *A,
        const double *B,
        const double *Q,
        double tolerance,
        int64_t max_iterations,
        double *arena,
        double *out,
        int64_t n,
        int64_t k,
        int64_t ld_out,
    ) nogil

    ctypedef void (*meas_fn)(
        const double *x,
        const double *params,
        double *out,
    ) noexcept nogil

cdef extern from "linear.h":
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
        double *steady_state
        const double *y
        double *x0
        double *P0
        int symmetrize
        int joseph_cov
        double jitter
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

    ctypedef struct kf_smooth_outputs:
        kf_outputs *fwd_out
        double *x_smooth
        double *P_smooth
        double *eps_smooth

    arena_size kf_arena_size(int64_t n, int64_t m, int64_t k) nogil
    int c_kf_forward "kf_forward"(
        const kf_inputs *inp,
        double *arena,
        kf_outputs *outp,
    ) nogil

    arena_size kf_smooth_arena_size(int64_t n, int64_t m,
                                    int64_t k, int64_t T) nogil
    int c_kf_smooth "kf_smooth"(
        const kf_inputs *inp,
        double *arena,
        kf_smooth_outputs *outp,
    ) nogil

cdef extern from "extended.h":
    ctypedef struct ekf_inputs:
        meas_fn meas
        meas_fn jac
        double *A
        double *B
        double *calib_params
        double *Q
        double *R
        double *steady_state
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
        int joseph_cov
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

    ctypedef struct ekf_smooth_outputs:
        ekf_outputs *fwd_out
        double *x_smooth
        double *P_smooth
        double *eps_smooth

    arena_size ekf_arena_size(int64_t n, int64_t m, int64_t k) nogil
    int c_ekf_forward "ekf_forward"(
        const ekf_inputs *inp,
        double *arena,
        ekf_outputs *outp,
    ) nogil

    arena_size ekf_smooth_arena_size(int64_t n, int64_t m,
                                     int64_t k, int64_t T) nogil
    int c_ekf_smooth "ekf_smooth"(
        const ekf_inputs *inp,
        double *arena,
        ekf_smooth_outputs *outp,
    ) nogil

cdef extern from "unscented.h":
    ctypedef struct ukf_inputs:
        meas_fn meas
        double *hx
        double *gx
        double *bu
        double *hxx
        double *gxx
        double *hxu
        double *gxu
        double *huu
        double *guu
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

    arena_size ukf_arena_size(
        int64_t n_state,
        int64_t n_ctrl,
        int64_t n_exog,
        int64_t n_obs,
    ) nogil
    int c_ukf_forward "ukf_forward"(
        const ukf_inputs *inp,
        double *arena,
        ukf_outputs *outp,
    ) nogil

KF_OK = c_KF_OK
KF_ERR_SHAPE_MISMATCH = c_KF_ERR_SHAPE_MISMATCH
KF_ERR_ALLOC = c_KF_ERR_ALLOC
KF_ERR_LYAPUNOV_CONVERGENCE = c_KF_ERR_LYAPUNOV_CONVERGENCE
KF_ERR_P_FACTORIZATION = c_KF_ERR_P_FACTORIZATION
KF_ERR_Q_FACTORIZATION = c_KF_ERR_Q_FACTORIZATION
KF_ERR_S_FACTORIZATION = c_KF_ERR_S_FACTORIZATION
KF_ERR_BAD_HYPERPARAMS = c_KF_ERR_BAD_HYPERPARAMS


cdef tuple _kf_build(
    kf_inputs *inp,
    kf_outputs *outp,
    double *loglik,
    int64_t T,
    nmk,
    A, B,
    C, d,
    Q, R,
    steady_state,
    y,
    x0, P0,
    bint symmetrize,
    bint joseph_cov,
    double jitter,
    bint store_history,
):
    """Fill the linear filter's input and output structs.

    Returns ``(inputs, history)``: the coerced inputs the structs point into,
    and the history arrays. Both must stay referenced until the kernel returns.
    """
    cdef int64_t n = nmk[0]
    cdef int64_t m = nmk[1]
    cdef int64_t k = nmk[2]

    if T <= 0:
        raise ValueError("y must have at least one row.")

    cdef int64_t hist_T = T if store_history else 0

    A_arr = np.ascontiguousarray(A, dtype=np.float64)
    B_arr = np.ascontiguousarray(B, dtype=np.float64)
    C_arr = np.ascontiguousarray(C, dtype=np.float64)
    d_arr = np.ascontiguousarray(d, dtype=np.float64)
    Q_arr = np.ascontiguousarray(Q, dtype=np.float64)
    R_arr = np.ascontiguousarray(R, dtype=np.float64)
    ss_arr = np.ascontiguousarray(steady_state, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    x0_arr = np.ascontiguousarray(x0, dtype=np.float64)
    P0_arr = np.ascontiguousarray(P0, dtype=np.float64)

    cdef double[:, ::1] Av = A_arr
    cdef double[:, ::1] Bv = B_arr
    cdef double[:, ::1] Cv = C_arr
    cdef double[::1] dv = d_arr
    cdef double[:, ::1] Qv = Q_arr
    cdef double[:, ::1] Rv = R_arr
    cdef double[::1] ssv = ss_arr
    cdef const double[:, ::1] yv = y_arr
    cdef double[::1] x0v = x0_arr
    cdef double[:, ::1] P0v = P0_arr

    x_pred = np.zeros((hist_T, n), dtype=np.float64)
    x_filt = np.zeros((hist_T, n), dtype=np.float64)
    P_pred = np.zeros((hist_T, n, n), dtype=np.float64)
    P_filt = np.zeros((hist_T, n, n), dtype=np.float64)
    y_pred = np.zeros((hist_T, m), dtype=np.float64)
    y_filt = np.zeros((hist_T, m), dtype=np.float64)
    innov = np.zeros((hist_T, m), dtype=np.float64)
    std_innov = np.zeros((hist_T, m), dtype=np.float64)
    S = np.zeros((hist_T, m, m), dtype=np.float64)
    eps_hat = np.zeros((hist_T, k), dtype=np.float64)

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
    inp.steady_state = &ssv[0]
    inp.y = &yv[0, 0]
    inp.x0 = &x0v[0]
    inp.P0 = &P0v[0, 0]
    inp.symmetrize = symmetrize
    inp.joseph_cov = joseph_cov
    inp.jitter = jitter
    inp.store_history = store_history

    outp.x_pred = &x_pred_mv[0, 0]
    outp.x_filt = &x_filt_mv[0, 0]
    outp.P_pred = &P_pred_mv[0, 0, 0]
    outp.P_filt = &P_filt_mv[0, 0, 0]
    outp.y_pred = &y_pred_mv[0, 0]
    outp.y_filt = &y_filt_mv[0, 0]
    outp.innov = &innov_mv[0, 0]
    outp.std_innov = &std_innov_mv[0, 0]
    outp.S = &S_mv[0, 0, 0]
    outp.eps_hat = &eps_hat_mv[0, 0]
    outp.loglik = loglik

    return (
        (A_arr, B_arr, C_arr, d_arr, Q_arr, R_arr, ss_arr, y_arr, x0_arr, P0_arr),
        (x_pred, x_filt, P_pred, P_filt, y_pred, y_filt, innov, std_innov, S,
         eps_hat),
    )


def kf_forward(
    int64_t T,
    nmk,
    A, B,
    C, d,
    Q, R,
    steady_state,
    y,
    x0, P0,
    bint symmetrize,
    bint joseph_cov,
    double jitter,
    bint store_history=True,
):
    """Run the linear Kalman filter."""
    cdef kf_inputs inp
    cdef kf_outputs outp
    cdef double loglik = 0.0
    _, hist = _kf_build(
        &inp, &outp, &loglik, T, nmk, A, B, C, d, Q, R, steady_state, y, x0,
        P0, symmetrize, joseph_cov, jitter, store_history,
    )

    arena = np.empty(kf_arena_size(inp.n, inp.m, inp.k).n_float, dtype=np.float64)
    cdef double[::1] arena_mv = arena

    cdef int status
    with nogil:
        status = c_kf_forward(&inp, &arena_mv[0], &outp)

    return status, (*hist, loglik)


def kf_smooth(
    int64_t T,
    nmk,
    A, B,
    C, d,
    Q, R,
    steady_state,
    y,
    x0, P0,
    bint symmetrize,
    bint joseph_cov,
    double jitter,
):
    """Run the linear Kalman filter and smoother."""
    cdef kf_inputs inp
    cdef kf_outputs outp
    cdef double loglik = 0.0
    _, hist = _kf_build(
        &inp, &outp, &loglik, T, nmk, A, B, C, d, Q, R, steady_state, y, x0,
        P0, symmetrize, joseph_cov, jitter, True,
    )
    cdef int64_t n = inp.n
    cdef int64_t k = inp.k

    x_smooth = np.zeros((T, n), dtype=np.float64)
    P_smooth = np.zeros((T, n, n), dtype=np.float64)
    eps_smooth = np.zeros((T, k), dtype=np.float64)
    cdef double[:, ::1] x_smooth_mv = x_smooth
    cdef double[:, :, ::1] P_smooth_mv = P_smooth
    cdef double[:, ::1] eps_smooth_mv = eps_smooth

    cdef kf_smooth_outputs sout
    sout.fwd_out = &outp
    sout.x_smooth = &x_smooth_mv[0, 0]
    sout.P_smooth = &P_smooth_mv[0, 0, 0]
    sout.eps_smooth = &eps_smooth_mv[0, 0]

    arena = np.empty(
        kf_smooth_arena_size(n, inp.m, k, T).n_float, dtype=np.float64
    )
    cdef double[::1] arena_mv = arena

    cdef int status
    with nogil:
        status = c_kf_smooth(&inp, &arena_mv[0], &sout)

    return status, (*hist, loglik), (x_smooth, P_smooth, eps_smooth)


def stationary_covariance(
    A,
    B,
    Q,
    double tolerance=1e-12,
    int64_t max_iterations=64,
):
    """Solve ``P = A P A.T + B Q B.T`` and return ``(status, P)``."""
    cdef double[:, ::1] Av = np.ascontiguousarray(A, dtype=np.float64)
    cdef double[:, ::1] Bv = np.ascontiguousarray(B, dtype=np.float64)
    cdef double[:, ::1] Qv = np.ascontiguousarray(Q, dtype=np.float64)
    cdef int64_t n = Av.shape[0]
    cdef int64_t k = Bv.shape[1]

    if n <= 0:
        raise ValueError("A must have positive state dimensions.")
    if Av.shape[1] != n:
        raise ValueError("A must be square.")
    if Bv.shape[0] != n:
        raise ValueError("B must have the same row count as A.")
    if Qv.shape[0] != k or Qv.shape[1] != k:
        raise ValueError("Q must be square with dimension B.shape[1].")

    P = np.empty((n, n), dtype=np.float64)
    arena = np.empty(
        kf_stationary_covariance_arena_size(n, k).n_float,
        dtype=np.float64,
    )

    cdef double[:, ::1] Pv = P
    cdef double[::1] arena_mv = arena
    cdef int status
    with nogil:
        status = kf_stationary_covariance(
            &Av[0, 0],
            &Bv[0, 0],
            &Qv[0, 0],
            tolerance,
            max_iterations,
            &arena_mv[0],
            &Pv[0, 0],
            n,
            k,
            n,
        )
    return status, P


cdef tuple _ekf_build(
    ekf_inputs *inp,
    ekf_outputs *outp,
    double *loglik,
    size_t meas_addr,
    size_t jac_addr,
    A, B,
    calib_params,
    Q, R,
    steady_state,
    y,
    x0, P0,
    bint symmetrize,
    bint joseph_cov,
    double jitter,
    bint store_history,
):
    """Fill the extended filter's input and output structs.

    Returns ``(inputs, history)``: the coerced inputs the structs point into,
    and the history arrays. Both must stay referenced until the kernel returns.
    """
    cdef int64_t n = A.shape[0]
    cdef int64_t k = B.shape[1]
    cdef int64_t T = y.shape[0]
    cdef int64_t m = y.shape[1]
    cdef int64_t n_par = calib_params.shape[0]

    if meas_addr == 0 or jac_addr == 0:
        raise ValueError("The extended filter requires nonzero measurement "
                         "and jacobian function addresses.")
    if T <= 0:
        raise ValueError("y must have at least one row.")

    cdef int64_t hist_T = T if store_history else 0

    A_arr = np.ascontiguousarray(A, dtype=np.float64)
    B_arr = np.ascontiguousarray(B, dtype=np.float64)
    Q_arr = np.ascontiguousarray(Q, dtype=np.float64)
    R_arr = np.ascontiguousarray(R, dtype=np.float64)
    ss_arr = np.ascontiguousarray(steady_state, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    x0_arr = np.ascontiguousarray(x0, dtype=np.float64)
    P0_arr = np.ascontiguousarray(P0, dtype=np.float64)
    calib_arr = np.ascontiguousarray(calib_params, dtype=np.float64)

    cdef double[:, ::1] Av = A_arr
    cdef double[:, ::1] Bv = B_arr
    cdef double[:, ::1] Qv = Q_arr
    cdef double[:, ::1] Rv = R_arr
    cdef double[::1] ssv = ss_arr
    cdef const double[:, ::1] yv = y_arr
    cdef double[::1] x0v = x0_arr
    cdef double[:, ::1] P0v = P0_arr
    cdef double[::1] calib_paramsv = calib_arr

    x_pred = np.zeros((hist_T, n), dtype=np.float64)
    x_filt = np.zeros((hist_T, n), dtype=np.float64)
    P_pred = np.zeros((hist_T, n, n), dtype=np.float64)
    P_filt = np.zeros((hist_T, n, n), dtype=np.float64)
    y_pred = np.zeros((hist_T, m), dtype=np.float64)
    y_filt = np.zeros((hist_T, m), dtype=np.float64)
    innov = np.zeros((hist_T, m), dtype=np.float64)
    std_innov = np.zeros((hist_T, m), dtype=np.float64)
    S = np.zeros((hist_T, m, m), dtype=np.float64)
    eps_hat = np.zeros((hist_T, k), dtype=np.float64)

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

    inp.meas = <meas_fn><void*>meas_addr
    inp.jac = <meas_fn><void*>jac_addr
    inp.A = &Av[0, 0]
    inp.B = &Bv[0, 0]
    inp.calib_params = &calib_paramsv[0] if n_par > 0 else NULL
    inp.Q = &Qv[0, 0]
    inp.R = &Rv[0, 0]
    inp.steady_state = &ssv[0]
    inp.y = &yv[0, 0]
    inp.x0 = &x0v[0]
    inp.P0 = &P0v[0, 0]
    inp.T = T
    inp.n = n
    inp.m = m
    inp.k = k
    inp.n_par = n_par
    inp.jitter = jitter
    inp.symmetrize = symmetrize
    inp.joseph_cov = joseph_cov
    inp.store_history = store_history

    outp.x_pred = &x_pred_mv[0, 0]
    outp.x_filt = &x_filt_mv[0, 0]
    outp.P_pred = &P_pred_mv[0, 0, 0]
    outp.P_filt = &P_filt_mv[0, 0, 0]
    outp.y_pred = &y_pred_mv[0, 0]
    outp.y_filt = &y_filt_mv[0, 0]
    outp.innov = &innov_mv[0, 0]
    outp.std_innov = &std_innov_mv[0, 0]
    outp.S = &S_mv[0, 0, 0]
    outp.eps_hat = &eps_hat_mv[0, 0]
    outp.loglik = loglik

    return (
        (A_arr, B_arr, Q_arr, R_arr, ss_arr, y_arr, x0_arr, P0_arr, calib_arr),
        (x_pred, x_filt, P_pred, P_filt, y_pred, y_filt, innov, std_innov, S,
         eps_hat),
    )


def ekf_forward(
    size_t meas_addr,
    size_t jac_addr,
    A, B,
    calib_params,
    Q, R,
    steady_state,
    y,
    x0, P0,
    bint symmetrize,
    bint joseph_cov,
    double jitter,
    bint store_history=True,
):
    """Run the native extended Kalman filter (linear transition, nonlinear
    measurement via the ``meas``/``jac`` @cfunc addresses)."""
    cdef ekf_inputs inp
    cdef ekf_outputs outp
    cdef double loglik = 0.0
    _, hist = _ekf_build(
        &inp, &outp, &loglik, meas_addr, jac_addr, A, B, calib_params, Q, R,
        steady_state, y, x0, P0, symmetrize, joseph_cov, jitter, store_history,
    )

    arena = np.empty(ekf_arena_size(inp.n, inp.m, inp.k).n_float, dtype=np.float64)
    cdef double[::1] arena_mv = arena

    cdef int status
    with nogil:
        status = c_ekf_forward(&inp, &arena_mv[0], &outp)

    return status, (*hist, loglik)


def ekf_smooth(
    size_t meas_addr,
    size_t jac_addr,
    A, B,
    calib_params,
    Q, R,
    steady_state,
    y,
    x0, P0,
    bint symmetrize,
    bint joseph_cov,
    double jitter,
):
    """Run the extended Kalman filter and smoother."""
    cdef ekf_inputs inp
    cdef ekf_outputs outp
    cdef double loglik = 0.0
    _, hist = _ekf_build(
        &inp, &outp, &loglik, meas_addr, jac_addr, A, B, calib_params, Q, R,
        steady_state, y, x0, P0, symmetrize, joseph_cov, jitter, True,
    )
    cdef int64_t n = inp.n
    cdef int64_t k = inp.k
    cdef int64_t T = inp.T

    x_smooth = np.zeros((T, n), dtype=np.float64)
    P_smooth = np.zeros((T, n, n), dtype=np.float64)
    eps_smooth = np.zeros((T, k), dtype=np.float64)
    cdef double[:, ::1] x_smooth_mv = x_smooth
    cdef double[:, :, ::1] P_smooth_mv = P_smooth
    cdef double[:, ::1] eps_smooth_mv = eps_smooth

    cdef ekf_smooth_outputs sout
    sout.fwd_out = &outp
    sout.x_smooth = &x_smooth_mv[0, 0]
    sout.P_smooth = &P_smooth_mv[0, 0, 0]
    sout.eps_smooth = &eps_smooth_mv[0, 0]

    arena = np.empty(
        ekf_smooth_arena_size(n, inp.m, k, T).n_float, dtype=np.float64
    )
    cdef double[::1] arena_mv = arena

    cdef int status
    with nogil:
        status = c_ekf_smooth(&inp, &arena_mv[0], &sout)

    return status, (*hist, loglik), (x_smooth, P_smooth, eps_smooth)


def ukf_forward(
    size_t meas_addr,
    hx,
    gx,
    bu,
    hxx,
    gxx,
    hxu,
    gxu,
    huu,
    guu,
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

    ``bu`` spans every variable rather than the state block: a control responds
    to an innovation contemporaneously. ``hxu``/``gxu`` and ``huu``/``guu`` are
    the quadratic blocks the shocks earn once they are no longer states.
    """
    cdef double[:, ::1] hxv = np.ascontiguousarray(hx, dtype=np.float64)
    cdef double[:, ::1] gxv = np.ascontiguousarray(gx, dtype=np.float64)
    cdef double[:, ::1] buv = np.ascontiguousarray(bu, dtype=np.float64)
    cdef double[:, :, ::1] hxxv = np.ascontiguousarray(hxx, dtype=np.float64)
    cdef double[:, :, ::1] gxxv = np.ascontiguousarray(gxx, dtype=np.float64)
    cdef double[:, :, ::1] hxuv = np.ascontiguousarray(hxu, dtype=np.float64)
    cdef double[:, :, ::1] gxuv = np.ascontiguousarray(gxu, dtype=np.float64)
    cdef double[:, :, ::1] huuv = np.ascontiguousarray(huu, dtype=np.float64)
    cdef double[:, :, ::1] guuv = np.ascontiguousarray(guu, dtype=np.float64)
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
    cdef int64_t n_exog = buv.shape[1]
    cdef int64_t n_obs = obsv.shape[1]
    cdef int64_t n_params = paramsv.shape[0]
    cdef int64_t T = obsv.shape[0]
    cdef int64_t nz = 2 * n_state
    cdef int64_t n_var = n_state + n_ctrl

    if meas_addr == 0:
        raise ValueError("ukf_forward requires a nonzero "
                         "measurement function address.")
    if n_state <= 0:
        raise ValueError("hx must have at least one state.")
    if n_obs <= 0:
        raise ValueError("obs must have at least one observable column.")
    if T <= 0:
        raise ValueError("obs must have at least one row.")
    if hxv.shape[1] != n_state:
        raise ValueError("hx must have shape (n_state, n_state).")
    if buv.shape[0] != n_var:
        raise ValueError("bu must have shape (n_state + n_ctrl, n_exog).")
    if gxv.shape[1] != n_state:
        raise ValueError("gx must have shape (n_ctrl, n_state).")
    if hxxv.shape[0] != n_state or hxxv.shape[1] != n_state or hxxv.shape[2] != n_state:
        raise ValueError("hxx must have shape (n_state, n_state, n_state).")
    if gxxv.shape[0] != n_ctrl or gxxv.shape[1] != n_state or gxxv.shape[2] != n_state:
        raise ValueError("gxx must have shape (n_ctrl, n_state, n_state).")
    if hxuv.shape[0] != n_state or hxuv.shape[1] != n_state or hxuv.shape[2] != n_exog:
        raise ValueError("hxu must have shape (n_state, n_state, n_exog).")
    if gxuv.shape[0] != n_ctrl or gxuv.shape[1] != n_state or gxuv.shape[2] != n_exog:
        raise ValueError("gxu must have shape (n_ctrl, n_state, n_exog).")
    if huuv.shape[0] != n_state or huuv.shape[1] != n_exog or huuv.shape[2] != n_exog:
        raise ValueError("huu must have shape (n_state, n_exog, n_exog).")
    if guuv.shape[0] != n_ctrl or guuv.shape[1] != n_exog or guuv.shape[2] != n_exog:
        raise ValueError("guu must have shape (n_ctrl, n_exog, n_exog).")
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
    arena = np.empty(
        ukf_arena_size(n_state, n_ctrl, n_exog, n_obs).n_float, dtype=np.float64
    )
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
    cdef double[::1] arena_mv = arena

    cdef ukf_inputs inp
    inp.meas = <meas_fn><void*>meas_addr
    inp.hx = &hxv[0, 0]
    inp.bu = &buv[0, 0] if n_exog > 0 else NULL
    inp.hxx = &hxxv[0, 0, 0]
    inp.hxu = &hxuv[0, 0, 0] if n_exog > 0 else NULL
    inp.huu = &huuv[0, 0, 0] if n_exog > 0 else NULL
    inp.hss = &hssv[0]
    inp.steady_state = &steady_statev[0]
    inp.params = &paramsv[0] if n_params > 0 else NULL
    inp.Q = &Qv[0, 0] if n_exog > 0 else NULL
    inp.R = &Rv[0, 0]
    inp.obs = &obsv[0, 0]
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
        inp.gxu = &gxuv[0, 0, 0] if n_exog > 0 else NULL
        inp.guu = &guuv[0, 0, 0] if n_exog > 0 else NULL
    else:
        inp.gx = NULL
        inp.gxx = NULL
        inp.gss = NULL
        inp.gxu = NULL
        inp.guu = NULL

    cdef ukf_outputs outp
    outp.x1_pred = &x1_pred_mv[0, 0]
    outp.x2_pred = &x2_pred_mv[0, 0]
    outp.x1_filt = &x1_filt_mv[0, 0]
    outp.x2_filt = &x2_filt_mv[0, 0]
    outp.x_pred = &x_pred_mv[0, 0]
    outp.x_filt = &x_filt_mv[0, 0]
    outp.P_pred = &P_pred_mv[0, 0, 0]
    outp.P_filt = &P_filt_mv[0, 0, 0]
    outp.y_pred = &y_pred_mv[0, 0]
    outp.y_filt = &y_filt_mv[0, 0]
    outp.innov = &innov_mv[0, 0]
    outp.std_innov = &std_innov_mv[0, 0]
    outp.S = &S_mv[0, 0, 0]
    outp.loglik = &loglik

    cdef int status
    with nogil:
        status = c_ukf_forward(&inp, &arena_mv[0], &outp)

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
