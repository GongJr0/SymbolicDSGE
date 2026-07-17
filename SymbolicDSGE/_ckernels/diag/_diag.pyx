# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Thin Cython shim mapping NumPy buffers to the pure-C diagnostic-test kernels.

No numeric logic here -- only buffer->pointer marshalling and the GIL release.
The algorithms live in diag.c. Each ``def`` mirrors the matching numba kernel in
SymbolicDSGE/_diag_tests/ and returns the same (status, ...) tuple shape. A
status of ``DIAG_FALLBACK`` means the design was rank-deficient and the caller
must re-run the whole statistic via numba (which has the lstsq/SVD fallback).
"""

import numpy as np

from libc.stdint cimport int64_t


cdef extern from "diag.h":
    int DIAG_FALLBACK

    int sdsge_bg_stat(const double *eps, const double *X, int64_t n, int64_t K,
                      int64_t lags, double *stat_out) nogil
    int sdsge_bp_aux(const double *eps, const double *X_aug, int64_t n,
                     int64_t p, double *rss_out, double *tss_out) nogil
    int sdsge_chow_stat(const double *y, const double *X, int64_t T, int64_t p,
                        int64_t t_break, double *stat_out) nogil
    int sdsge_recursive_residuals(const double *y, const double *X, int64_t T,
                                  int64_t p, double *w_out) nogil
    int sdsge_cusum_series(const double *y, const double *X, int64_t T,
                           int64_t p, double *series_out) nogil
    int sdsge_cusum_stat(const double *y, const double *X, int64_t T,
                         int64_t p, double *stat_out) nogil
    int sdsge_cusumsq_stat(const double *y, const double *X, int64_t T,
                           int64_t p, int64_t *n_out, double *stat_out) nogil
    int sdsge_acorr(const double *x, const int64_t n, const int64_t L,
                    double *z_scratch, double *out) nogil
    int sdsge_lb_stat(const double *x, const int64_t n, int64_t L,
                      double *z_scratch, double *acorr_scratch,
                      double *out) nogil
    int sdsge_jb_stat(const double *x, int64_t n, double *out) nogil
cdef extern from "diag_wald.h":
    ctypedef enum KernelID:
        BARTLETT
        PARZEN
        QS
        KERNEL_COUNT

    void sdsge_fill_mean_ax0(const double *x, int64_t n, int64_t p,
                             double *mean) nogil
    void sdsge_fill_centered_ax0(const double *x, const double *mean, int64_t n,
                                 int64_t p, double *centered) nogil

    void sdsge_hac_estimator_matmul(double *r, KernelID kernel_id, int64_t L,
                                    int64_t n, int64_t p, double *gamma_scratch,
                                    double *out) nogil

    int sdsge_wald_stat_from_mean_and_cov(const double *mean,
                                          const double *target,
                                          const double *omega, int64_t n,
                                          int64_t p, double *dev_scratch,
                                          double *L_scratch,
                                          double *stat_out) nogil

    int sdsge_symmetric_outer_prod_2dim(const double *x, int64_t n, int64_t p,
                                        int64_t q, double *out) nogil

    int sdsge_fill_symmetric_target_vec(const double *target, double atol,
                                        double rtol, int64_t p,
                                        double *out) nogil

# Re-exported so the Python dispatch layer can recognise the "retry in numba"
# signal without hard-coding the magic number.
FALLBACK = DIAG_FALLBACK


def bg_stat(eps, X, int64_t lags):
    """Breusch-Godfrey LM statistic. Returns (status, stat)."""
    cdef double[::1] eps_mv = np.ascontiguousarray(eps, dtype=np.float64)
    cdef double[:, ::1] X_mv = np.ascontiguousarray(X, dtype=np.float64)

    cdef int64_t n = eps_mv.shape[0]
    cdef int64_t K = X_mv.shape[1]
    cdef double stat = 0.0
    cdef int status
    with nogil:
        status = sdsge_bg_stat(&eps_mv[0], &X_mv[0, 0], n, K, lags, &stat)
    return status, stat


def bp_aux(eps, X_aug):
    """Breusch-Pagan auxiliary regression. Returns (status, rss, tss)."""
    cdef double[::1] eps_mv = np.ascontiguousarray(eps, dtype=np.float64)
    cdef double[:, ::1] X_aug_mv = np.ascontiguousarray(X_aug, dtype=np.float64)

    cdef int64_t n = X_aug_mv.shape[0]
    cdef int64_t p = X_aug_mv.shape[1]
    cdef double rss = 0.0
    cdef double tss = 0.0
    cdef int status
    with nogil:
        status = sdsge_bp_aux(&eps_mv[0], &X_aug_mv[0, 0], n, p, &rss, &tss)
    return status, rss, tss


def chow_stat(y, X, int64_t t_break):
    """Chow break-point F statistic. Returns (status, stat)."""
    cdef double[::1] y_mv = np.ascontiguousarray(y, dtype=np.float64)
    cdef double[:, ::1] X_mv = np.ascontiguousarray(X, dtype=np.float64)

    cdef int64_t T = X_mv.shape[0]
    cdef int64_t p = X_mv.shape[1]
    cdef double stat = 0.0
    cdef int status
    with nogil:
        status = sdsge_chow_stat(&y_mv[0], &X_mv[0, 0], T, p, t_break, &stat)
    return status, stat


def recursive_residuals(y, X):
    """Brown-Durbin-Evans recursive residuals. Returns (status, w)."""
    cdef double[::1] y_mv = np.ascontiguousarray(y, dtype=np.float64)
    cdef double[:, ::1] X_mv = np.ascontiguousarray(X, dtype=np.float64)

    cdef int64_t T = X_mv.shape[0]
    cdef int64_t p = X_mv.shape[1]
    cdef int64_t w_len = T - p if T > p else 0
    w = np.empty(w_len, dtype=np.float64)
    cdef double[::1] w_mv = w
    cdef double *w_ptr = &w_mv[0] if w_len > 0 else NULL
    cdef int status
    with nogil:
        status = sdsge_recursive_residuals(&y_mv[0], &X_mv[0, 0], T, p, w_ptr)
    return status, w


def cusum_series(y, X):
    """Standardized CUSUM series. Returns (status, series)."""
    cdef double[::1] y_mv = np.ascontiguousarray(y, dtype=np.float64)
    cdef double[:, ::1] X_mv = np.ascontiguousarray(X, dtype=np.float64)

    cdef int64_t T = X_mv.shape[0]
    cdef int64_t p = X_mv.shape[1]
    cdef int64_t s_len = T - p if T > p else 0
    series = np.empty(s_len, dtype=np.float64)
    cdef double[::1] s_mv = series
    cdef double *s_ptr = &s_mv[0] if s_len > 0 else NULL
    cdef int status
    with nogil:
        status = sdsge_cusum_series(&y_mv[0], &X_mv[0, 0], T, p, s_ptr)
    return status, series


def cusum_stat(y, X):
    """CUSUM statistic. Returns (status, stat)."""
    cdef double[::1] y_mv = np.ascontiguousarray(y, dtype=np.float64)
    cdef double[:, ::1] X_mv = np.ascontiguousarray(X, dtype=np.float64)

    cdef int64_t T = X_mv.shape[0]
    cdef int64_t p = X_mv.shape[1]
    cdef double stat = 0.0
    cdef int status
    with nogil:
        status = sdsge_cusum_stat(&y_mv[0], &X_mv[0, 0], T, p, &stat)
    return status, stat


def cusumsq_stat(y, X):
    """CUSUM-of-squares statistic. Returns (status, n, stat)."""
    cdef double[::1] y_mv = np.ascontiguousarray(y, dtype=np.float64)
    cdef double[:, ::1] X_mv = np.ascontiguousarray(X, dtype=np.float64)

    cdef int64_t T = X_mv.shape[0]
    cdef int64_t p = X_mv.shape[1]
    cdef int64_t n = 0
    cdef double stat = 0.0
    cdef int status
    with nogil:
        status = sdsge_cusumsq_stat(&y_mv[0], &X_mv[0, 0], T, p, &n, &stat)
    return status, n, stat


def jb_stat(x):
    """Jarque-Bera normality statistic. Returns (status, stat)."""
    cdef double[::1] x_mv = np.ascontiguousarray(x, dtype=np.float64)

    cdef int64_t n = x_mv.shape[0]
    cdef double stat = 0.0
    cdef int status
    cdef double *x_ptr = &x_mv[0] if n > 0 else NULL
    with nogil:
        status = sdsge_jb_stat(x_ptr, n, &stat)
    return status, np.float64(stat)


def acorr(x, int64_t L):
    """Autocorrelation of x up to lag L. Returns (status, out(L+1)).

    Mirrors the numba ``acorr`` (no L/n clamping here); ``out`` is length L+1 and
    z_scratch is length n. status is DIAG_UDEF_VARIANCE for a constant series.
    """
    cdef double[::1] x_mv = np.ascontiguousarray(x, dtype=np.float64)

    cdef int64_t n = x.shape[0]
    out = np.empty(L + 1, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)
    cdef double[::1] out_mv = out
    cdef double[::1] z_mv = z
    cdef double *x_ptr = &x_mv[0] if n > 0 else NULL
    cdef double *z_ptr = &z_mv[0] if n > 0 else NULL
    cdef int status
    with nogil:
        status = sdsge_acorr(x_ptr, n, L, z_ptr, &out_mv[0])
    return status, out


def lb_stat(x, int64_t L):
    """Ljung-Box statistic for x up to lag L. Returns (status, stat).

    The kernel clamps L to n-1 and validates n/L internally; the two length-n
    scratch buffers cover the clamped lag (L_eff <= n-1, so L_eff+1 <= n).
    """
    cdef double[::1] x_mv = np.ascontiguousarray(x, dtype=np.float64)

    cdef int64_t n = x_mv.shape[0]
    cdef double stat = 0.0
    cdef int status
    z = np.empty(n, dtype=np.float64)
    rho = np.empty(n, dtype=np.float64)
    cdef double[::1] z_mv = z
    cdef double[::1] rho_mv = rho
    cdef double *x_ptr = &x_mv[0] if n > 0 else NULL
    cdef double *z_ptr = &z_mv[0] if n > 0 else NULL
    cdef double *rho_ptr = &rho_mv[0] if n > 0 else NULL
    with nogil:
        status = sdsge_lb_stat(x_ptr, n, L, z_ptr, rho_ptr, &stat)
    return status, np.float64(stat)


def hac_estimator_matmul(r, int kernel_id, int64_t L):
    """HAC long-run covariance (Gamma_0 + sum_j w_j(Gamma_j + Gamma_j')) / n.

    Full-estimator parity with the numba ``jit_hac_estimator_matmul``: same
    inputs (centered moment array, integer kernel id, bandwidth) and the same
    (p, p) output -- no separate Gamma_0/scaling codepath on the caller side.
    """
    cdef double[:, ::1] r_mv = np.ascontiguousarray(r, dtype=np.float64)
    cdef int64_t n = r_mv.shape[0]
    cdef int64_t p = r_mv.shape[1]
    omega = np.empty((p, p), dtype=np.float64)
    gamma = np.empty((p, p), dtype=np.float64)
    cdef double[:, ::1] out_mv = omega
    cdef double[:, ::1] gamma_mv = gamma
    with nogil:
        sdsge_hac_estimator_matmul(&r_mv[0, 0], <KernelID>kernel_id, L, n, p,
                                   &gamma_mv[0, 0], &out_mv[0, 0])
    return omega


def fill_mean_ax0(x):
    """Column means of x over axis 0. Returns mean(p)."""
    cdef double[:, ::1] x_mv = np.ascontiguousarray(x, dtype=np.float64)

    cdef int64_t n = x_mv.shape[0]
    cdef int64_t p = x_mv.shape[1]
    mean = np.empty(p, dtype=np.float64)
    cdef double[::1] mean_mv = mean
    with nogil:
        sdsge_fill_mean_ax0(&x_mv[0, 0], n, p, &mean_mv[0])
    return mean


def fill_centered_ax0(x, mean):
    """x with its column means subtracted. Returns centered(n, p)."""
    cdef double[:, ::1] x_mv = np.ascontiguousarray(x, dtype=np.float64)
    cdef double[::1] mean_mv = np.ascontiguousarray(mean, dtype=np.float64)

    cdef int64_t n = x.shape[0]
    cdef int64_t p = x.shape[1]
    centered = np.empty((n, p), dtype=np.float64)
    cdef double[:, ::1] centered_mv = centered
    with nogil:
        sdsge_fill_centered_ax0(&x_mv[0, 0], &mean_mv[0], n, p, &centered_mv[0, 0])
    return centered


def wald_stat_from_mean_and_cov(mean, target,
                                omega, int64_t n):
    """Wald statistic n * dev^T omega^-1 dev with dev = mean - target.

    Returns (status, stat). status is DIAG_OK, or FALLBACK when omega is not
    positive definite (the caller recomputes via the numba LU path).
    """
    cdef int64_t p = mean.shape[0]
    cdef double stat = 0.0
    cdef int status

    cdef double[::1] mean_mv = np.ascontiguousarray(mean, dtype=np.float64)
    cdef double[::1] target_mv = np.ascontiguousarray(target, dtype=np.float64)
    cdef double[:, ::1] omega_mv = np.ascontiguousarray(omega, dtype=np.float64)

    dev = np.empty(p, dtype=np.float64)
    chol = np.empty((p, p), dtype=np.float64)
    cdef double[::1] dev_mv = dev
    cdef double[:, ::1] chol_mv = chol
    with nogil:
        status = sdsge_wald_stat_from_mean_and_cov(
            &mean_mv[0], &target_mv[0], &omega_mv[0, 0], n, p,
            &dev_mv[0], &chol_mv[0, 0], &stat)
    return status, stat


def symmetric_outer_prod_2dim(x):
    """Per-row vech of the outer product x_t x_t'. Returns (status, out(n, q))."""
    cdef double[:, ::1] x_mv = np.ascontiguousarray(x, dtype=np.float64)

    cdef int64_t n = x_mv.shape[0]
    cdef int64_t p = x_mv.shape[1]
    cdef int64_t q = p * (p + 1) // 2
    out = np.empty((n, q), dtype=np.float64)
    cdef double[:, ::1] out_mv = out
    cdef int status
    with nogil:
        status = sdsge_symmetric_outer_prod_2dim(&x_mv[0, 0], n, p, q, &out_mv[0, 0])
    return status, out


def fill_symmetric_target_vec(target, double atol, double rtol):
    """Pack the upper triangle of a symmetric target into a vech vector.

    Returns (status, vec(q)); status is DIAG_BAD_SHAPE if the matrix is not
    symmetric within (atol, rtol).
    """
    cdef double[:, ::1] target_mv = np.ascontiguousarray(target, dtype=np.float64)

    cdef int64_t p = target_mv.shape[0]
    cdef int64_t q = p * (p + 1) // 2
    vec = np.empty(q, dtype=np.float64)
    cdef double[::1] vec_mv = vec
    cdef int status
    with nogil:
        status = sdsge_fill_symmetric_target_vec(&target_mv[0, 0], atol, rtol, p,
                                                 &vec_mv[0])
    return status, vec
