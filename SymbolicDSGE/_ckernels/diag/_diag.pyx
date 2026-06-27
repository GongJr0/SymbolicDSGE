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


def bg_stat(double[::1] eps, double[:, ::1] X, int64_t lags):
    """Breusch-Godfrey LM statistic. Returns (status, stat)."""
    cdef int64_t n = eps.shape[0]
    cdef int64_t K = X.shape[1]
    cdef double stat = 0.0
    cdef int status
    with nogil:
        status = sdsge_bg_stat(&eps[0], &X[0, 0], n, K, lags, &stat)
    return status, stat


def bp_aux(double[::1] eps, double[:, ::1] X_aug):
    """Breusch-Pagan auxiliary regression. Returns (status, rss, tss)."""
    cdef int64_t n = X_aug.shape[0]
    cdef int64_t p = X_aug.shape[1]
    cdef double rss = 0.0
    cdef double tss = 0.0
    cdef int status
    with nogil:
        status = sdsge_bp_aux(&eps[0], &X_aug[0, 0], n, p, &rss, &tss)
    return status, rss, tss


def chow_stat(double[::1] y, double[:, ::1] X, int64_t t_break):
    """Chow break-point F statistic. Returns (status, stat)."""
    cdef int64_t T = X.shape[0]
    cdef int64_t p = X.shape[1]
    cdef double stat = 0.0
    cdef int status
    with nogil:
        status = sdsge_chow_stat(&y[0], &X[0, 0], T, p, t_break, &stat)
    return status, stat


def recursive_residuals(double[::1] y, double[:, ::1] X):
    """Brown-Durbin-Evans recursive residuals. Returns (status, w)."""
    cdef int64_t T = X.shape[0]
    cdef int64_t p = X.shape[1]
    cdef int64_t w_len = T - p if T > p else 0
    w = np.empty(w_len, dtype=np.float64)
    cdef double[::1] w_mv = w
    cdef double *w_ptr = &w_mv[0] if w_len > 0 else NULL
    cdef int status
    with nogil:
        status = sdsge_recursive_residuals(&y[0], &X[0, 0], T, p, w_ptr)
    return status, w


def cusum_series(double[::1] y, double[:, ::1] X):
    """Standardized CUSUM series. Returns (status, series)."""
    cdef int64_t T = X.shape[0]
    cdef int64_t p = X.shape[1]
    cdef int64_t s_len = T - p if T > p else 0
    series = np.empty(s_len, dtype=np.float64)
    cdef double[::1] s_mv = series
    cdef double *s_ptr = &s_mv[0] if s_len > 0 else NULL
    cdef int status
    with nogil:
        status = sdsge_cusum_series(&y[0], &X[0, 0], T, p, s_ptr)
    return status, series


def cusum_stat(double[::1] y, double[:, ::1] X):
    """CUSUM statistic. Returns (status, stat)."""
    cdef int64_t T = X.shape[0]
    cdef int64_t p = X.shape[1]
    cdef double stat = 0.0
    cdef int status
    with nogil:
        status = sdsge_cusum_stat(&y[0], &X[0, 0], T, p, &stat)
    return status, stat


def cusumsq_stat(double[::1] y, double[:, ::1] X):
    """CUSUM-of-squares statistic. Returns (status, n, stat)."""
    cdef int64_t T = X.shape[0]
    cdef int64_t p = X.shape[1]
    cdef int64_t n = 0
    cdef double stat = 0.0
    cdef int status
    with nogil:
        status = sdsge_cusumsq_stat(&y[0], &X[0, 0], T, p, &n, &stat)
    return status, n, stat


def hac_estimator_matmul(double[:, ::1] r, int kernel_id, int64_t L):
    """HAC long-run covariance (Gamma_0 + sum_j w_j(Gamma_j + Gamma_j')) / n.

    Full-estimator parity with the numba ``jit_hac_estimator_matmul``: same
    inputs (centered moment array, integer kernel id, bandwidth) and the same
    (p, p) output -- no separate Gamma_0/scaling codepath on the caller side.
    """
    cdef int64_t n = r.shape[0]
    cdef int64_t p = r.shape[1]
    omega = np.empty((p, p), dtype=np.float64)
    gamma = np.empty((p, p), dtype=np.float64)
    cdef double[:, ::1] out_mv = omega
    cdef double[:, ::1] gamma_mv = gamma
    with nogil:
        sdsge_hac_estimator_matmul(&r[0, 0], <KernelID>kernel_id, L, n, p,
                                   &gamma_mv[0, 0], &out_mv[0, 0])
    return omega


def fill_mean_ax0(double[:, ::1] x):
    """Column means of x over axis 0. Returns mean(p)."""
    cdef int64_t n = x.shape[0]
    cdef int64_t p = x.shape[1]
    mean = np.empty(p, dtype=np.float64)
    cdef double[::1] mean_mv = mean
    with nogil:
        sdsge_fill_mean_ax0(&x[0, 0], n, p, &mean_mv[0])
    return mean


def fill_centered_ax0(double[:, ::1] x, double[::1] mean):
    """x with its column means subtracted. Returns centered(n, p)."""
    cdef int64_t n = x.shape[0]
    cdef int64_t p = x.shape[1]
    centered = np.empty((n, p), dtype=np.float64)
    cdef double[:, ::1] centered_mv = centered
    with nogil:
        sdsge_fill_centered_ax0(&x[0, 0], &mean[0], n, p, &centered_mv[0, 0])
    return centered


def wald_stat_from_mean_and_cov(double[::1] mean, double[::1] target,
                                double[:, ::1] omega, int64_t n):
    """Wald statistic n * dev^T omega^-1 dev with dev = mean - target.

    Returns (status, stat). status is DIAG_OK, or FALLBACK when omega is not
    positive definite (the caller recomputes via the numba LU path).
    """
    cdef int64_t p = mean.shape[0]
    cdef double stat = 0.0
    cdef int status
    dev = np.empty(p, dtype=np.float64)
    chol = np.empty((p, p), dtype=np.float64)
    cdef double[::1] dev_mv = dev
    cdef double[:, ::1] chol_mv = chol
    with nogil:
        status = sdsge_wald_stat_from_mean_and_cov(
            &mean[0], &target[0], &omega[0, 0], n, p,
            &dev_mv[0], &chol_mv[0, 0], &stat)
    return status, stat


def symmetric_outer_prod_2dim(double[:, ::1] x):
    """Per-row vech of the outer product x_t x_t'. Returns (status, out(n, q))."""
    cdef int64_t n = x.shape[0]
    cdef int64_t p = x.shape[1]
    cdef int64_t q = p * (p + 1) // 2
    out = np.empty((n, q), dtype=np.float64)
    cdef double[:, ::1] out_mv = out
    cdef int status
    with nogil:
        status = sdsge_symmetric_outer_prod_2dim(&x[0, 0], n, p, q, &out_mv[0, 0])
    return status, out


def fill_symmetric_target_vec(double[:, ::1] target, double atol, double rtol):
    """Pack the upper triangle of a symmetric target into a vech vector.

    Returns (status, vec(q)); status is DIAG_BAD_SHAPE if the matrix is not
    symmetric within (atol, rtol).
    """
    cdef int64_t p = target.shape[0]
    cdef int64_t q = p * (p + 1) // 2
    vec = np.empty(q, dtype=np.float64)
    cdef double[::1] vec_mv = vec
    cdef int status
    with nogil:
        status = sdsge_fill_symmetric_target_vec(&target[0, 0], atol, rtol, p,
                                                 &vec_mv[0])
    return status, vec
