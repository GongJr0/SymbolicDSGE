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
