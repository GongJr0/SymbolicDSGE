# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Caller-buffer shims for native non-Wald Monte Carlo test statistics.

These functions return only ``(statistic, status)``. Distribution metadata and
p-values are replication-invariant and belong to the post-loop layer.
"""

import numpy as np

from libc.stdint cimport int64_t


cdef extern from "diag.h":
    int sdsge_bg_stat(
        const double *eps, const double *X, int64_t n, int64_t k,
        int64_t lags, double *arena, double *stat_out,
    ) nogil
    int sdsge_bp_aux(
        const double *eps, const double *X_aug, int64_t n, int64_t p,
        double *arena, double *rss_out, double *tss_out,
    ) nogil
    int sdsge_chow_stat(
        const double *y, const double *X, int64_t n, int64_t p,
        int64_t t_break, double *arena, double *stat_out,
    ) nogil
    int sdsge_cusum_stat(
        const double *y, const double *X, int64_t n, int64_t p,
        double *arena, double *stat_out,
    ) nogil
    int sdsge_cusumsq_stat(
        const double *y, const double *X, int64_t n, int64_t p,
        int64_t *n_out, double *arena, double *stat_out,
    ) nogil
    int sdsge_lb_stat(
        const double *x, int64_t n, int64_t lags, double *z_scratch,
        double *acorr_scratch, double *stat_out,
    ) nogil
    int sdsge_jb_stat(const double *x, int64_t n, double *stat_out) nogil


def ljung_box_fit(x, int64_t lags, z_scratch, acorr_scratch):
    """Return the Ljung-Box ``(statistic, status)`` using caller scratch."""
    cdef double[::1] x_mv = np.ascontiguousarray(x, dtype=np.float64)
    cdef double[::1] z_mv = z_scratch
    cdef double[::1] acorr_mv = acorr_scratch
    cdef double stat = 0.0
    cdef int status
    with nogil:
        status = sdsge_lb_stat(
            &x_mv[0], x_mv.shape[0], lags, &z_mv[0], &acorr_mv[0], &stat
        )
    return stat, status


def jarque_bera_fit(x):
    """Return the allocation-free Jarque-Bera ``(statistic, status)``."""
    cdef double[::1] x_mv = np.ascontiguousarray(x, dtype=np.float64)
    cdef double stat = 0.0
    cdef int status
    cdef double *x_ptr = &x_mv[0] if x_mv.shape[0] > 0 else NULL
    with nogil:
        status = sdsge_jb_stat(x_ptr, x_mv.shape[0], &stat)
    return stat, status


def breusch_pagan_fit(residuals, X_aug, bint robust, arena):
    """Return a Breusch-Pagan ``(statistic, status)`` from an augmented design.

    ``X_aug`` must include the leading auxiliary-regression intercept column.
    """
    cdef double[::1] residuals_mv = np.ascontiguousarray(
        residuals, dtype=np.float64
    )
    cdef double[:, ::1] X_aug_mv = np.ascontiguousarray(X_aug, dtype=np.float64)
    cdef double[::1] arena_mv = arena
    cdef double rss = 0.0
    cdef double tss = 0.0
    cdef double stat = np.nan
    cdef double r2
    cdef int status
    with nogil:
        status = sdsge_bp_aux(
            &residuals_mv[0], &X_aug_mv[0, 0], X_aug_mv.shape[0],
            X_aug_mv.shape[1], &arena_mv[0], &rss, &tss
        )
    if status != 0:
        return stat, status
    if robust:
        if tss <= 0.0:
            stat = 0.0
        else:
            r2 = 1.0 - rss / tss
            if r2 < 0.0:
                r2 = 0.0
            elif r2 > 1.0:
                r2 = 1.0
            stat = r2 * residuals_mv.shape[0]
    else:
        stat = 0.5 * (tss - rss)
        if stat < 0.0:
            stat = 0.0
    return stat, status


def breusch_godfrey_fit(residuals, X, int64_t lags, arena):
    """Return the Breusch-Godfrey ``(statistic, status)``."""
    cdef double[::1] residuals_mv = np.ascontiguousarray(
        residuals, dtype=np.float64
    )
    cdef double[:, ::1] X_mv = np.ascontiguousarray(X, dtype=np.float64)
    cdef double[::1] arena_mv = arena
    cdef double stat = 0.0
    cdef int status
    with nogil:
        status = sdsge_bg_stat(
            &residuals_mv[0], &X_mv[0, 0], X_mv.shape[0], X_mv.shape[1], lags,
            &arena_mv[0], &stat
        )
    return stat, status


def chow_fit(y, X, int64_t t_break, arena):
    """Return the Chow break-point ``(statistic, status)``."""
    cdef double[::1] y_mv = np.ascontiguousarray(y, dtype=np.float64)
    cdef double[:, ::1] X_mv = np.ascontiguousarray(X, dtype=np.float64)
    cdef double[::1] arena_mv = arena
    cdef double stat = 0.0
    cdef int status
    with nogil:
        status = sdsge_chow_stat(
            &y_mv[0], &X_mv[0, 0], X_mv.shape[0], X_mv.shape[1], t_break,
            &arena_mv[0], &stat
        )
    return stat, status


def cusum_fit(y, X, arena):
    """Return the CUSUM ``(statistic, status)``."""
    cdef double[::1] y_mv = np.ascontiguousarray(y, dtype=np.float64)
    cdef double[:, ::1] X_mv = np.ascontiguousarray(X, dtype=np.float64)
    cdef double[::1] arena_mv = arena
    cdef double stat = 0.0
    cdef int status
    with nogil:
        status = sdsge_cusum_stat(
            &y_mv[0], &X_mv[0, 0], X_mv.shape[0], X_mv.shape[1],
            &arena_mv[0], &stat
        )
    return stat, status


def cusumsq_fit(y, X, arena):
    """Return the CUSUMSQ ``(statistic, status)``."""
    cdef double[::1] y_mv = np.ascontiguousarray(y, dtype=np.float64)
    cdef double[:, ::1] X_mv = np.ascontiguousarray(X, dtype=np.float64)
    cdef double[::1] arena_mv = arena
    cdef int64_t n_out = 0
    cdef double stat = 0.0
    cdef int status
    with nogil:
        status = sdsge_cusumsq_stat(
            &y_mv[0], &X_mv[0, 0], X_mv.shape[0], X_mv.shape[1], &n_out,
            &arena_mv[0], &stat
        )
    return stat, status
