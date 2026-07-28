# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Caller-buffer shims for native Monte Carlo test statistics.

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

cdef extern from "diag_wald.h":
    int sdsge_wald_stat_from_mean_and_cov(
        const double *mean, const double *target, const double *omega,
        int64_t n, int64_t p, double *dev_scratch, double *factor_scratch,
        int64_t *pivot_scratch, double *solved_scratch, double *stat_out,
    ) nogil
    int sdsge_wald_mean_hac(
        const double *g, const double *target, int64_t n, int64_t q,
        int kernel_id, int bandwidth_mode, int64_t manual_bandwidth,
        double *arena, int64_t *pivot_scratch, double *stat_out,
    ) nogil
    int sdsge_wald_covariance_hac(
        const double *g, const double *target, int64_t n, int64_t q,
        int kernel_id, int bandwidth_mode, int64_t manual_bandwidth,
        double *arena, int64_t *pivot_scratch, double *stat_out,
    ) nogil
    int sdsge_wald_second_moment_hac(
        const double *g, const double *target, int64_t n, int64_t q,
        int kernel_id, int bandwidth_mode, int64_t manual_bandwidth,
        double *arena, int64_t *pivot_scratch, double *stat_out,
    ) nogil


WALD_BW_MANUAL = 0
WALD_BW_WOOLDRIDGE = 1
WALD_BW_ANDREWS = 2
WALD_BW_AUTO = 3


def ljung_box_runner(x, int64_t lags, z_scratch, acorr_scratch):
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


def jarque_bera_runner(x):
    """Return the allocation-free Jarque-Bera ``(statistic, status)``."""
    cdef double[::1] x_mv = np.ascontiguousarray(x, dtype=np.float64)
    cdef double stat = 0.0
    cdef int status
    cdef double *x_ptr = &x_mv[0] if x_mv.shape[0] > 0 else NULL
    with nogil:
        status = sdsge_jb_stat(x_ptr, x_mv.shape[0], &stat)
    return stat, status


def wald_runner(
    mean, target, omega, int64_t n, dev_scratch, factor_scratch,
    pivot_scratch, solved_scratch,
):
    """Return the Wald ``(statistic, status)`` using caller-owned scratch.

    ``dev_scratch`` and ``solved_scratch`` have length ``p``;
    ``factor_scratch`` has shape ``(p, p)``; and ``pivot_scratch`` has length
    ``p``, where ``p`` is the dimension of ``mean``.
    """
    cdef double[::1] mean_mv = np.ascontiguousarray(mean, dtype=np.float64)
    cdef double[::1] target_mv = np.ascontiguousarray(target, dtype=np.float64)
    cdef double[:, ::1] omega_mv = np.ascontiguousarray(omega, dtype=np.float64)
    cdef double[::1] dev_mv = dev_scratch
    cdef double[:, ::1] factor_mv = factor_scratch
    cdef int64_t[::1] pivot_mv = pivot_scratch
    cdef double[::1] solved_mv = solved_scratch
    cdef double stat = np.nan
    cdef int status
    with nogil:
        status = sdsge_wald_stat_from_mean_and_cov(
            &mean_mv[0], &target_mv[0], &omega_mv[0, 0], n, mean_mv.shape[0],
            &dev_mv[0], &factor_mv[0, 0], &pivot_mv[0], &solved_mv[0], &stat,
        )
    return stat, status


def wald_mean_hac_runner(
    g, target, int kernel_id, int bandwidth_mode, int64_t manual_bandwidth,
    arena, pivot_scratch,
):
    """Return mean-Wald HAC ``(statistic, status)`` using caller scratch."""
    cdef double[:, ::1] g_mv = np.ascontiguousarray(g, dtype=np.float64)
    cdef double[::1] target_mv = np.ascontiguousarray(target, dtype=np.float64)
    cdef double[::1] arena_mv = arena
    cdef int64_t[::1] pivot_mv = pivot_scratch
    cdef double stat = np.nan
    cdef int status
    with nogil:
        status = sdsge_wald_mean_hac(
            &g_mv[0, 0], &target_mv[0], g_mv.shape[0], g_mv.shape[1],
            kernel_id, bandwidth_mode, manual_bandwidth, &arena_mv[0],
            &pivot_mv[0], &stat,
        )
    return stat, status


def wald_covariance_hac_runner(
    g, target, int kernel_id, int bandwidth_mode, int64_t manual_bandwidth,
    arena, pivot_scratch,
):
    """Return covariance-Wald HAC ``(statistic, status)`` using caller scratch."""
    cdef double[:, ::1] g_mv = np.ascontiguousarray(g, dtype=np.float64)
    cdef double[:, ::1] target_mv = np.ascontiguousarray(target, dtype=np.float64)
    cdef double[::1] arena_mv = arena
    cdef int64_t[::1] pivot_mv = pivot_scratch
    cdef double stat = np.nan
    cdef int status
    with nogil:
        status = sdsge_wald_covariance_hac(
            &g_mv[0, 0], &target_mv[0, 0], g_mv.shape[0], g_mv.shape[1],
            kernel_id, bandwidth_mode, manual_bandwidth, &arena_mv[0],
            &pivot_mv[0], &stat,
        )
    return stat, status


def wald_second_moment_hac_runner(
    g, target, int kernel_id, int bandwidth_mode, int64_t manual_bandwidth,
    arena, pivot_scratch,
):
    """Return second-moment Wald HAC ``(statistic, status)`` using caller scratch."""
    cdef double[:, ::1] g_mv = np.ascontiguousarray(g, dtype=np.float64)
    cdef double[:, ::1] target_mv = np.ascontiguousarray(target, dtype=np.float64)
    cdef double[::1] arena_mv = arena
    cdef int64_t[::1] pivot_mv = pivot_scratch
    cdef double stat = np.nan
    cdef int status
    with nogil:
        status = sdsge_wald_second_moment_hac(
            &g_mv[0, 0], &target_mv[0, 0], g_mv.shape[0], g_mv.shape[1],
            kernel_id, bandwidth_mode, manual_bandwidth, &arena_mv[0],
            &pivot_mv[0], &stat,
        )
    return stat, status


def breusch_pagan_runner(residuals, X_aug, bint robust, arena):
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


def breusch_godfrey_runner(residuals, X, int64_t lags, arena):
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


def chow_runner(y, X, int64_t t_break, arena):
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


def cusum_runner(y, X, arena):
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


def cusumsq_runner(y, X, arena):
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
