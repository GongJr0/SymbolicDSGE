# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Thin Cython shim mapping NumPy buffers to the pure-C Monte Carlo transform
kernels.

No numeric logic here -- only buffer->pointer marshalling, scratch allocation,
and the GIL release. The algorithms live in transforms.c. Each ``def`` mirrors
the matching Python op in
``SymbolicDSGE/monte_carlo/step_factories.py`` and returns the same
array shape, including the row counts that shrink with ``order`` or ``window``.

Arguments a kernel is not defined on (a window wider than the sample, a
non-positive order, a ddof that empties the denominator) are screened here and
raised as ``ValueError`` naming the offending pair, so the messages match the
ones the pure-Python ops raised. ``SDSGE_TRANSFORM_BAD_ARG`` coming back from a
kernel anyway is a backstop, not the usual path.

Scratch is allocated per call here: the native replication loop calls the C
entry points directly and owns its buffers.
"""

import numpy as np
from libc.stdint cimport int64_t

cdef extern from "transforms.h":
    int64_t SDSGE_TRANSFORM_SUCCESS
    int64_t SDSGE_TRANSFORM_BAD_ARG

    int64_t sdsge_standardize_ax0(const double *x, int64_t ddof, int64_t n,
                                  int64_t p, double *scratch, double *out) nogil

    int64_t sdsge_log(const double *x, double offset, int64_t n, int64_t p,
                      double *out) nogil

    int64_t sdsge_log_diff(const double *x, double offset, int64_t n, int64_t p,
                           double *scratch, double *out) nogil

    int64_t sdsge_diff(const double *x, int64_t order, int64_t n, int64_t p,
                       double *scratch, double *out) nogil

    int64_t sdsge_rolling_mean(const double *x, int64_t n, int64_t p,
                               int64_t window, double *scratch,
                               double *out) nogil

    int64_t sdsge_rolling_var(const double *x, int64_t n, int64_t p,
                              int64_t window, int64_t ddof, double *scratch,
                              double *out) nogil

    int64_t sdsge_rolling_std(const double *x, int64_t n, int64_t p,
                              int64_t window, int64_t ddof, double *scratch,
                              double *out) nogil

BAD_ARG = SDSGE_TRANSFORM_BAD_ARG


cdef inline void _check(int64_t status) except *:
    """Backstop for a rejection the wrappers below did not screen for.

    The wrappers raise the specific message first, so reaching this means the
    kernel refused something Python thought was fine; it is a guard against
    returning an unwritten buffer, not the usual error path.
    """
    if status != SDSGE_TRANSFORM_SUCCESS:
        raise ValueError(
            f"transform kernel rejected its arguments (status {status})."
        )


cdef inline void _check_window(int64_t window, int64_t n) except *:
    if window < 1:
        raise ValueError("rolling window must be at least 1.")
    if window > n:
        raise ValueError(
            f"rolling window ({window}) exceeds input length ({n})."
        )


cdef inline void _check_ddof(int64_t ddof, int64_t span, str span_name) except *:
    """Reject a ddof that empties (or inverts) the denominator.

    NumPy answers this case with a warning and a NaN; the kernels refuse it, so
    the wrapper reports which argument pair is at fault instead of leaving the
    caller to read a status code.
    """
    if ddof >= span:
        raise ValueError(
            f"ddof ({ddof}) must be smaller than the {span_name} ({span})."
        )


def standardize_ax0(x, int64_t ddof=0):
    """Per-column z-score over axis 0. Returns out(n, p).

    Columns with zero standard deviation come back as zeros.
    """
    cdef double[:, ::1] x_mv = np.ascontiguousarray(x, dtype=np.float64)
    cdef int64_t n = x_mv.shape[0]
    cdef int64_t p = x_mv.shape[1]
    cdef int64_t status

    if n == 0 or p == 0:
        raise ValueError("standardize requires a non-empty sample.")
    _check_ddof(ddof, n, "sample length")

    out = np.empty((n, p), dtype=np.float64)
    scratch = np.empty(2 * p, dtype=np.float64)
    cdef double[:, ::1] out_mv = out
    cdef double[::1] scratch_mv = scratch
    with nogil:
        status = sdsge_standardize_ax0(&x_mv[0, 0], ddof, n, p,
                                       &scratch_mv[0], &out_mv[0, 0])
    _check(status)
    return out


def log_transform(x, double offset=0.0):
    """``log(x + offset)`` elementwise. Returns out(n, p)."""
    cdef double[:, ::1] x_mv = np.ascontiguousarray(x, dtype=np.float64)
    cdef int64_t n = x_mv.shape[0]
    cdef int64_t p = x_mv.shape[1]
    cdef int64_t status

    if n == 0 or p == 0:
        raise ValueError("log requires a non-empty sample.")

    out = np.empty((n, p), dtype=np.float64)
    cdef double[:, ::1] out_mv = out
    with nogil:
        status = sdsge_log(&x_mv[0, 0], offset, n, p, &out_mv[0, 0])
    _check(status)
    return out


def log_diff_transform(x, double offset=0.0):
    """One-period log differences down the time axis. Returns out(n - 1, p)."""
    cdef double[:, ::1] x_mv = np.ascontiguousarray(x, dtype=np.float64)
    cdef int64_t n = x_mv.shape[0]
    cdef int64_t p = x_mv.shape[1]
    cdef int64_t status

    if n == 0 or p == 0:
        raise ValueError("log_diff requires a non-empty sample.")

    out = np.empty((n - 1, p), dtype=np.float64)
    if n == 1:  # no rows to write; the kernel would have nothing to do
        return out

    scratch = np.empty(p, dtype=np.float64)
    cdef double[:, ::1] out_mv = out
    cdef double[::1] scratch_mv = scratch
    with nogil:
        status = sdsge_log_diff(&x_mv[0, 0], offset, n, p,
                                &scratch_mv[0], &out_mv[0, 0])
    _check(status)
    return out


def diff_transform(x, int64_t order=1):
    """``order``-th difference down the time axis. Returns out(n - order, p)."""
    cdef double[:, ::1] x_mv = np.ascontiguousarray(x, dtype=np.float64)
    cdef int64_t n = x_mv.shape[0]
    cdef int64_t p = x_mv.shape[1]
    cdef int64_t status

    if n == 0 or p == 0:
        raise ValueError("diff requires a non-empty sample.")
    if order < 1:
        raise ValueError("diff order must be at least 1.")

    cdef int64_t n_out = n - order if n > order else 0
    out = np.empty((n_out, p), dtype=np.float64)
    if n_out == 0:  # every row is consumed priming the difference levels
        return out

    scratch = np.empty(order * p, dtype=np.float64)
    cdef double[:, ::1] out_mv = out
    cdef double[::1] scratch_mv = scratch
    with nogil:
        status = sdsge_diff(&x_mv[0, 0], order, n, p,
                            &scratch_mv[0], &out_mv[0, 0])
    _check(status)
    return out


def rolling_mean(x, int64_t window=10):
    """Trailing rolling mean. Returns out(n - window + 1, p)."""
    cdef double[:, ::1] x_mv = np.ascontiguousarray(x, dtype=np.float64)
    cdef int64_t n = x_mv.shape[0]
    cdef int64_t p = x_mv.shape[1]
    cdef int64_t status

    if n == 0 or p == 0:
        raise ValueError("rolling_mean requires a non-empty sample.")
    _check_window(window, n)

    out = np.empty((n - window + 1, p), dtype=np.float64)
    scratch = np.empty(p, dtype=np.float64)
    cdef double[:, ::1] out_mv = out
    cdef double[::1] scratch_mv = scratch
    with nogil:
        status = sdsge_rolling_mean(&x_mv[0, 0], n, p, window,
                                    &scratch_mv[0], &out_mv[0, 0])
    _check(status)
    return out


def rolling_var(x, int64_t window=10, int64_t ddof=0):
    """Trailing rolling variance. Returns out(n - window + 1, p)."""
    return _rolling_moment(x, window, ddof, False)


def rolling_std(x, int64_t window=10, int64_t ddof=0):
    """Trailing rolling standard deviation. Returns out(n - window + 1, p)."""
    return _rolling_moment(x, window, ddof, True)


cdef _rolling_moment(x, int64_t window, int64_t ddof, bint take_sqrt):
    cdef double[:, ::1] x_mv = np.ascontiguousarray(x, dtype=np.float64)
    cdef int64_t n = x_mv.shape[0]
    cdef int64_t p = x_mv.shape[1]
    cdef int64_t status

    if n == 0 or p == 0:
        raise ValueError("rolling moments require a non-empty sample.")
    _check_window(window, n)
    _check_ddof(ddof, window, "window")

    out = np.empty((n - window + 1, p), dtype=np.float64)
    scratch = np.empty(2 * p, dtype=np.float64)
    cdef double[:, ::1] out_mv = out
    cdef double[::1] scratch_mv = scratch
    with nogil:
        if take_sqrt:
            status = sdsge_rolling_std(&x_mv[0, 0], n, p, window, ddof,
                                       &scratch_mv[0], &out_mv[0, 0])
        else:
            status = sdsge_rolling_var(&x_mv[0, 0], n, p, window, ddof,
                                       &scratch_mv[0], &out_mv[0, 0])
    _check(status)
    return out
