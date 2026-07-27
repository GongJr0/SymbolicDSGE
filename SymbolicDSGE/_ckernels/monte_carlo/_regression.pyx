

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Thin Python shim for native Monte Carlo regression kernels.

The caller owns result and solver-work buffers. The eventual native MC
executor can use the same ABI with its preallocated trace rows and scratch
space.
"""

from libc.stdint cimport int64_t
import numpy as np

cdef extern from "regression.h":
    ctypedef struct sdsge_mc_regression_record:
        int64_t n
        int64_t p
        double *coef
        double *se
        double ssr
        double sst
        int64_t status

    void sdsge_mc_ols_fit(
        const double *X,
        const double *y,
        sdsge_mc_regression_record *rec,
        double *L,
        double *G,
        double *g,
        double *work,
    ) nogil


def ols_fit(X, y, coef, se, L, G, g, work, bint intercept=True):
    """Fit OLS and return raw native result fields.

    ``X`` must be a two-dimensional design and ``y`` a one-dimensional
    response with the same number of rows. ``coef`` and ``se`` must each have
    one slot per effective regressor. ``L``, ``G``, ``g``, and ``work`` are
    caller-owned solver scratch buffers of sizes ``p*p``, ``p*p``, ``p``, and
    ``p``. When requested, the intercept column is materialized here before
    the GIL is released.
    """
    cdef double[:, ::1] X_mv
    cdef double[::1] y_mv
    cdef double[::1] coef_mv
    cdef double[::1] se_mv
    cdef double[::1] L_mv
    cdef double[::1] G_mv
    cdef double[::1] g_mv
    cdef double[::1] work_mv
    cdef int64_t n
    cdef int64_t p
    cdef sdsge_mc_regression_record rec

    X_array = np.ascontiguousarray(X, dtype=np.float64)
    y_array = np.ascontiguousarray(y, dtype=np.float64)
    if X_array.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if y_array.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if X_array.shape[0] == 0:
        raise ValueError("OLS requires at least one row.")
    if X_array.shape[1] == 0 and not intercept:
        raise ValueError("OLS without an intercept requires at least one regressor.")

    if intercept:
        X_array = np.column_stack((np.ones(X_array.shape[0]), X_array))

    X_mv = X_array
    y_mv = y_array
    n = X_mv.shape[0]
    p = X_mv.shape[1]

    coef_mv = coef
    se_mv = se
    L_mv = L
    G_mv = G
    g_mv = g
    work_mv = work

    rec.n = n
    rec.p = p
    rec.coef = &coef_mv[0]
    rec.se = &se_mv[0]
    with nogil:
        sdsge_mc_ols_fit(
            &X_mv[0, 0], &y_mv[0], &rec, &L_mv[0], &G_mv[0], &g_mv[0], &work_mv[0]
        )

    return rec.ssr, rec.sst, rec.status
