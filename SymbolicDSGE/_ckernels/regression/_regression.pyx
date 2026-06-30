# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Thin Cython shim for the native regression kernels.

No numeric logic here -- only buffer->pointer marshalling, scratch allocation,
and the GIL release. The algorithms live in regression.c; each ``def`` mirrors
the matching numba helper in SymbolicDSGE/regression so the parity tests can hit
them directly. ``chol_solve_L2`` is the single ridge solve; ``ridge_grid_search``
runs the whole alpha grid in one native call (the Gram is formed once). A
RANK_DEFICIENT status mirrors the numba contract (NaN coef, empty L).
"""

import numpy as np
from libc.stdint cimport int64_t


cdef extern from "regression.h":
    void sdsge_chol_solve_L2(const double *X, const double *y, int64_t n,
                             int64_t p, double alpha, int64_t intercept,
                             double *coef, double *L, double *dof,
                             int64_t *status, double *G, double *G_unpen,
                             double *g, double *col) nogil
    void sdsge_ridge_grid_search(const double *X, const double *y, int64_t n,
                                 int64_t p, const double *alphas, int64_t num,
                                 int64_t criterion, int64_t intercept,
                                 double *out_alpha, double *out_coef,
                                 double *out_obj, int64_t *out_status,
                                 double *G_base, double *G, double *g, double *L,
                                 double *coef, double *col) nogil


def chol_solve_L2(double[:, ::1] X, double[::1] y, double alpha, bint intercept):
    """Ridge solve. Returns ``(coef, L, eff_dof, status)`` (L empty if not PD)."""
    cdef int64_t n = X.shape[0]
    cdef int64_t p = X.shape[1]
    cdef double[::1] coef = np.empty(p, dtype=np.float64)
    cdef double[:, ::1] L = np.empty((p, p), dtype=np.float64)
    cdef double[:, ::1] G = np.empty((p, p), dtype=np.float64)
    cdef double[:, ::1] G_unpen = np.empty((p, p), dtype=np.float64)
    cdef double[::1] g = np.empty(p, dtype=np.float64)
    cdef double[::1] col = np.empty(p, dtype=np.float64)
    cdef double dof = 0.0
    cdef int64_t status = 0
    with nogil:
        sdsge_chol_solve_L2(&X[0, 0], &y[0], n, p, alpha, intercept,
                            &coef[0], &L[0, 0], &dof, &status,
                            &G[0, 0], &G_unpen[0, 0], &g[0], &col[0])
    if status != 0:
        return np.asarray(coef), np.empty((0, 0), dtype=np.float64), dof, status
    return np.asarray(coef), np.asarray(L), dof, status


def ridge_grid_search(double[:, ::1] X, double[::1] y, double[::1] alphas,
                      int64_t criterion, bint intercept):
    """Ridge grid search. Returns ``(alpha, coef, obj_value, status)``."""
    cdef int64_t n = X.shape[0]
    cdef int64_t p = X.shape[1]
    cdef int64_t num = alphas.shape[0]
    cdef double[::1] out_coef = np.empty(p, dtype=np.float64)
    cdef double[:, ::1] G_base = np.empty((p, p), dtype=np.float64)
    cdef double[:, ::1] G = np.empty((p, p), dtype=np.float64)
    cdef double[:, ::1] L = np.empty((p, p), dtype=np.float64)
    cdef double[::1] g = np.empty(p, dtype=np.float64)
    cdef double[::1] coef = np.empty(p, dtype=np.float64)
    cdef double[::1] col = np.empty(p, dtype=np.float64)
    cdef double out_alpha = 0.0
    cdef double out_obj = 0.0
    cdef int64_t out_status = 0
    with nogil:
        sdsge_ridge_grid_search(&X[0, 0], &y[0], n, p, &alphas[0], num,
                                criterion, intercept, &out_alpha, &out_coef[0],
                                &out_obj, &out_status, &G_base[0, 0], &G[0, 0],
                                &g[0], &L[0, 0], &coef[0], &col[0])
    return out_alpha, np.asarray(out_coef), out_obj, out_status
