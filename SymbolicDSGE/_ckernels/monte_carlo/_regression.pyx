# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Thin Python shim for native Monte Carlo regression kernels.

The caller owns the result and solver-work buffers.
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
    void sdsge_mc_ridge_fit(
        const double *X, const double *y, double alpha, int64_t intercept,
        sdsge_mc_regression_record *rec, double *L, double *G,
        double *G_unpen, double *g, double *col,
    ) nogil
    void sdsge_mc_ridge_gs_fit(
        const double *X, const double *y, const double *alphas,
        int64_t n_alpha, int64_t criterion, int64_t intercept,
        sdsge_mc_regression_record *rec, double *G_base, double *G,
        double *L, double *g, double *coef_work, double *col,
    ) nogil
    void sdsge_mc_lasso_fit(
        const double *X, const double *y, double alpha, int64_t intercept,
        int64_t max_iter, double tol, sdsge_mc_regression_record *rec,
        double *G_base, double *G, double *g, double *Gcoef,
    ) nogil
    void sdsge_mc_lasso_gs_fit(
        const double *X, const double *y, const double *alphas,
        int64_t n_alpha, int64_t intercept, int64_t max_iter, double tol,
        sdsge_mc_regression_record *rec, double *G_base, double *G,
        double *g, double *lam_path, double *beta_path, double *beta_grid,
        double *work,
    ) nogil
    void sdsge_mc_elastic_net_fit(
        const double *X, const double *y, double alpha, double l1_ratio,
        int64_t intercept, int64_t max_iter, double tol,
        sdsge_mc_regression_record *rec, double *G_base, double *G,
        double *g, double *Gcoef,
    ) nogil
    void sdsge_mc_elastic_net_gs_fit(
        const double *X, const double *y, const double *alphas,
        int64_t n_alpha, double l1_ratio, int64_t criterion,
        int64_t intercept, int64_t max_iter, double tol,
        sdsge_mc_regression_record *rec, double *G_base, double *G,
        double *g, double *beta_grid, int64_t *statuses, double *Gcoef,
        double *beta, double *dof_work,
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


def ridge_fit(X, y, coef, double alpha, L, G, G_unpen, g, col,
              bint intercept=True):
    """Fit ridge into caller-owned result and solver buffers."""
    cdef double[:, ::1] X_mv
    cdef double[::1] y_mv
    cdef double[::1] coef_mv = coef
    cdef double[::1] L_mv = L
    cdef double[::1] G_mv = G
    cdef double[::1] G_unpen_mv = G_unpen
    cdef double[::1] g_mv = g
    cdef double[::1] col_mv = col
    cdef sdsge_mc_regression_record rec

    X_array = np.ascontiguousarray(X, dtype=np.float64)
    y_array = np.ascontiguousarray(y, dtype=np.float64)
    if intercept:
        X_array = np.column_stack((np.ones(X_array.shape[0]), X_array))
    X_mv = X_array
    y_mv = y_array
    rec.n = X_mv.shape[0]
    rec.p = X_mv.shape[1]
    rec.coef = &coef_mv[0]
    rec.se = NULL
    with nogil:
        sdsge_mc_ridge_fit(
            &X_mv[0, 0], &y_mv[0], alpha, intercept, &rec, &L_mv[0],
            &G_mv[0], &G_unpen_mv[0], &g_mv[0], &col_mv[0]
        )
    return rec.ssr, rec.sst, rec.status


def ridge_gs_fit(X, y, alphas, int64_t criterion, coef, G_base, G, L, g,
                 coef_work, col, bint intercept=True):
    """Run native ridge grid search into caller-owned buffers."""
    cdef double[:, ::1] X_mv
    cdef double[::1] y_mv
    cdef double[::1] alphas_mv = alphas
    cdef double[::1] coef_mv = coef
    cdef double[::1] G_base_mv = G_base
    cdef double[::1] G_mv = G
    cdef double[::1] L_mv = L
    cdef double[::1] g_mv = g
    cdef double[::1] coef_work_mv = coef_work
    cdef double[::1] col_mv = col
    cdef sdsge_mc_regression_record rec

    X_array = np.ascontiguousarray(X, dtype=np.float64)
    y_array = np.ascontiguousarray(y, dtype=np.float64)
    if intercept:
        X_array = np.column_stack((np.ones(X_array.shape[0]), X_array))
    X_mv = X_array
    y_mv = y_array
    rec.n = X_mv.shape[0]
    rec.p = X_mv.shape[1]
    rec.coef = &coef_mv[0]
    rec.se = NULL
    with nogil:
        sdsge_mc_ridge_gs_fit(
            &X_mv[0, 0], &y_mv[0], &alphas_mv[0], alphas_mv.shape[0], criterion,
            intercept, &rec, &G_base_mv[0], &G_mv[0], &L_mv[0], &g_mv[0],
            &coef_work_mv[0], &col_mv[0]
        )
    return rec.ssr, rec.sst, rec.status


def lasso_fit(X, y, coef, double alpha, int64_t max_iter, double tol,
              G_base, G, g, Gcoef, bint intercept=True):
    """Fit lasso into caller-owned result and Gram-solver buffers."""
    cdef double[:, ::1] X_mv
    cdef double[::1] y_mv
    cdef double[::1] coef_mv = coef
    cdef double[::1] G_base_mv = G_base
    cdef double[::1] G_mv = G
    cdef double[::1] g_mv = g
    cdef double[::1] Gcoef_mv = Gcoef
    cdef sdsge_mc_regression_record rec

    X_array = np.ascontiguousarray(X, dtype=np.float64)
    y_array = np.ascontiguousarray(y, dtype=np.float64)
    if intercept:
        X_array = np.column_stack((np.ones(X_array.shape[0]), X_array))
    X_mv = X_array
    y_mv = y_array
    rec.n = X_mv.shape[0]
    rec.p = X_mv.shape[1]
    rec.coef = &coef_mv[0]
    rec.se = NULL
    with nogil:
        sdsge_mc_lasso_fit(
            &X_mv[0, 0], &y_mv[0], alpha, intercept, max_iter, tol, &rec,
            &G_base_mv[0], &G_mv[0], &g_mv[0], &Gcoef_mv[0]
        )
    return rec.ssr, rec.sst, rec.status


def lasso_gs_fit(X, y, alphas, coef, int64_t max_iter, double tol,
                 G_base, G, g, lam_path, beta_path, beta_grid, work,
                 bint intercept=True):
    """Run native lasso grid search into caller-owned path and solver buffers."""
    cdef double[:, ::1] X_mv
    cdef double[::1] y_mv
    cdef double[::1] alphas_mv = alphas
    cdef double[::1] coef_mv = coef
    cdef double[::1] G_base_mv = G_base
    cdef double[::1] G_mv = G
    cdef double[::1] g_mv = g
    cdef double[::1] lam_path_mv = lam_path
    cdef double[::1] beta_path_mv = beta_path
    cdef double[::1] beta_grid_mv = beta_grid
    cdef double[::1] work_mv = work
    cdef sdsge_mc_regression_record rec

    X_array = np.ascontiguousarray(X, dtype=np.float64)
    y_array = np.ascontiguousarray(y, dtype=np.float64)
    if intercept:
        X_array = np.column_stack((np.ones(X_array.shape[0]), X_array))
    X_mv = X_array
    y_mv = y_array
    rec.n = X_mv.shape[0]
    rec.p = X_mv.shape[1]
    rec.coef = &coef_mv[0]
    rec.se = NULL
    with nogil:
        sdsge_mc_lasso_gs_fit(
            &X_mv[0, 0], &y_mv[0], &alphas_mv[0], alphas_mv.shape[0],
            intercept, max_iter, tol, &rec, &G_base_mv[0], &G_mv[0], &g_mv[0],
            &lam_path_mv[0], &beta_path_mv[0], &beta_grid_mv[0], &work_mv[0]
        )
    return rec.ssr, rec.sst, rec.status


def elastic_net_fit(X, y, coef, double alpha, double l1_ratio,
                    int64_t max_iter, double tol, G_base, G, g, Gcoef,
                    bint intercept=True):
    """Fit elastic net into caller-owned result and Gram-solver buffers."""
    cdef double[:, ::1] X_mv
    cdef double[::1] y_mv
    cdef double[::1] coef_mv = coef
    cdef double[::1] G_base_mv = G_base
    cdef double[::1] G_mv = G
    cdef double[::1] g_mv = g
    cdef double[::1] Gcoef_mv = Gcoef
    cdef sdsge_mc_regression_record rec

    X_array = np.ascontiguousarray(X, dtype=np.float64)
    y_array = np.ascontiguousarray(y, dtype=np.float64)
    if intercept:
        X_array = np.column_stack((np.ones(X_array.shape[0]), X_array))
    X_mv = X_array
    y_mv = y_array
    rec.n = X_mv.shape[0]
    rec.p = X_mv.shape[1]
    rec.coef = &coef_mv[0]
    rec.se = NULL
    with nogil:
        sdsge_mc_elastic_net_fit(
            &X_mv[0, 0], &y_mv[0], alpha, l1_ratio, intercept, max_iter, tol,
            &rec, &G_base_mv[0], &G_mv[0], &g_mv[0], &Gcoef_mv[0]
        )
    return rec.ssr, rec.sst, rec.status


def elastic_net_gs_fit(X, y, alphas, double l1_ratio, int64_t criterion,
                       coef, int64_t max_iter, double tol, G_base, G, g,
                       beta_grid, statuses, Gcoef, beta, dof_work,
                       bint intercept=True):
    """Run native elastic-net grid search into caller-owned buffers."""
    cdef double[:, ::1] X_mv
    cdef double[::1] y_mv
    cdef double[::1] alphas_mv = alphas
    cdef double[::1] coef_mv = coef
    cdef double[::1] G_base_mv = G_base
    cdef double[::1] G_mv = G
    cdef double[::1] g_mv = g
    cdef double[::1] beta_grid_mv = beta_grid
    cdef int64_t[::1] statuses_mv = statuses
    cdef double[::1] Gcoef_mv = Gcoef
    cdef double[::1] beta_mv = beta
    cdef double[::1] dof_work_mv = dof_work
    cdef sdsge_mc_regression_record rec

    X_array = np.ascontiguousarray(X, dtype=np.float64)
    y_array = np.ascontiguousarray(y, dtype=np.float64)
    if intercept:
        X_array = np.column_stack((np.ones(X_array.shape[0]), X_array))
    X_mv = X_array
    y_mv = y_array
    rec.n = X_mv.shape[0]
    rec.p = X_mv.shape[1]
    rec.coef = &coef_mv[0]
    rec.se = NULL
    with nogil:
        sdsge_mc_elastic_net_gs_fit(
            &X_mv[0, 0], &y_mv[0], &alphas_mv[0], alphas_mv.shape[0], l1_ratio,
            criterion, intercept, max_iter, tol, &rec, &G_base_mv[0], &G_mv[0],
            &g_mv[0], &beta_grid_mv[0], &statuses_mv[0], &Gcoef_mv[0],
            &beta_mv[0], &dof_work_mv[0]
        )
    return rec.ssr, rec.sst, rec.status
