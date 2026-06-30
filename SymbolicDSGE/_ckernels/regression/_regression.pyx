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
    void sdsge_ols_chol_solve(const double *X, const double *y, int64_t n,
                              int64_t p, double *coef, double *L, int64_t *status,
                              double *G, double *g) nogil


cdef extern from "elastic_net.h":
    int64_t sdsge_en_gram_cd(const double *G, const double *g, int64_t k,
                             double alpha_l1, double alpha_l2, const double *beta0,
                             int64_t max_iter, double tol, double *coef,
                             double *Gcoef) nogil
    void sdsge_en_gram_cd_path(const double *G, const double *g, int64_t k,
                               const double *alpha_grid, int64_t n_alpha,
                               double l1_ratio, int64_t max_iter, double tol,
                               double *coefs, int64_t *statuses, double *Gcoef,
                               double *beta) nogil
    double sdsge_en_active_dof(const double *G, const double *beta, int64_t k,
                               double alpha_l2, int64_t intercept,
                               double atol) nogil


cdef extern from "lasso.h":
    int64_t sdsge_lasso_gram_cd(const double *G, const double *g, int64_t k,
                                double alpha, int64_t max_iter, double tol,
                                double *coef, double *Gcoef) nogil
    int64_t sdsge_lars_lasso_gram(const double *G, const double *c, int64_t k,
                                  int64_t max_iter, double tol, double *lam_path,
                                  double *beta_path, int64_t *n_knots) nogil
    void sdsge_lasso_path_eval(const double *lam_path, const double *beta_path,
                               int64_t n_knots, int64_t k, const double *lam_grid,
                               int64_t n_grid, double *out) nogil


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


def lasso_gram_cd(double[:, ::1] G, double[::1] g, double alpha,
                  int64_t max_iter, double tol):
    """Coordinate-descent lasso solve. Returns ``(coef, status)``."""
    cdef int64_t k = G.shape[0]
    cdef double[::1] coef = np.empty(k, dtype=np.float64)
    cdef double[::1] Gcoef = np.empty(k, dtype=np.float64)
    cdef int64_t status
    with nogil:
        status = sdsge_lasso_gram_cd(&G[0, 0], &g[0], k, alpha, max_iter, tol,
                                     &coef[0], &Gcoef[0])
    return np.asarray(coef), status


def lars_lasso_gram(double[:, ::1] G, double[::1] c, int64_t max_iter, double tol):
    """Full LARS-Lasso path. Returns ``(lam_path, beta_path, status)``."""
    cdef int64_t k = G.shape[0]
    cdef double[::1] lam_path = np.empty(max_iter + 1, dtype=np.float64)
    cdef double[:, ::1] beta_path = np.empty((max_iter + 1, k), dtype=np.float64)
    cdef int64_t n_knots = 0
    cdef int64_t status
    with nogil:
        status = sdsge_lars_lasso_gram(&G[0, 0], &c[0], k, max_iter, tol,
                                       &lam_path[0], &beta_path[0, 0], &n_knots)
    return (
        np.asarray(lam_path[:n_knots]),
        np.asarray(beta_path[:n_knots]),
        status,
    )


def lasso_path_eval(double[::1] lam_path, double[:, ::1] beta_path,
                    double[::1] lam_grid):
    """Evaluate a LARS-Lasso path at a lambda grid. Returns ``beta_grid``."""
    cdef int64_t n_knots = lam_path.shape[0]
    cdef int64_t k = beta_path.shape[1]
    cdef int64_t n_grid = lam_grid.shape[0]
    cdef double[:, ::1] out = np.empty((n_grid, k), dtype=np.float64)
    with nogil:
        sdsge_lasso_path_eval(&lam_path[0], &beta_path[0, 0], n_knots, k,
                              &lam_grid[0], n_grid, &out[0, 0])
    return np.asarray(out)


def ols_chol_solve(double[:, ::1] X, double[::1] y):
    """OLS via the Cholesky normal equations. Returns ``(coef, L, status)``."""
    cdef int64_t n = X.shape[0]
    cdef int64_t p = X.shape[1]
    cdef double[::1] coef = np.empty(p, dtype=np.float64)
    cdef double[:, ::1] L = np.empty((p, p), dtype=np.float64)
    cdef double[:, ::1] G = np.empty((p, p), dtype=np.float64)
    cdef double[::1] g = np.empty(p, dtype=np.float64)
    cdef int64_t status = 0
    with nogil:
        sdsge_ols_chol_solve(&X[0, 0], &y[0], n, p, &coef[0], &L[0, 0],
                             &status, &G[0, 0], &g[0])
    if status != 0:
        return np.asarray(coef), np.empty((0, 0), dtype=np.float64), status
    return np.asarray(coef), np.asarray(L), status


def elastic_net_gram_cd(double[:, ::1] G, double[::1] g, double alpha_l1,
                        double alpha_l2, double[::1] beta0, int64_t max_iter,
                        double tol):
    """Elastic-net coordinate descent. Returns ``(coef, status)``."""
    cdef int64_t k = G.shape[0]
    cdef double[::1] coef = np.empty(k, dtype=np.float64)
    cdef double[::1] Gcoef = np.empty(k, dtype=np.float64)
    cdef int64_t status
    with nogil:
        status = sdsge_en_gram_cd(&G[0, 0], &g[0], k, alpha_l1, alpha_l2,
                                  &beta0[0], max_iter, tol, &coef[0], &Gcoef[0])
    return np.asarray(coef), status


def elastic_net_gram_cd_path(double[:, ::1] G, double[::1] g,
                             double[::1] alpha_grid, double l1_ratio,
                             int64_t max_iter, double tol):
    """Warm-start elastic-net path. Returns ``(coefs, statuses)``."""
    cdef int64_t k = G.shape[0]
    cdef int64_t n_alpha = alpha_grid.shape[0]
    cdef double[:, ::1] coefs = np.empty((n_alpha, k), dtype=np.float64)
    cdef int64_t[::1] statuses = np.empty(n_alpha, dtype=np.int64)
    cdef double[::1] Gcoef = np.empty(k, dtype=np.float64)
    cdef double[::1] beta = np.empty(k, dtype=np.float64)
    with nogil:
        sdsge_en_gram_cd_path(&G[0, 0], &g[0], k, &alpha_grid[0], n_alpha,
                              l1_ratio, max_iter, tol, &coefs[0, 0],
                              &statuses[0], &Gcoef[0], &beta[0])
    return np.asarray(coefs), np.asarray(statuses)


def elastic_net_active_dof(double[:, ::1] G, double[::1] beta, double alpha_l2,
                           bint intercept, double atol):
    """Active-set effective degrees of freedom. Returns a float."""
    cdef int64_t k = G.shape[0]
    cdef double dof
    with nogil:
        dof = sdsge_en_active_dof(&G[0, 0], &beta[0], k, alpha_l2, intercept, atol)
    return dof
