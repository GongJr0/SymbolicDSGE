#ifndef SDSGE_MC_REGRESSION_H
#define SDSGE_MC_REGRESSION_H

#include "../_common/sdsge_common.h"

typedef struct {
  i64 n; // number of samples
  i64 p; // number of regressors (with intercept)

  f64 *coef; // (p,)
  f64 *se;   // NULL if not OLS. (p,)

  f64 ssr;
  f64 sst;
  i64 status;
} sdsge_mc_regression_record;

/* Regression Wrappers for natve MC Steps */
void sdsge_mc_ols_fit(const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y,
                      sdsge_mc_regression_record *SDSGE_RESTRICT rec,
                      f64 *SDSGE_RESTRICT L, f64 *SDSGE_RESTRICT G,
                      f64 *SDSGE_RESTRICT g, f64 *SDSGE_RESTRICT work);

void sdsge_mc_ridge_fit(const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y,
                        f64 alpha, i64 intercept,
                        sdsge_mc_regression_record *SDSGE_RESTRICT rec,
                        f64 *SDSGE_RESTRICT L, f64 *SDSGE_RESTRICT G,
                        f64 *SDSGE_RESTRICT G_unpen, f64 *SDSGE_RESTRICT g,
                        f64 *SDSGE_RESTRICT col);

void sdsge_mc_ridge_gs_fit(
    const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y,
    const f64 *SDSGE_RESTRICT alphas, i64 n_alpha, i64 criterion,
    i64 intercept, sdsge_mc_regression_record *SDSGE_RESTRICT rec,
    f64 *SDSGE_RESTRICT G_base, f64 *SDSGE_RESTRICT G,
    f64 *SDSGE_RESTRICT L, f64 *SDSGE_RESTRICT g,
    f64 *SDSGE_RESTRICT coef_work, f64 *SDSGE_RESTRICT col);

/* Sparse estimators receive a design with an explicit leading intercept column
 * when intercept is nonzero. They center the slope Gram internally and restore
 * that intercept in rec->coef. All listed buffers are caller-owned.
 *
 * lasso_fit: G_base(p*p), G(p*p), g(p), Gcoef(p).
 * lasso_gs_fit: additionally lam_path(max_iter+1), beta_path((max_iter+1)*k),
 * beta_grid(n_alpha*k), and work(k*k + 8*k), where k = p - intercept.
 * elastic_net_fit: G_base(p*p), G(p*p), g(p), Gcoef(p).
 * elastic_net_gs_fit: beta_grid(n_alpha*k), statuses(n_alpha), Gcoef(p),
 * beta(p), and dof_work(3*k*k + k), in addition to G_base, G, and g. */
void sdsge_mc_lasso_fit(
    const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y, f64 alpha,
    i64 intercept, i64 max_iter, f64 tol,
    sdsge_mc_regression_record *SDSGE_RESTRICT rec,
    f64 *SDSGE_RESTRICT G_base, f64 *SDSGE_RESTRICT G,
    f64 *SDSGE_RESTRICT g, f64 *SDSGE_RESTRICT Gcoef);

void sdsge_mc_lasso_gs_fit(
    const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y,
    const f64 *SDSGE_RESTRICT alphas, i64 n_alpha, i64 intercept,
    i64 max_iter, f64 tol, sdsge_mc_regression_record *SDSGE_RESTRICT rec,
    f64 *SDSGE_RESTRICT G_base, f64 *SDSGE_RESTRICT G,
    f64 *SDSGE_RESTRICT g, f64 *SDSGE_RESTRICT lam_path,
    f64 *SDSGE_RESTRICT beta_path, f64 *SDSGE_RESTRICT beta_grid,
    f64 *SDSGE_RESTRICT work);

void sdsge_mc_elastic_net_fit(
    const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y, f64 alpha,
    f64 l1_ratio, i64 intercept, i64 max_iter, f64 tol,
    sdsge_mc_regression_record *SDSGE_RESTRICT rec,
    f64 *SDSGE_RESTRICT G_base, f64 *SDSGE_RESTRICT G,
    f64 *SDSGE_RESTRICT g, f64 *SDSGE_RESTRICT Gcoef);

void sdsge_mc_elastic_net_gs_fit(
    const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y,
    const f64 *SDSGE_RESTRICT alphas, i64 n_alpha, f64 l1_ratio,
    i64 criterion, i64 intercept, i64 max_iter, f64 tol,
    sdsge_mc_regression_record *SDSGE_RESTRICT rec,
    f64 *SDSGE_RESTRICT G_base, f64 *SDSGE_RESTRICT G,
    f64 *SDSGE_RESTRICT g, f64 *SDSGE_RESTRICT beta_grid,
    i64 *SDSGE_RESTRICT statuses, f64 *SDSGE_RESTRICT Gcoef,
    f64 *SDSGE_RESTRICT beta, f64 *SDSGE_RESTRICT dof_work);

#endif /* SDSGE_MC_REGRESSION_H */
