#include "regression.h"
#include "../_common/sdsge_linalg.h"
#include "../regression/elastic_net.h"
#include "../regression/lasso.h"
#include "../regression/regression.h"
#include <math.h>
#include <stddef.h>

static void sdsge_mc_set_failure(sdsge_mc_regression_record *rec) {
  for (i64 i = 0; i < rec->p; ++i)
    rec->coef[i] = NAN;
  rec->ssr = NAN;
  rec->sst = NAN;
  if (rec->se != NULL)
    for (i64 i = 0; i < rec->p; ++i)
      rec->se[i] = NAN;
}

static void sdsge_mc_compute_ssr_sst(const f64 *X, const f64 *y,
                                     sdsge_mc_regression_record *rec) {
  const i64 n = rec->n;
  const i64 p = rec->p;
  f64 ssr = 0.0;
  f64 sst = 0.0;
  f64 y_mean = 0.0;
  for (i64 i = 0; i < n; ++i)
    y_mean += y[i];
  y_mean /= (f64)n;

  for (i64 i = 0; i < n; ++i) {
    const f64 *row = X + i * p;
    f64 fitted = 0.0;
    for (i64 j = 0; j < p; ++j)
      fitted += row[j] * rec->coef[j];
    const f64 residual = y[i] - fitted;
    const f64 centered = y[i] - y_mean;
    ssr += residual * residual;
    sst += centered * centered;
  }
  rec->ssr = ssr;
  rec->sst = sst;
}

static f64 sdsge_mc_objective(i64 criterion, f64 rss, i64 n, f64 dof) {
  if (criterion == REGRESSION_CRIT_LOSS)
    return rss;
  if (rss <= 0.0)
    return -INFINITY;
  const f64 nf = (f64)n;
  const f64 base = nf * log(rss / nf);
  return criterion == REGRESSION_CRIT_AIC ? base + 2.0 * dof
                                          : base + log(nf) * dof;
}

/* Form the n-scaled Gram and rhs used by the sparse solvers. For an explicit
 * intercept column, retain the raw first row of G_base to restore the intercept
 * after solving the centered slope problem. G and g hold the compact centered
 * slope Gram/rhs in their leading (p - 1)-square/vector entries. */
static void sdsge_mc_sparse_gram(const f64 *X, const f64 *y,
                                 const sdsge_mc_regression_record *rec,
                                 i64 intercept, f64 *G_base, f64 *G, f64 *g) {
  const i64 n = rec->n;
  const i64 p = rec->p;
  const i64 k = p - (intercept ? 1 : 0);
  const f64 nf = (f64)n;
  sdsge_gram(X, G_base, n, p);
  sdsge_gram_rhs(X, y, g, n, p);
  for (i64 i = 0; i < p * p; ++i)
    G_base[i] /= nf;
  for (i64 i = 0; i < p; ++i)
    g[i] /= nf;

  if (intercept) {
    const f64 y_mean = g[0];
    for (i64 i = 0; i < k; ++i) {
      const i64 row = i + 1;
      g[i] = g[row] - G_base[row * p] * y_mean;
      for (i64 j = 0; j < k; ++j) {
        const i64 col = j + 1;
        G[i * k + j] = G_base[row * p + col] - G_base[row * p] * G_base[col];
      }
    }
    g[k] = y_mean;
  } else {
    for (i64 i = 0; i < p * p; ++i)
      G[i] = G_base[i];
  }
}

static void sdsge_mc_restore_intercept(const sdsge_mc_regression_record *rec,
                                       const f64 *G_base, f64 y_mean) {
  f64 intercept = y_mean;
  for (i64 j = 1; j < rec->p; ++j)
    intercept -= G_base[j] * rec->coef[j];
  rec->coef[0] = intercept;
}

void sdsge_mc_ols_fit(const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y,
                      sdsge_mc_regression_record *SDSGE_RESTRICT rec,
                      f64 *SDSGE_RESTRICT L, f64 *SDSGE_RESTRICT G,
                      f64 *SDSGE_RESTRICT g, f64 *SDSGE_RESTRICT work) {
  i64 n = rec->n;
  i64 p = rec->p;

  sdsge_ols_chol_solve(X, y, n, p, rec->coef, L, &rec->status, G, g);

  if (rec->status == REGRESSION_OK) {
    sdsge_mc_compute_ssr_sst(X, y, rec);

    if (rec->se != NULL) {
      if (n <= p) {
        for (i64 j = 0; j < p; ++j) {
          rec->se[j] = NAN;
        }
      } else {
        const f64 sigma2 = rec->ssr / (f64)(n - p);
        for (i64 j = 0; j < p; ++j) {
          for (i64 i = 0; i < p; ++i) {
            work[i] = (i == j) ? 1.0 : 0.0;
          }
          sdsge_forward_subst(L, work, work, p);

          f64 inv_diag = 0.0;
          for (i64 i = 0; i < p; ++i) {
            inv_diag += work[i] * work[i];
          }
          rec->se[j] = sqrt(sigma2 * inv_diag);
        }
      }
    }
  } else {
    sdsge_mc_set_failure(rec);
  }
}

void sdsge_mc_ridge_fit(const f64 *SDSGE_RESTRICT X,
                        const f64 *SDSGE_RESTRICT y, f64 alpha, i64 intercept,
                        sdsge_mc_regression_record *SDSGE_RESTRICT rec,
                        f64 *SDSGE_RESTRICT L, f64 *SDSGE_RESTRICT G,
                        f64 *SDSGE_RESTRICT G_unpen, f64 *SDSGE_RESTRICT g,
                        f64 *SDSGE_RESTRICT col) {
  f64 dof = 0.0;
  sdsge_chol_solve_L2(X, y, rec->n, rec->p, alpha, intercept, rec->coef, L,
                      &dof, &rec->status, G, G_unpen, g, col);
  if (rec->status == REGRESSION_OK)
    sdsge_mc_compute_ssr_sst(X, y, rec);
  else
    sdsge_mc_set_failure(rec);
}

void sdsge_mc_ridge_gs_fit(
    const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y,
    const f64 *SDSGE_RESTRICT alphas, i64 n_alpha, i64 criterion, i64 intercept,
    sdsge_mc_regression_record *SDSGE_RESTRICT rec, f64 *SDSGE_RESTRICT G_base,
    f64 *SDSGE_RESTRICT G, f64 *SDSGE_RESTRICT L, f64 *SDSGE_RESTRICT g,
    f64 *SDSGE_RESTRICT coef_work, f64 *SDSGE_RESTRICT col) {
  f64 alpha = NAN;
  f64 objective = NAN;
  sdsge_ridge_grid_search(X, y, rec->n, rec->p, alphas, n_alpha, criterion,
                          intercept, &alpha, rec->coef, &objective,
                          &rec->status, G_base, G, g, L, coef_work, col);
  if (rec->status == REGRESSION_OK)
    sdsge_mc_compute_ssr_sst(X, y, rec);
  else
    sdsge_mc_set_failure(rec);
}

void sdsge_mc_lasso_fit(const f64 *SDSGE_RESTRICT X,
                        const f64 *SDSGE_RESTRICT y, f64 alpha, i64 intercept,
                        i64 max_iter, f64 tol,
                        sdsge_mc_regression_record *SDSGE_RESTRICT rec,
                        f64 *SDSGE_RESTRICT G_base, f64 *SDSGE_RESTRICT G,
                        f64 *SDSGE_RESTRICT g, f64 *SDSGE_RESTRICT Gcoef) {
  const i64 k = rec->p - (intercept ? 1 : 0);
  sdsge_mc_sparse_gram(X, y, rec, intercept, G_base, G, g);
  rec->status = sdsge_lasso_gram_cd(G, g, k, alpha, max_iter, tol,
                                    rec->coef + (intercept ? 1 : 0), Gcoef);
  if (rec->status == REGRESSION_OK) {
    if (intercept)
      sdsge_mc_restore_intercept(rec, G_base, g[k]);
    sdsge_mc_compute_ssr_sst(X, y, rec);
  } else {
    sdsge_mc_set_failure(rec);
  }
}

void sdsge_mc_lasso_gs_fit(
    const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y,
    const f64 *SDSGE_RESTRICT alphas, i64 n_alpha, i64 intercept, i64 max_iter,
    f64 tol, sdsge_mc_regression_record *SDSGE_RESTRICT rec,
    f64 *SDSGE_RESTRICT G_base, f64 *SDSGE_RESTRICT G, f64 *SDSGE_RESTRICT g,
    f64 *SDSGE_RESTRICT lam_path, f64 *SDSGE_RESTRICT beta_path,
    f64 *SDSGE_RESTRICT beta_grid, f64 *SDSGE_RESTRICT work) {
  const i64 k = rec->p - (intercept ? 1 : 0);
  i64 n_knots = 0;
  sdsge_mc_sparse_gram(X, y, rec, intercept, G_base, G, g);
  rec->status = sdsge_lars_lasso_gram(G, g, k, max_iter, tol, lam_path,
                                      beta_path, &n_knots, work);
  if (rec->status != REGRESSION_OK) {
    sdsge_mc_set_failure(rec);
    return;
  }
  sdsge_lasso_path_eval(lam_path, beta_path, n_knots, k, alphas, n_alpha,
                        beta_grid);

  f64 best_rss = INFINITY;
  for (i64 a = 0; a < n_alpha; ++a) {
    const f64 *beta = beta_grid + a * k;
    f64 rss = 0.0;
    for (i64 row = 0; row < rec->n; ++row) {
      const f64 *Xrow = X + row * rec->p;
      f64 fitted = intercept ? g[k] : 0.0;
      if (intercept)
        for (i64 j = 1; j < rec->p; ++j)
          fitted += (Xrow[j] - G_base[j]) * beta[j - 1];
      else
        for (i64 j = 0; j < rec->p; ++j)
          fitted += Xrow[j] * beta[j];
      const f64 residual = y[row] - fitted;
      rss += residual * residual;
    }
    if (a == 0 || rss < best_rss) {
      best_rss = rss;
      for (i64 j = 0; j < k; ++j)
        rec->coef[j + (intercept ? 1 : 0)] = beta[j];
    }
  }
  if (intercept)
    sdsge_mc_restore_intercept(rec, G_base, g[k]);
  sdsge_mc_compute_ssr_sst(X, y, rec);
}

void sdsge_mc_elastic_net_fit(
    const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y, f64 alpha,
    f64 l1_ratio, i64 intercept, i64 max_iter, f64 tol,
    sdsge_mc_regression_record *SDSGE_RESTRICT rec, f64 *SDSGE_RESTRICT G_base,
    f64 *SDSGE_RESTRICT G, f64 *SDSGE_RESTRICT g, f64 *SDSGE_RESTRICT Gcoef) {
  const i64 k = rec->p - (intercept ? 1 : 0);
  const f64 alpha_l1 = alpha * l1_ratio;
  const f64 alpha_l2 = alpha * (1.0 - l1_ratio);
  sdsge_mc_sparse_gram(X, y, rec, intercept, G_base, G, g);
  for (i64 j = 0; j < k; ++j)
    Gcoef[j] = 0.0;
  rec->status = sdsge_en_gram_cd(G, g, k, alpha_l1, alpha_l2, Gcoef, max_iter,
                                 tol, rec->coef + (intercept ? 1 : 0), Gcoef);
  if (rec->status == REGRESSION_OK) {
    if (intercept)
      sdsge_mc_restore_intercept(rec, G_base, g[k]);
    sdsge_mc_compute_ssr_sst(X, y, rec);
  } else {
    sdsge_mc_set_failure(rec);
  }
}

void sdsge_mc_elastic_net_gs_fit(
    const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y,
    const f64 *SDSGE_RESTRICT alphas, i64 n_alpha, f64 l1_ratio, i64 criterion,
    i64 intercept, i64 max_iter, f64 tol,
    sdsge_mc_regression_record *SDSGE_RESTRICT rec, f64 *SDSGE_RESTRICT G_base,
    f64 *SDSGE_RESTRICT G, f64 *SDSGE_RESTRICT g, f64 *SDSGE_RESTRICT beta_grid,
    i64 *SDSGE_RESTRICT statuses, f64 *SDSGE_RESTRICT Gcoef,
    f64 *SDSGE_RESTRICT beta, f64 *SDSGE_RESTRICT dof_work) {
  const i64 k = rec->p - (intercept ? 1 : 0);
  sdsge_mc_sparse_gram(X, y, rec, intercept, G_base, G, g);
  sdsge_en_gram_cd_path(G, g, k, alphas, n_alpha, l1_ratio, max_iter, tol,
                        beta_grid, statuses, Gcoef, beta);

  f64 best_objective = INFINITY;
  for (i64 a = 0; a < n_alpha; ++a) {
    if (statuses[a] != REGRESSION_OK)
      continue;
    const f64 *coef = beta_grid + a * k;
    f64 rss = 0.0;
    for (i64 row = 0; row < rec->n; ++row) {
      const f64 *Xrow = X + row * rec->p;
      f64 fitted = intercept ? g[k] : 0.0;
      if (intercept)
        for (i64 j = 1; j < rec->p; ++j)
          fitted += (Xrow[j] - G_base[j]) * coef[j - 1];
      else
        for (i64 j = 0; j < rec->p; ++j)
          fitted += Xrow[j] * coef[j];
      const f64 residual = y[row] - fitted;
      rss += residual * residual;
    }
    const f64 dof = sdsge_en_active_dof(
        G, coef, k, alphas[a] * (1.0 - l1_ratio), intercept, tol, dof_work);
    const f64 objective = sdsge_mc_objective(criterion, rss, rec->n, dof);
    if (objective < best_objective) {
      best_objective = objective;
      rec->status = REGRESSION_OK;
      for (i64 j = 0; j < k; ++j)
        rec->coef[j + (intercept ? 1 : 0)] = coef[j];
    }
  }
  if (best_objective == INFINITY) {
    rec->status = REGRESSION_NON_CONVERGENT;
    sdsge_mc_set_failure(rec);
    return;
  }
  if (intercept)
    sdsge_mc_restore_intercept(rec, G_base, g[k]);
  sdsge_mc_compute_ssr_sst(X, y, rec);
}
