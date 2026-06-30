#include "regression.h"
#include <math.h>

/* Information criterion score, mirroring the numba aic / bic / l2_loss in
 * SymbolicDSGE/regression/utils.py. l2_loss returns the raw rss regardless of
 * sign; aic/bic return -inf when rss <= 0 (matching the numba guard). k is the
 * effective degrees of freedom. */
static f64 sdsge_ridge_objective(i64 criterion, f64 rss, i64 n, f64 k) {
  if (criterion == REGRESSION_CRIT_LOSS)
    return rss;
  if (rss <= 0.0)
    return -INFINITY;
  const f64 nf = (f64)n;
  const f64 base = nf * log(rss / nf);
  if (criterion == REGRESSION_CRIT_AIC)
    return base + 2.0 * k;
  return base + log(nf) * k; /* REGRESSION_CRIT_BIC */
}

/* trace(G_pen^-1 @ G_unpen): for each column c, solve (L L^T) s = G_unpen[:, c]
 * reusing the factor L and accumulate s[c]. col(p) is scratch. */
static f64 sdsge_ridge_eff_dof(const f64 *SDSGE_RESTRICT L,
                               const f64 *SDSGE_RESTRICT G_unpen, i64 p,
                               f64 *SDSGE_RESTRICT col) {
  f64 dof = 0.0;
  for (i64 c = 0; c < p; ++c) {
    for (i64 r = 0; r < p; ++r)
      col[r] = G_unpen[r * p + c];
    sdsge_forward_subst(L, col, col, p);
    sdsge_backward_subst_chol_t(L, col, col, p);
    dof += col[c];
  }
  return dof;
}

/* Ridge (L2) normal-equation solve, mirroring the numba ``chol_solve_L2`` in
 * SymbolicDSGE/regression/solvers.py. Forms the n-scaled Gram, adds the L2
 * penalty to the diagonal (skipping the intercept column when present), factors
 * the penalized Gram once, and recovers both the coefficients and the effective
 * degrees of freedom (the trace of the ridge smoother) from that single factor.
 *
 * The Gram primitives and Cholesky live in _common/sdsge_linalg; sdsge_gram is
 * documented to accumulate in the same order as the numba xtx_xty manual branch,
 * so this stays bit-parity with the reference for the small/medium designs the
 * regression path uses (the numba BLAS branch at n>=1e5 / p>=100 reorders the
 * summation -- parity there is not claimed).
 *
 * All buffers are caller-allocated (no VLAs / malloc, MSVC-portable):
 *   X(n,p) row-major, y(n)                       -- inputs
 *   coef(p)                                       -- out: ridge coefficients
 *   L(p,p)                                        -- out: Cholesky factor of the
 *                                                    penalized Gram (valid only
 *                                                    when *status == OK)
 *   dof, status                                   -- out scalars
 *   G(p,p), G_unpen(p,p), g(p), col(p)            -- scratch
 *
 * On a non-positive-definite penalized Gram, coef is filled with NaN, *dof is
 * NaN, and *status is REGRESSION_RANK_DEFICIENT (the caller falls back to lstsq,
 * matching the numba try/except contract). L is left undefined in that case. */
void sdsge_chol_solve_L2(const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y,
                         i64 n, i64 p, f64 alpha, i64 intercept,
                         f64 *SDSGE_RESTRICT coef, f64 *SDSGE_RESTRICT L,
                         f64 *SDSGE_RESTRICT dof, i64 *SDSGE_RESTRICT status,
                         f64 *SDSGE_RESTRICT G, f64 *SDSGE_RESTRICT G_unpen,
                         f64 *SDSGE_RESTRICT g, f64 *SDSGE_RESTRICT col) {
  const f64 nf = (f64)n;

  /* G := XᵀX / n, g := Xᵀy / n. */
  sdsge_gram(X, G, n, p);
  sdsge_gram_rhs(X, y, g, n, p);
  for (i64 i = 0; i < p * p; ++i)
    G[i] /= nf;
  for (i64 i = 0; i < p; ++i)
    g[i] /= nf;

  /* Keep the unpenalized Gram for the effective-dof smoother trace. */
  for (i64 i = 0; i < p * p; ++i)
    G_unpen[i] = G[i];

  /* L2 penalty on the diagonal; skip the intercept column when present. */
  for (i64 i = (intercept ? 1 : 0); i < p; ++i)
    G[i * p + i] += alpha;

  /* Factor the penalized Gram once: G = L Lᵀ, then coef = G⁻¹ g. */
  if (sdsge_chol(G, 0.0, L, p) != SDSGE_OK) {
    for (i64 i = 0; i < p; ++i)
      coef[i] = NAN;
    *dof = NAN;
    *status = REGRESSION_RANK_DEFICIENT;
    return;
  }
  sdsge_forward_subst(L, g, coef, p);
  sdsge_backward_subst_chol_t(L, coef, coef, p);

  /* Effective dof = trace(G_pen⁻¹ @ G_unpen), the ridge smoother trace. */
  *dof = sdsge_ridge_eff_dof(L, G_unpen, p, col);

  *status = REGRESSION_OK;
}

void sdsge_ridge_grid_search(const f64 *SDSGE_RESTRICT X,
                             const f64 *SDSGE_RESTRICT y, i64 n, i64 p,
                             const f64 *SDSGE_RESTRICT alphas, i64 num,
                             i64 criterion, i64 intercept,
                             f64 *SDSGE_RESTRICT out_alpha,
                             f64 *SDSGE_RESTRICT out_coef,
                             f64 *SDSGE_RESTRICT out_obj,
                             i64 *SDSGE_RESTRICT out_status,
                             f64 *SDSGE_RESTRICT G_base, f64 *SDSGE_RESTRICT G,
                             f64 *SDSGE_RESTRICT g, f64 *SDSGE_RESTRICT L,
                             f64 *SDSGE_RESTRICT coef, f64 *SDSGE_RESTRICT col) {
  const f64 nf = (f64)n;

  /* Gram is alpha-invariant -- form it (and the rhs) once. G_base doubles as the
   * unpenalized Gram for the effective-dof smoother trace. */
  sdsge_gram(X, G_base, n, p);
  sdsge_gram_rhs(X, y, g, n, p);
  for (i64 i = 0; i < p * p; ++i)
    G_base[i] /= nf;
  for (i64 i = 0; i < p; ++i)
    g[i] /= nf;

  f64 best_obj = INFINITY;

  for (i64 a = 0; a < num; ++a) {
    const f64 alpha = alphas[a];
    f64 obj;
    i64 status;

    /* G := G_base + alpha on the (non-intercept) diagonal. */
    for (i64 i = 0; i < p * p; ++i)
      G[i] = G_base[i];
    for (i64 i = (intercept ? 1 : 0); i < p; ++i)
      G[i * p + i] += alpha;

    if (sdsge_chol(G, 0.0, L, p) != SDSGE_OK) {
      for (i64 j = 0; j < p; ++j)
        coef[j] = NAN;
      obj = INFINITY;
      status = REGRESSION_RANK_DEFICIENT;
    } else {
      sdsge_forward_subst(L, g, coef, p);
      sdsge_backward_subst_chol_t(L, coef, coef, p);
      const f64 dof = sdsge_ridge_eff_dof(L, G_base, p, col);

      /* rss = sum_r (y_r - X_r . coef)^2 (manual reduction, matching the numba
       * loop branch). */
      f64 rss = 0.0;
      for (i64 r = 0; r < n; ++r) {
        const f64 *Xr = X + r * p;
        f64 yhat = 0.0;
        for (i64 j = 0; j < p; ++j)
          yhat += Xr[j] * coef[j];
        const f64 resid = y[r] - yhat;
        rss += resid * resid;
      }
      obj = sdsge_ridge_objective(criterion, rss, n, dof);
      status = REGRESSION_OK;
    }

    /* argmin with first-wins ties (matches numpy.argmin). On the winning index
     * snapshot the coef/alpha/obj/status; failed alphas carry NaN coef + inf
     * obj, exactly like the numba traces. */
    if (a == 0 || obj < best_obj) {
      best_obj = obj;
      *out_alpha = alpha;
      *out_obj = obj;
      *out_status = status;
      for (i64 j = 0; j < p; ++j)
        out_coef[j] = coef[j];
    }
  }
}

void sdsge_ols_chol_solve(const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y,
                          i64 n, i64 p, f64 *SDSGE_RESTRICT coef,
                          f64 *SDSGE_RESTRICT L, i64 *SDSGE_RESTRICT status,
                          f64 *SDSGE_RESTRICT G, f64 *SDSGE_RESTRICT g) {
  /* OLS normal equations: G = XᵀX, g = Xᵀy, then a single Cholesky solve. Unlike
   * the numba chol_solve (a hand-rolled PD gate followed by a *second* LAPACK
   * factorization), sdsge_chol_solve folds the PD check into the one factor. On a
   * non-PD Gram the caller falls back to lstsq, matching the numba contract. */
  sdsge_gram(X, G, n, p);
  sdsge_gram_rhs(X, y, g, n, p);
  if (sdsge_chol_solve(G, g, coef, L, p) != SDSGE_OK) {
    for (i64 i = 0; i < p; ++i)
      coef[i] = NAN;
    *status = REGRESSION_RANK_DEFICIENT;
    return;
  }
  *status = REGRESSION_OK;
}
