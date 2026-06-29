#include "regression.h"
#include <math.h>

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

  /* Effective dof = trace(G_pen⁻¹ @ G_unpen). For each column c, solve
   * (L Lᵀ) s = G_unpen[:, c] reusing the factor and accumulate s[c], the
   * c-th diagonal entry of the smoother. */
  *dof = 0.0;
  for (i64 c = 0; c < p; ++c) {
    for (i64 r = 0; r < p; ++r)
      col[r] = G_unpen[r * p + c];
    sdsge_forward_subst(L, col, col, p);
    sdsge_backward_subst_chol_t(L, col, col, p);
    *dof += col[c];
  }

  *status = REGRESSION_OK;
}
