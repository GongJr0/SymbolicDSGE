#ifndef SDSGE_LASSO_H
#define SDSGE_LASSO_H

#include "../_common/sdsge_common.h"
#include "regression.h" /* RegressionStatus codes */

/* Lasso kernels, mirroring the numba helpers in SymbolicDSGE/regression/lasso.
 * The numba reference is compiled with fastmath=True; this C is strict IEEE, so
 * the two are NOT bit-identical -- they converge to the same (unique) lasso
 * minimizer, and the parity tests compare at the solver tolerance, not ULP. */

/* Coordinate descent on the Gram. coef(k) is the output, Gcoef(k) is caller
 * scratch. Returns REGRESSION_OK on convergence, else REGRESSION_NON_CONVERGENT.
 * G(k,k) and g(k) are the (already n-scaled) Gram and cross-product. */
i64 sdsge_lasso_gram_cd(const f64 *G, const f64 *g, i64 k, f64 alpha,
                        i64 max_iter, f64 tol, f64 *coef, f64 *Gcoef);

/* Full LARS-Lasso path on the Gram. The caller allocates lam_path(max_iter+1)
 * and beta_path((max_iter+1)*k); the number of knots actually written is
 * returned in *n_knots (slice both to that). Internal active-set scratch is
 * malloc'd. Returns REGRESSION_OK / REGRESSION_NON_CONVERGENT. */
i64 sdsge_lars_lasso_gram(const f64 *G, const f64 *c, i64 k, i64 max_iter, f64 tol,
                          f64 *lam_path, f64 *beta_path, i64 *n_knots);

/* Evaluate a LARS-Lasso path (lam_path/beta_path, n_knots rows of width k) at a
 * descending lambda grid (n_grid), writing out(n_grid, k). */
void sdsge_lasso_path_eval(const f64 *lam_path, const f64 *beta_path, i64 n_knots,
                           i64 k, const f64 *lam_grid, i64 n_grid, f64 *out);

#endif /* SDSGE_LASSO_H */
