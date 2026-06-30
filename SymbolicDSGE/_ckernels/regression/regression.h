#ifndef SDSGE_REGRESSION_H
#define SDSGE_REGRESSION_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_linalg.h"

typedef enum {
  REGRESSION_OK = 0,
  REGRESSION_RANK_DEFICIENT = -1,
  REGRESSION_NON_CONVERGENT = -2,
} RegressionStatus;

/* Grid-search information criteria. Mirror the numba objective functions in
 * SymbolicDSGE/regression/utils.py (aic / bic / l2_loss); the Python dispatcher
 * maps the "aic"/"bic"/"loss" literal onto these codes. */
typedef enum {
  REGRESSION_CRIT_AIC = 1,
  REGRESSION_CRIT_BIC = 2,
  REGRESSION_CRIT_LOSS = 3,
} RegressionCriterion;

/* Ridge (L2) normal-equation solve. Mirrors the numba ``chol_solve_L2``;
 * writes coef(p), the Cholesky factor L(p,p) of the penalized Gram, the
 * effective dof, and a RegressionStatus into *status. G/G_unpen/g/col are
 * caller-allocated scratch of sizes (p,p)/(p,p)/(p)/(p). See regression.c. */
void sdsge_chol_solve_L2(const f64 *X, const f64 *y, i64 n, i64 p, f64 alpha,
                         i64 intercept, f64 *coef, f64 *L, f64 *dof,
                         i64 *status, f64 *G, f64 *G_unpen, f64 *g, f64 *col);

/* Ridge grid search over a precomputed alpha grid. Mirrors the numba
 * ``l2_grid_search``: forms the Gram ONCE (it is alpha-invariant), then per
 * alpha re-applies the diagonal penalty, factors, solves, scores via the
 * selected criterion, and keeps the argmin. Writes the winning alpha, coef(p),
 * objective value, and RegressionStatus. alphas(num) is caller-supplied (built
 * by the numba log_grid). Scratch: G_base/G/L (p,p), g/coef/col (p). */
void sdsge_ridge_grid_search(const f64 *X, const f64 *y, i64 n, i64 p,
                             const f64 *alphas, i64 num, i64 criterion,
                             i64 intercept, f64 *out_alpha, f64 *out_coef,
                             f64 *out_obj, i64 *out_status, f64 *G_base, f64 *G,
                             f64 *g, f64 *L, f64 *coef, f64 *col);

#endif /* SDSGE_REGRESSION_H */
