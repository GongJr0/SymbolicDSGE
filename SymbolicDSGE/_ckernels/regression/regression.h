#ifndef SDSGE_REGRESSION_H
#define SDSGE_REGRESSION_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_linalg.h"

typedef enum {
  REGRESSION_OK = 0,
  REGRESSION_RANK_DEFICIENT = -1,
  REGRESSION_NON_CONVERGENT = -2,
} RegressionStatus;

/* Ridge (L2) normal-equation solve. Mirrors the numba ``chol_solve_L2``;
 * writes coef(p), the Cholesky factor L(p,p) of the penalized Gram, the
 * effective dof, and a RegressionStatus into *status. G/G_unpen/g/col are
 * caller-allocated scratch of sizes (p,p)/(p,p)/(p)/(p). See regression.c. */
void sdsge_chol_solve_L2(const f64 *X, const f64 *y, i64 n, i64 p, f64 alpha,
                         i64 intercept, f64 *coef, f64 *L, f64 *dof,
                         i64 *status, f64 *G, f64 *G_unpen, f64 *g, f64 *col);

#endif /* SDSGE_REGRESSION_H */
