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

#endif /* SDSGE_MC_REGRESSION_H */
