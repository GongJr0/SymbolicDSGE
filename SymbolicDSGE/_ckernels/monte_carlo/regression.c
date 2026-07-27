#include "regression.h"
#include "../_common/sdsge_linalg.h"
#include "../regression/regression.h"
#include <math.h>

void sdsge_mc_ols_fit(const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y,
                      sdsge_mc_regression_record *SDSGE_RESTRICT rec,
                      f64 *SDSGE_RESTRICT L, f64 *SDSGE_RESTRICT G,
                      f64 *SDSGE_RESTRICT g, f64 *SDSGE_RESTRICT work) {
  i64 n = rec->n;
  i64 p = rec->p;

  sdsge_ols_chol_solve(X, y, n, p, rec->coef, L, &rec->status, G, g);

  if (rec->status == REGRESSION_OK) {
    f64 ssr = 0.0;
    f64 sst = 0.0;
    f64 y_mean = 0.0;
    for (i64 i = 0; i < n; ++i) {
      y_mean += y[i];
    }
    y_mean /= (f64)n;

    for (i64 i = 0; i < n; ++i) {
      const f64 *row = X + i * p;
      f64 fitted = 0.0;
      for (i64 j = 0; j < p; ++j) {
        fitted += row[j] * rec->coef[j];
      }
      const f64 residual = y[i] - fitted;
      const f64 centered = y[i] - y_mean;
      ssr += residual * residual;
      sst += centered * centered;
    }
    rec->ssr = ssr;
    rec->sst = sst;

    if (rec->se != NULL) {
      if (n <= p) {
        for (i64 j = 0; j < p; ++j) {
          rec->se[j] = NAN;
        }
      } else {
        const f64 sigma2 = ssr / (f64)(n - p);
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
    for (i64 i = 0; i < p; i++) {
      rec->coef[i] = NAN;
    }

    rec->ssr = NAN;
    rec->sst = NAN;

    if (rec->se != NULL) {
      for (i64 i = 0; i < p; i++) {
        rec->se[i] = NAN;
      }
    }
  }
}
