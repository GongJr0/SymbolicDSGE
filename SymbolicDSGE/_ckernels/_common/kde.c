#include "kde.h"
#include "sdsge_common.h"
#include <math.h>

void sdsge_gaussian_kde1d(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 m,
                          const f64 factor, f64 *SDSGE_RESTRICT grid,
                          f64 *SDSGE_RESTRICT density) {
  f64 lo = INFINITY;
  f64 hi = -INFINITY;
  f64 mu = 0.0;
  for (i64 i = 0; i < n; ++i) {
    lo = x[i] < lo ? x[i] : lo;
    hi = x[i] > hi ? x[i] : hi;
    mu += x[i];
  }
  mu /= (f64)n;

  f64 ss = 0.0;
  for (i64 i = 0; i < n; ++i) {
    const f64 dev = x[i] - mu;
    ss += dev * dev;
  }
  const f64 h = factor * sqrt(ss / (f64)(n - 1));
  const f64 step = (hi - lo) / (f64)(m - 1);
  for (i64 j = 0; j < m; ++j) {
    grid[j] = (f64)j * step + lo;
    density[j] = 0.0;
  }
  grid[m - 1] = hi;

  if (!(h > 0.0)) { /* constant column, or non-finite input */
    for (i64 j = 0; j < m; ++j) {
      density[j] = NAN;
    }
    return;
  }

  /* Along the grid each kernel is exp of a quadratic in j: consecutive terms
   * differ by a ratio that itself shrinks by q per step. */
  const f64 inv_h = 1.0 / h;
  const f64 inv_step = 1.0 / step;
  const f64 d = step * inv_h;
  const f64 q = exp(-d * d);

  for (i64 i = 0; i < n; ++i) {
    i64 js = (i64)floor((x[i] - lo) * inv_step + 0.5);
    js = js < 0 ? 0 : (js >= m ? m - 1 : js);
    const f64 u0 = ((f64)js * step + lo - x[i]) * inv_h;
    const f64 e0 = exp(-0.5 * u0 * u0);
    density[js] += e0;

    f64 e = e0;
    f64 r = exp(-u0 * d - 0.5 * d * d);
    for (i64 j = js + 1; j < m; ++j) {
      e *= r;
      if (e < SDSGE_KDE_CUTOFF) {
        break;
      }
      density[j] += e;
      r *= q;
    }

    e = e0;
    r = exp(u0 * d - 0.5 * d * d);
    for (i64 j = js - 1; j >= 0; --j) {
      e *= r;
      if (e < SDSGE_KDE_CUTOFF) {
        break;
      }
      density[j] += e;
      r *= q;
    }
  }

  const f64 norm = SDSGE_INV_SQRT_2PI * inv_h / (f64)n;
  for (i64 j = 0; j < m; ++j) {
    density[j] *= norm;
  }
}
