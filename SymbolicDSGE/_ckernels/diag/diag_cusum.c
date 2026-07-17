#include "diag_cusum.h"

#include <math.h>

/* Standard-normal CDF / survival function, erf/erfc based to match the numba
 * _cdf / _sf helpers (0.5*(1+erf(x/sqrt2)) and 0.5*erfc(x/sqrt2)). */
static inline f64 norm_cdf(f64 x) { return 0.5 * (1.0 + erf(x / SQRT2)); }
static inline f64 norm_sf(f64 x) { return 0.5 * erfc(x / SQRT2); }

/* Raw Durbin boundary-crossing probability, unclamped. Internal for now: the
 * isf Newton solve (once ported) needs the raw monotone form and will promote
 * this to an exported symbol. */
static inline f64 alpha_from_a(f64 a) {
  return 2.0 * (norm_sf(2.0 * a) + exp(-4.0 * a * a) * norm_cdf(a));
}

f64 sdsge_cusum_sf(f64 a) { return min_f64(alpha_from_a(a), 1.0); }

void sdsge_cusum_sf_into(const f64 *SDSGE_RESTRICT a, i64 n,
                         f64 *SDSGE_RESTRICT out) {
  for (i64 i = 0; i < n; ++i) {
    out[i] = sdsge_cusum_sf(a[i]);
  }
}
