#include "as241.h"

#include <math.h>

/* Rational-approximation coefficients from Wichura (1988), AS 241. Stored in
 * ascending power order (index i is the coefficient of x^i), matching the numba
 * reference's `horner(coeffs, x)` which folds coeffs[::-1]. */

/* Central region, |q| <= 0.425. */
static const f64 AS241_A[8] = {
    3.3871328727963666080,     1.3314166789178437745e2,
    1.9715909503065514427e3,   1.3731693765509461125e4,
    4.5921953931549871457e4,   6.7265770927008700853e4,
    3.3430575583588128105e4,   2.5090809287301226727e3,
};
static const f64 AS241_B[8] = {
    1.0,                       4.2313330701600911252e1,
    6.8718700749205790830e2,   5.3941960214247511077e3,
    2.1213794301586595867e4,   3.9307895800092710610e4,
    2.8729085735721942674e4,   5.2264952788528545610e3,
};

/* Intermediate tail, r <= 5. */
static const f64 AS241_C[8] = {
    1.42343711074968357734,    4.63033784615654529590,
    5.76949722146069140550,    3.64784832476320460504,
    1.27045825245236838258,    2.41780725177450611770e-1,
    2.27238449892691845833e-2, 7.74545014278341407640e-4,
};
static const f64 AS241_D[8] = {
    1.0,                       2.05319162663775882187,
    1.67638483018380384940,    6.89767334985100004550e-1,
    1.48103976427480074590e-1, 1.51986665636164571966e-2,
    5.47593808499534494600e-4, 1.05075007164441684324e-9,
};

/* Far tail, r > 5. */
static const f64 AS241_E[8] = {
    6.65790464350110377720,    5.46378491116411436990,
    1.78482653991729133580,    2.96560571828504891230e-1,
    2.65321895265761230930e-2, 1.24266094738807843860e-3,
    2.71155556874348757815e-5, 2.01033439929228813265e-7,
};
static const f64 AS241_F[8] = {
    1.0,                       5.99832206555887937690e-1,
    1.36929880922735805310e-1, 1.48753612908506148525e-2,
    7.86869131145613259100e-4, 1.84631831751005468180e-5,
    1.42151175831644588870e-7, 2.04426310338993978564e-15,
};

/* Horner evaluation of a degree-7 polynomial given ascending-order coeffs. */
static inline f64 as241_poly7(const f64 *SDSGE_RESTRICT c, f64 x) {
  f64 y = c[7];
  for (i64 i = 6; i >= 0; --i) {
    y = y * x + c[i];
  }
  return y;
}

f64 sdsge_ndtri_as241(f64 p) {
  if (p <= 0.0) {
    return -INFINITY;
  }
  if (p >= 1.0) {
    return INFINITY;
  }

  const f64 q = p - 0.5;

  if (fabs(q) <= 0.425) {
    const f64 r = 0.180625 - q * q;
    return q * as241_poly7(AS241_A, r) / as241_poly7(AS241_B, r);
  }

  f64 r = (q < 0.0) ? p : (1.0 - p);
  r = sqrt(-log(r));

  f64 x;
  if (r <= 5.0) {
    r -= 1.6;
    x = as241_poly7(AS241_C, r) / as241_poly7(AS241_D, r);
  } else {
    r -= 5.0;
    x = as241_poly7(AS241_E, r) / as241_poly7(AS241_F, r);
  }

  return (q < 0.0) ? -x : x;
}

void sdsge_ndtri_as241_into(const f64 *SDSGE_RESTRICT p, i64 n,
                            f64 *SDSGE_RESTRICT out) {
  for (i64 i = 0; i < n; ++i) {
    out[i] = sdsge_ndtri_as241(p[i]);
  }
}

f64 sdsge_erfinv_from_as241(f64 y) {
  return sdsge_ndtri_as241(0.5 * (y + 1.0) / SQRT2);
}
