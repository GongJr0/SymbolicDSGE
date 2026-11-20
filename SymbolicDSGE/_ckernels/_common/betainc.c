/* --- MODIFIED IMPLEMENTATION OF LEWIS VAN WINKLE'S INCBETA.C ---
 *
 * zlib License
 *
 * Regularized Incomplete Beta Function
 *
 * Copyright (c) 2016, 2017 Lewis Van Winkle
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */
#include "betainc.h"
#include "as241.h"
#include "sdsge_common.h"
#include <float.h>
#include <math.h>

f64 sdsge_betainc(const f64 a, const f64 b, const f64 x) {
  const f64 STOP = 4 * DBL_EPSILON; // 1e-8 in original source
  const f64 TINY = 1.0e-30;
  const f64 MAXITER = 2000;

  if (a <= 0.0 || b <= 0.0) {
    return NAN;
  }

  if (x > (a + 1) / (a + b + 2)) {
    return 1.0 - sdsge_betainc(b, a, 1.0 - x);
  }

  const f64 ln_beta = lgamma(a) + lgamma(b) - lgamma(a + b);
  const f64 front = exp(log(x) * a + log1p(-x) * b - ln_beta) / a;

  f64 f = 1.0, c = 1.0, d = 0.0;

  for (i64 i = 0; i <= MAXITER; ++i) {
    const i64 m = i / 2;
    f64 num;

    if (i == 0) {
      num = 1.0;
    } else if (i % 2 == 0) {
      num = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
    } else {
      num =
          -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
    }

    d = 1.0 + num * d;
    if (fabs(d) < TINY) {
      d = TINY;
    }
    d = 1.0 / d;
    c = 1.0 + num / c;
    if (fabs(c) < TINY) {
      c = TINY;
    }
    f64 cd = c * d;
    f *= cd;

    if (fabs(cd - 1.0) < STOP) {
      return front * (f - 1.0);
    }
  }
  return NAN; // No convergence.
}
/* --- END OF MODIFIED IMPLEMENTATION OF LEWIS VAN WINKLE'S INCBETA.C --- */

/* --- INVERSE REGULARIZED INCOMPLETE BETA FUNCTION ---
 *
 * References:
 *
 * - KL Majumder, GP Bhattacharjee, Algorithm AS 63: The incomplete Beta
 * Integral, Applied Statistics, Volume 22, Number 3, 1973, pages 409-411.
 *
 * - GW Cran, KJ Martin, GE Thomas, Remark AS R19 and Algorithm AS 109: A Remark
 * on Algorithms AS 63: The Incomplete Beta Integral and AS 64: Inverse of the
 * Incomplete Beta Integeral, Applied Statistics, Volume 26, Number 1, 1977,
 * pages 111-114.
 *
 * Note: The implementation requires (a >= 1) && (b >= 1).
 */

static f64 betaincinv_start(const f64 a, const f64 b, const f64 p) {
  const f64 y = -sdsge_ndtri_as241(p);
  const f64 lam = (y * y - 3.0) / 6.0;
  const f64 ra = 1.0 / (2.0 * a - 1.0);
  const f64 rb = 1.0 / (2.0 * b - 1.0);
  const f64 h = 2.0 / (ra + rb);
  const f64 w =
      y * sqrt(h + lam) / h - (rb - ra) * (lam + 5.0 / 6.0 - 2.0 / (3.0 * h));

  return a / (a + b * exp(2.0 * w));
}

f64 sdsge_betaincinv(f64 a, f64 b, f64 p) {
  const f64 STEP_TOL = 1.0e-14;
  const i64 MAXITER = 40;

  if (isnan(a) || isnan(b) || isnan(p) || a <= 0.0 || b <= 0.0 || p < 0.0 ||
      p > 1.0) {
    return NAN;
  }
  if (p == 0.0) {
    return 0.0;
  }
  if (p == 1.0) {
    return 1.0;
  }

  const f64 ln_beta = lgamma(a) + lgamma(b) - lgamma(a + b);

  f64 lo = 0.0;
  f64 hi = 1.0;
  f64 x = betaincinv_start(a, b, p);

  if (!(x > 0.0 && x < 1.0)) {
    x = 0.5;
  }

  for (i64 i = 0; i < MAXITER; ++i) {
    const f64 f = sdsge_betainc(a, b, x) - p;

    if (isnan(f)) {
      return NAN;
    }
    if (f == 0.0) {
      return x;
    }
    if (f > 0.0) {
      hi = x;
    } else {
      lo = x;
    }

    const f64 d1 = exp((a - 1.0) * log(x) + (b - 1.0) * log1p(-x) - ln_beta);
    f64 step;

    if (d1 > 0.0 && isfinite(d1)) {
      const f64 t = f / d1;
      const f64 r = (a - 1.0) / x - (b - 1.0) / (1.0 - x);
      const f64 den = 1.0 - 0.5 * t * r;

      step = (den > 0.0) ? -t / den : -t;

      /* Convergence settles before the bracket is consulted. A sub-ulp step
       * is the answer even though it lands on the endpoint just set from
       * sign(f); clamping it there would discard the root and leave the
       * iteration crawling with a density that has underflowed. */
      if (fabs(step) <= STEP_TOL * fabs(x)) {
        return x + step;
      }

      if (!(lo < x - t && x - t < hi) || !(lo < x + step && x + step < hi)) {
        step = 0.5 * (lo + hi) - x;
      }
    } else {
      step = 0.5 * (lo + hi) - x;
    }

    x += step;
  }
  return 0.5 * (lo + hi);
}
/* --- END OF INVERSE REGULARIZED INCOMPLETE BETA FUNCTION --- */
