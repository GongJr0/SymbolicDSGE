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
#include "sdsge_common.h"
#include <float.h>
#include <math.h>

static const f64 STOP = 4 * DBL_EPSILON; // 1e-8 in original source
static const f64 TINY = 1.0e-30;
static const f64 MAXITER = 2000;

f64 sdsge_betainc(f64 a, f64 b, f64 x) {

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
