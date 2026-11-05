#include "ess.h"
#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

/* In-place radix-2 FFT.
 * `inverse` != 0 selects inverse (un-normalized) transform. The 1/n
 * normalization cancels when normalizing the autocorrelation terms later. n
 * must be a power of 2.
 */
static void fft(c128 *a, i64 n, int inverse) {
  /* Bit-reversal permutation */
  for (i64 i = 1, j = 0; i < n; ++i) {
    i64 bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;

    if (i < j) {
      c128 tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
    }
  }

  for (i64 len = 2; len <= n;) {
    f64 theta = (inverse ? TWO_PI : -TWO_PI) / (f64)len;
    f64 wlen_re = cos(theta);
    f64 wlen_im = sin(theta);

    for (i64 i = 0; i < n; i += len) {
      f64 wr = 1.0, wi = 0.0;

      for (i64 j = 0; j < len / 2; ++j) {
        c128 u = a[i + j];
        c128 z = a[i + j + len / 2];

        c128 v = {z.re * wr - z.im * wi, z.re * wi + z.im * wr};

        a[i + j] = (c128){u.re + v.re, u.im + v.im};
        a[i + j + len / 2] = (c128){u.re - v.re, u.im - v.im};

        f64 next_wr = wr * wlen_re - wi * wlen_im;
        wi = wr * wlen_im + wi * wlen_re;
        wr = next_wr;
      }
    }

    if (len == n)
      break;
    len *= 2;
  }
}

static inline i64 next_pow2(i64 n) {
  i64 p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}
arena_size sdsge_acorr_fft_arena_size(const i64 n) {
  return make_sizer(2 * next_pow2(2 * n - 1), 0);
}

int sdsge_acorr_fft(const f64 *SDSGE_RESTRICT x, const i64 n,
                    f64 *SDSGE_RESTRICT arena, f64 *SDSGE_RESTRICT out) {
  if (!x || !out || n <= 0 || n > (INT64_MAX >> 2) + 1) {
    return SDSGE_ESS_EBADSHAPE;
  }

  i64 N = 2 * n - 1;
  i64 m = next_pow2(N);

  c128 *a = (c128 *)arena;
  memset(a + n, 0, (size_t)(m - n) * sizeof(c128));

  f64 mu = 0.0;
  for (i64 i = 0; i < n; ++i) {
    mu += x[i];
  }
  mu /= (f64)n;

  for (i64 i = 0; i < n; ++i) {
    a[i] = c128_make(x[i] - mu, 0.0);
  }
  fft(a, m, 0);

  for (i64 i = 0; i < m; ++i) {
    a[i].re = c128_abs2(a[i]);
    a[i].im = 0.0;
  }

  fft(a, m, 1);

  if (a[0].re <= 0.0 || !isfinite(a[0].re)) {
    for (i64 l = 0; l < n; ++l) {
      out[l] = NAN;
    }
    return SDSGE_ESS_EUDEFVARIANCE;
  }

  f64 norm = 1 / a[0].re;

  out[0] = 1.0; // x * (1/x) != 1.0 in fp
  for (i64 i = 1; i < n; ++i) {
    out[i] = a[i].re * norm;
  }
  return SDSGE_ESS_OK;
}

/*--- ESS 1D and ND ---*/

arena_size sdsge_ess_1d_arena_size(const i64 n) {
  return add_arena(sdsge_acorr_fft_arena_size(n), make_sizer(n, 0));
}

int sdsge_ess_1d(const f64 *SDSGE_RESTRICT x, const i64 n,
                 f64 *SDSGE_RESTRICT arena, f64 *SDSGE_RESTRICT out) {
  if (n < 2) {
    out[0] = NAN;
    return SDSGE_ESS_EBADSHAPE;
  }
  f64 *acorr_a = arena;
  f64 *rho = arena + sdsge_acorr_fft_arena_size(n).n_float;
  int acorr_status = sdsge_acorr_fft(x, n, acorr_a, rho);
  if (acorr_status != SDSGE_ESS_OK) {
    out[0] = NAN;
    return acorr_status;
  }

  f64 total = 0.0;
  f64 prev = INFINITY;
  f64 pair, tau;
  int decayed = 0;
  for (i64 m = 0; 2 * m + 1 < n; ++m) {
    pair = rho[2 * m] + rho[2 * m + 1];
    if (pair <= 0.0) {
      decayed = 1;
      break;
    }
    if (pair > prev) {
      pair = prev;
    }
    total += pair;
    prev = pair;
  }
  if (!decayed) {
    out[0] = NAN;
    return SDSGE_ESS_ENODECAY;
  }

  tau = -1.0 + 2.0 * total;
  if (tau <= 0.0) {
    out[0] = NAN;
    return SDSGE_ESS_ENONPOSTAU;
  }
  out[0] = (f64)n / tau;
  return SDSGE_ESS_OK;
}
