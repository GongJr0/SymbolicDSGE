#include "ess.h"
#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"
#include "order_stats.h"
#include <math.h>
#include <omp.h>
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
static arena_size sdsge_split_acov_arena_size(const i64 N) {
  const i64 n = N / 2;
  return make_sizer(2 * next_pow2(2 * n - 1), 0);
}

static void sdsge_split_acov(const f64 *SDSGE_RESTRICT x, const i64 N,
                             f64 *SDSGE_RESTRICT arena,
                             f64 *SDSGE_RESTRICT accum,
                             f64 *SDSGE_RESTRICT mu_out) {
  i64 n = N / 2;
  i64 m = next_pow2(2 * n - 1);

  c128 *a = (c128 *)arena;
  memset(a + n, 0, (size_t)(m - n) * sizeof(c128));

  f64 mu_a = 0.0;
  f64 mu_b = 0.0;
  for (i64 i = 0; i < n; ++i) {
    mu_a += x[i];
    mu_b += x[i + n];
  }
  mu_a /= (f64)n;
  mu_b /= (f64)n;

  for (i64 i = 0; i < n; ++i) {
    a[i] = c128_make(x[i] - mu_a, x[i + n] - mu_b);
  }
  fft(a, m, 0);

  { /* k = 0 and k = m/2 are self-paired: A = Re(Z), B = Im(Z) */
    const f64 p = a[0].re, q = a[0].im;
    a[0].re = p * p;
    a[0].im = q * q;

    const i64 h = m / 2;
    const f64 ph = a[h].re, qh = a[h].im;
    a[h].re = ph * ph;
    a[h].im = qh * qh;
  }
  for (i64 k = 1; k < m / 2; ++k) {
    const i64 j = m - k;
    const c128 zk = a[k], zj = a[j];
    /* conj(Z[j]) = (zj.re, -zj.im) */
    const f64 sr = zk.re + zj.re, si = zk.im - zj.im;
    const f64 dr = zk.re - zj.re, di = zk.im + zj.im;
    const f64 pa = 0.25 * (sr * sr + si * si);
    const f64 pb = 0.25 * (dr * dr + di * di);
    a[k].re = pa;
    a[k].im = pb;
    a[j].re = pa;
    a[j].im = pb;
  }

  fft(a, m, 1);

  for (i64 l = 0; l < n; ++l) {
    accum[l] += a[l].re + a[l].im;
  }
  mu_out[0] = mu_a;
  mu_out[1] = mu_b;
}

/*--- ESS ---*/

/* Geyer's initial positive sequence with the monotone clamp applied as a
 * running minimum. `rho` is read in pairs, so it carries normalized
 * autocorrelations whatever arity produced them. */
static int geyer_tau(const f64 *SDSGE_RESTRICT rho, const i64 n,
                     f64 *SDSGE_RESTRICT tau_out) {
  f64 total = 0.0;
  f64 prev = INFINITY;
  int decayed = 0;
  for (i64 k = 0; 2 * k + 1 < n; ++k) {
    f64 pair = rho[2 * k] + rho[2 * k + 1];
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
    return SDSGE_ESS_ENODECAY;
  }
  const f64 tau = -1.0 + 2.0 * total;
  if (tau <= 0.0) {
    return SDSGE_ESS_ENONPOSTAU;
  }
  *tau_out = tau;
  return SDSGE_ESS_OK;
}

arena_size sdsge_ess_rhat1d_arena_size(const i64 N, const i64 n_chains) {
  i64 asz = sdsge_split_acov_arena_size(N).n_float + N / 2 + 2 * n_chains;
  asz += (8 - asz % 8) % 8; /* round up to multiple of 8 for alignment */
  return make_sizer(asz, 0);
}

int sdsge_ess_rhat1d(const f64 *SDSGE_RESTRICT x, const i64 N,
                     const i64 n_chains, f64 *SDSGE_RESTRICT arena,
                     f64 *SDSGE_RESTRICT ess, f64 *SDSGE_RESTRICT rhat) {
  if (N < 4 || N > (INT64_MAX >> 2) + 1 || n_chains < 1) {
    *ess = NAN;
    if (rhat) {
      *rhat = NAN;
    }
    return SDSGE_ESS_EBADSHAPE;
  }
  const i64 n = N / 2;
  const i64 m = next_pow2(2 * n - 1);
  const i64 n_sub = 2 * n_chains;
  const i64 acova = sdsge_split_acov_arena_size(N).n_float;

  const f64 acov_scale = 1.0 / ((f64)n_sub * (f64)m * (f64)n);
  const f64 w_scale = 1.0 / ((f64)n_sub * (f64)m * (f64)(n - 1));

  f64 *a = arena;
  f64 *accum = a + acova;
  f64 *mus = accum + n;
  memset(accum, 0, (size_t)n * sizeof(f64));

  *ess = NAN;
  if (rhat) {
    *rhat = NAN;
  }

  for (i64 c = 0; c < n_chains; ++c) {
    sdsge_split_acov(&x[c * N], N, a, accum, mus + 2 * c);
  }
  if (!isfinite(accum[0]) || accum[0] <= 0.0) {
    return SDSGE_ESS_EUDEFVARIANCE;
  }
  f64 mubar = 0.0;
  for (i64 i = 0; i < n_sub; ++i) {
    mubar += mus[i];
  }
  mubar /= (f64)n_sub;

  f64 b = 0.0;
  for (i64 i = 0; i < n_sub; ++i) {
    const f64 d = mus[i] - mubar;
    b += d * d;
  }
  b /= (f64)(n_sub - 1);

  const f64 w = accum[0] * w_scale;
  const f64 var_plus = accum[0] * acov_scale + b;

  if (rhat) {
    *rhat = sqrt(var_plus / w);
  }

  for (i64 l = 1; l < n; ++l) {
    accum[l] = 1.0 - (w - accum[l] * acov_scale) / var_plus;
  }
  accum[0] = 1.0;

  f64 tau = 0.0;
  int rc = geyer_tau(accum, n, &tau);
  if (rc != SDSGE_ESS_OK) {
    return rc;
  }
  *ess = (f64)(n_sub * n) / tau;

  return SDSGE_ESS_OK;
}

arena_size sdsge_ess_rhat_arena_size(const i64 N, const i64 n_chains,
                                     const i64 n_workers) {
  return make_sizer(
      n_workers * sdsge_ess_rhat1d_arena_size(N, n_chains).n_float, 0);
}

void sdsge_ess_rhat(const f64 *SDSGE_RESTRICT x, const i64 N, const i64 n_par,
                    const i64 n_chains, const i64 n_workers,
                    f64 *SDSGE_RESTRICT arena, f64 *SDSGE_RESTRICT ess,
                    f64 *SDSGE_RESTRICT rhat, int *SDSGE_RESTRICT status) {
  if (N < 4 || N > (INT64_MAX >> 2) + 1 || n_par < 1 || n_chains < 1) {
    for (i64 p = 0; p < n_par; ++p) {
      ess[p] = NAN;
      if (rhat) {
        rhat[p] = NAN;
      }
      status[p] = SDSGE_ESS_EBADSHAPE;
    }
    return;
  }
  i64 per_par = sdsge_ess_rhat1d_arena_size(N, n_chains).n_float;
#pragma omp parallel for num_threads((int)n_workers) if (n_par > 1)            \
    schedule(static)
  for (i64 p = 0; p < n_par; ++p) {
    const i64 wid = omp_get_thread_num();
    f64 *pa = arena + wid * per_par;
    status[p] = sdsge_ess_rhat1d(&x[p * n_chains * N], N, n_chains, pa, &ess[p],
                                 rhat ? &rhat[p] : NULL);
  }
}

arena_size sdsge_ess_quantile1d_arena_size(const i64 N, const i64 n_chains) {
  i64 asz = sdsge_ess_rhat1d_arena_size(N, n_chains).n_float + n_chains * N;
  asz += (8 - asz % 8) % 8; /* round up to multiple of 8 for alignment */
  return make_sizer(asz, 0);
}

int sdsge_ess_quantile1d(f64 *SDSGE_RESTRICT x, const i64 N, const i64 n_chains,
                         const f64 q, f64 *SDSGE_RESTRICT arena,
                         f64 *SDSGE_RESTRICT ess) {
  if (N < 4 || N > (INT64_MAX >> 2) + 1 || n_chains < 1) {
    *ess = NAN;
    return SDSGE_ESS_EBADSHAPE;
  }
  const i64 S = N * n_chains;
  const i64 per_sub = N / 2;
  const f64 trunc = S * log10((f64)S);

  f64 *aess = arena;
  f64 *q_scratch = aess + sdsge_ess_rhat1d_arena_size(N, n_chains).n_float;
  memcpy(q_scratch, x, S * sizeof(f64));

  const f64 x_q = sdsge_quantile1d(q_scratch, S, q);

  for (i64 c = 0; c < n_chains; ++c) {
    f64 *chain = x + c * N;
    i64 s0 = 0, s1 = 0;
    for (i64 i = 0; i < per_sub; ++i) {
      const i64 v0 = (chain[i] <= x_q);
      const i64 v1 = (chain[i + per_sub] <= x_q);
      chain[i] = v0;
      chain[i + per_sub] = v1;
      s0 += v0;
      s1 += v1;
    }
    if (s0 == 0 || s0 == per_sub || s1 == 0 || s1 == per_sub) {
      *ess = NAN;
      return SDSGE_ESS_EUDEFVARIANCE;
    }
  }
  int rc = sdsge_ess_rhat1d(x, N, n_chains, aess, ess, NULL);
  if (rc != SDSGE_ESS_OK) {
    *ess = NAN;
    return rc;
  }
  *ess = *ess < 0.0 ? 0.0 : *ess > trunc ? trunc : *ess;
  return SDSGE_ESS_OK;
}

arena_size sdsge_ess_quantile_arena_size(const i64 N, const i64 n_chains,
                                         const i64 n_workers) {
  const i64 asz = sdsge_ess_quantile1d_arena_size(N, n_chains).n_float;
  return make_sizer(n_workers * asz, 0);
}
void sdsge_ess_quantile(f64 *SDSGE_RESTRICT x, const i64 N, const i64 n_par,
                        const i64 n_chains, const i64 n_workers, const f64 q,
                        f64 *SDSGE_RESTRICT arena, f64 *SDSGE_RESTRICT ess,
                        int *SDSGE_RESTRICT status) {
  if (N < 4 || N > (INT64_MAX >> 2) + 1 || n_par < 1 || n_chains < 1) {
    for (i64 p = 0; p < n_par; ++p) {
      ess[p] = NAN;
      status[p] = SDSGE_ESS_EBADSHAPE;
    }
    return;
  }
  i64 per_par = sdsge_ess_quantile1d_arena_size(N, n_chains).n_float;
#pragma omp parallel for num_threads((int)n_workers) if (n_par > 1)            \
    schedule(static)
  for (i64 p = 0; p < n_par; ++p) {
    const i64 wid = omp_get_thread_num();
    f64 *pa = arena + wid * per_par;
    status[p] =
        sdsge_ess_quantile1d(&x[p * n_chains * N], N, n_chains, q, pa, &ess[p]);
  }
}
