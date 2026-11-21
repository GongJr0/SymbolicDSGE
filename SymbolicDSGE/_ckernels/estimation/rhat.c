#include "rhat.h"
#include "../_common/as241.h"
#include "../_common/sdsge_common.h"
#include "ess.h"
#include "order_stats.h"
#include <math.h>
#include <omp.h>
#include <stdint.h>

/* Split R-hat without the autocovariance. Same estimator and split as
 * sdsge_ess_rhat1d: halves of n = N/2, the last draw dropped at odd N. */
static f64 split_rhat1d(const f64 *SDSGE_RESTRICT x, const i64 N,
                        const i64 n_chains) {
  const i64 n = N / 2;
  const i64 n_sub = 2 * n_chains;

  f64 ss = 0.0;
  f64 mubar = 0.0;
  f64 m2 = 0.0;
  for (i64 c = 0; c < n_chains; ++c) {
    const f64 *a = x + c * N;
    const f64 *b = a + n;

    f64 mu_a = 0.0;
    f64 mu_b = 0.0;
    for (i64 i = 0; i < n; ++i) {
      mu_a += a[i];
      mu_b += b[i];
    }
    mu_a /= (f64)n;
    mu_b /= (f64)n;

    f64 ss_a = 0.0;
    f64 ss_b = 0.0;
    for (i64 i = 0; i < n; ++i) {
      const f64 da = a[i] - mu_a;
      const f64 db = b[i] - mu_b;
      ss_a += da * da;
      ss_b += db * db;
    }
    ss += ss_a + ss_b;

    f64 delta = mu_a - mubar;
    mubar += delta / (f64)(2 * c + 1);
    m2 += delta * (mu_a - mubar);
    delta = mu_b - mubar;
    mubar += delta / (f64)(2 * c + 2);
    m2 += delta * (mu_b - mubar);
  }

  if (!isfinite(ss) || ss <= 0.0) {
    return NAN;
  }
  const f64 w = ss / ((f64)n_sub * (f64)(n - 1));
  const f64 var_plus = ss / ((f64)n_sub * (f64)n) + m2 / (f64)(n_sub - 1);
  return sqrt(var_plus / w);
}

/* s is sorted ascending with idx carrying each value's original position;
 * z receives the normal scores in original order. Ties get the average rank. */
static void rank_normalize(const f64 *SDSGE_RESTRICT s,
                           const i64 *SDSGE_RESTRICT idx, const i64 S,
                           f64 *SDSGE_RESTRICT z) {
  const f64 denom = (f64)S + 0.25;
  i64 i = 0;
  while (i < S) {
    i64 j = i + 1;
    while (j < S && s[j] == s[i]) {
      ++j;
    }
    const f64 zval =
        sdsge_ndtri_as241((0.5 * (f64)(i + j + 1) - 0.375) / denom);
    for (i64 k = i; k < j; ++k) {
      z[idx[k]] = zval;
    }
    i = j;
  }
}

/* Normal scores of |x - median| with sorted s and idx carrying the original
 * positions. Ranks come from a merge outward from the median. */
static void fold_rank_normalize(const f64 *SDSGE_RESTRICT s,
                                const i64 *SDSGE_RESTRICT idx, const i64 S,
                                f64 *SDSGE_RESTRICT z) {
  const i64 h = S / 2;
  const f64 med = (S & 1) ? s[h] : 0.5 * (s[h - 1] + s[h]);
  const f64 denom = (f64)S + 0.25;

  i64 r = h;
  while (r > 0 && s[r - 1] >= med) {
    --r;
  }
  i64 l = r - 1;
  i64 pos = 0;
  while (l >= 0 || r < S) {
    const f64 dl = l >= 0 ? med - s[l] : INFINITY;
    const f64 dr = r < S ? s[r] - med : INFINITY;
    const f64 d = dl < dr ? dl : dr;

    const i64 l0 = l;
    const i64 r0 = r;
    while (l >= 0 && med - s[l] == d) {
      --l;
    }
    while (r < S && s[r] - med == d) {
      ++r;
    }
    const i64 g = (l0 - l) + (r - r0);
    const f64 zval =
        sdsge_ndtri_as241(((f64)pos + 0.5 * (f64)(g + 1) - 0.375) / denom);
    for (i64 k = l + 1; k <= l0; ++k) {
      z[idx[k]] = zval;
    }
    for (i64 k = r0; k < r; ++k) {
      z[idx[k]] = zval;
    }
    pos += g;
  }
}

static arena_size rank_normalized_rhat1d_arena_size(const i64 N,
                                                    const i64 n_chains) {
  i64 asz = sdsge_ess_rhat1d_arena_size(N, n_chains).n_float;
  i64 S = N * n_chains;
  return make_sizer(asz + S, S);
}
static void rank_normalized_rhat1d(f64 *SDSGE_RESTRICT x, const i64 N,
                                   const i64 n_chains,
                                   f64 *SDSGE_RESTRICT arena,
                                   i64 *SDSGE_RESTRICT iarena,
                                   f64 *SDSGE_RESTRICT folded_rhat,
                                   f64 *SDSGE_RESTRICT bulk_rhat,
                                   f64 *SDSGE_RESTRICT bulk_ess) {
  const i64 S = N * n_chains;

  f64 *aess = arena;
  f64 *z = aess + sdsge_ess_rhat1d_arena_size(N, n_chains).n_float;
  i64 *idx = iarena;

  int finite = 1;
  for (i64 i = 0; i < S; ++i) {
    idx[i] = i;
    finite &= isfinite(x[i]);
  }
  if (!finite) {
    *folded_rhat = NAN;
    *bulk_rhat = NAN;
    *bulk_ess = NAN;
    return;
  }

  sdsge_sort_pairs(x, idx, S);
  rank_normalize(x, idx, S, z);
  sdsge_ess_rhat1d(z, N, n_chains, aess, bulk_ess, bulk_rhat);
  fold_rank_normalize(x, idx, S, z);
  *folded_rhat = split_rhat1d(z, N, n_chains);
}

arena_size sdsge_rank_normalized_rhat_arena_size(const i64 N,
                                                 const i64 n_chains,
                                                 const i64 n_workers) {
  arena_size a1d = rank_normalized_rhat1d_arena_size(N, n_chains);
  return make_sizer(a1d.n_float * n_workers, a1d.n_int * n_workers);
}
void sdsge_rank_normalized_rhat(f64 *SDSGE_RESTRICT x, const i64 N,
                                const i64 n_par, const i64 n_chains,
                                const i64 n_workers, f64 *SDSGE_RESTRICT arena,
                                i64 *SDSGE_RESTRICT iarena,
                                f64 *SDSGE_RESTRICT folded_rhat,
                                f64 *SDSGE_RESTRICT bulk_rhat,
                                f64 *SDSGE_RESTRICT bulk_ess) {
  if (N < 4 || N > (INT64_MAX >> 2) + 1 || n_chains < 1) {
    for (i64 p = 0; p < n_par; ++p) {
      folded_rhat[p] = NAN;
      bulk_rhat[p] = NAN;
      bulk_ess[p] = NAN;
    }
    return;
  }
  const arena_size a1d = rank_normalized_rhat1d_arena_size(N, n_chains);
#pragma omp parallel for num_threads((int)n_workers) if (n_par > 1)            \
    schedule(static)
  for (i64 p = 0; p < n_par; ++p) {
    int wid = omp_get_thread_num();
    rank_normalized_rhat1d(&x[p * N * n_chains], N, n_chains,
                           &arena[wid * a1d.n_float], &iarena[wid * a1d.n_int],
                           &folded_rhat[p], &bulk_rhat[p], &bulk_ess[p]);
  }
}
