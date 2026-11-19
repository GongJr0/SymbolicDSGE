#include "mcse.h"
#include "../_common/betainc.h"
#include "ess.h"
#include "order_stats.h"
#include <math.h>
#include <string.h>

static arena_size sdsge_mcse_quantile1d_arena_size(const i64 N,
                                                   const i64 n_chains) {
  i64 asz = sdsge_ess_rhat1d_arena_size(N, n_chains).n_float + n_chains * N;
  asz += (8 - asz % 8) % 8;
  return make_sizer(asz, 0);
}

static f64 sdsge_mcse_quantile1d(f64 *SDSGE_RESTRICT x, const i64 N,
                                 const i64 n_chains, const f64 q,
                                 f64 *SDSGE_RESTRICT arena) {
  const i64 S = n_chains * N;
  const i64 n_sub = 2 * n_chains;
  const i64 per_sub = N / 2;
  const f64 n_trunc = (f64)(n_sub * per_sub); // Drops one at odd N

  f64 *aess = arena;
  f64 *q_scratch = aess + sdsge_ess_rhat1d_arena_size(N, n_chains).n_float;

  memcpy(q_scratch, x, S * sizeof(f64));

  const f64 x_q = sdsge_quantile1d(q_scratch, S, q);

  for (i64 c = 0; c < n_chains; ++c) {
    f64 *row = x + c * N;
    i64 s0 = 0, s1 = 0;
    for (i64 i = 0; i < per_sub; ++i) {
      const i64 v0 = (row[i] <= x_q);
      const i64 v1 = (row[i + per_sub] <= x_q);
      row[i] = v0;
      row[i + per_sub] = v1;
      s0 += v0;
      s1 += v1;
    }
    if (s0 == 0 || s0 == per_sub || s1 == 0 || s1 == per_sub) {
      return NAN;
    }
  }

  f64 ess;
  if (sdsge_ess_rhat1d(x, N, n_chains, aess, &ess, NULL) != SDSGE_ESS_OK) {
    return NAN;
  }
  ess = ess < 0.0 ? 0.0 : ess > n_trunc ? n_trunc : ess;

  const f64 lo =
      sdsge_betaincinv(ess * q + 1.0, ess * (1.0 - q) + 1.0, PHINEG1);
  const f64 hi = sdsge_betaincinv(ess * q + 1.0, ess * (1.0 - q) + 1.0, PHI1);

  i64 i_lo = (i64)floor((f64)S * lo - 1.0);
  i_lo = i_lo < 0 ? 0 : i_lo > S - 1 ? S - 1 : i_lo;

  i64 i_hi = (i64)ceil((f64)S * hi - 1.0);
  i_hi = i_hi < 0 ? 0 : i_hi > S - 1 ? S - 1 : i_hi;

  sdsge_select_kth(q_scratch, 0, S - 1, i_lo);
  const f64 x_lo = q_scratch[i_lo];
  sdsge_select_kth(q_scratch, i_lo, S - 1, i_hi);
  const f64 x_hi = q_scratch[i_hi];
  return (x_hi - x_lo) / 2.0;
}
