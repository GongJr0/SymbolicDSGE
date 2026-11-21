#include "mcse.h"
#include "../_common/betainc.h"
#include "ess.h"
#include "order_stats.h"
#include <math.h>
#include <omp.h>

static arena_size sdsge_mcse_quantile1d_arena_size(const i64 N,
                                                   const i64 n_chains) {
  return sdsge_ess_quantile1d_arena_size(N, n_chains);
}

static f64 sdsge_mcse_quantile1d(f64 *SDSGE_RESTRICT x, const i64 N,
                                 const i64 n_chains, const f64 q,
                                 f64 *SDSGE_RESTRICT arena) {
  const i64 S = n_chains * N;

  f64 ess = NAN;
  int rc = sdsge_ess_quantile1d(x, N, n_chains, q, arena, &ess);
  if (rc != SDSGE_ESS_OK) {
    return NAN;
  }

  /* The quantile ESS arena is (ess, q_scratch); the latter holds partitioned
   * values after the quantile ESS is computed. Indexing into the arena avoids a
   * second call and copy. */
  f64 *partitioned = arena + sdsge_ess_rhat1d_arena_size(N, n_chains).n_float;

  const f64 lo =
      sdsge_betaincinv(ess * q + 1.0, ess * (1.0 - q) + 1.0, PHINEG1);
  const f64 hi = sdsge_betaincinv(ess * q + 1.0, ess * (1.0 - q) + 1.0, PHI1);

  i64 i_lo = (i64)floor((f64)S * lo - 1.0);
  i_lo = i_lo < 0 ? 0 : i_lo > S - 1 ? S - 1 : i_lo;

  i64 i_hi = (i64)ceil((f64)S * hi - 1.0);
  i_hi = i_hi < 0 ? 0 : i_hi > S - 1 ? S - 1 : i_hi;

  sdsge_select_kth(partitioned, 0, S - 1, i_lo);
  const f64 x_lo = partitioned[i_lo];
  sdsge_select_kth(partitioned, i_lo, S - 1, i_hi);
  const f64 x_hi = partitioned[i_hi];
  return (x_hi - x_lo) / 2.0;
}

arena_size sdsge_mcse_quantile_arena_size(const i64 N, const i64 n_chains,
                                          const i64 n_workers) {
  const i64 asz = sdsge_mcse_quantile1d_arena_size(N, n_chains).n_float;
  return make_sizer(n_workers * asz, 0);
}

void sdsge_mcse_quantile(f64 *SDSGE_RESTRICT x, const i64 N, const i64 n_par,
                         const i64 n_chains, const i64 n_workers, const f64 q,
                         f64 *SDSGE_RESTRICT mcse, f64 *SDSGE_RESTRICT arena) {
  const i64 per_par = sdsge_mcse_quantile1d_arena_size(N, n_chains).n_float;

#pragma omp parallel for num_threads((int)n_workers) if (n_par > 1)            \
    schedule(static)
  for (i64 p = 0; p < n_par; ++p) {
    i64 wid = omp_get_thread_num();
    mcse[p] = sdsge_mcse_quantile1d(&x[p * N * n_chains], N, n_chains, q,
                                    &arena[wid * per_par]);
  }
}
