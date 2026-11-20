#include "kde.h"
#include "../_common/kde.h"
#include "../_common/sdsge_common.h"

void sdsge_marginal_kde(const f64 *SDSGE_RESTRICT x, const i64 N,
                        const i64 n_par, const i64 n_chains, const i64 m,
                        const i64 n_workers, const f64 factor,
                        f64 *SDSGE_RESTRICT grid, f64 *SDSGE_RESTRICT density) {
  i64 S = N * n_chains;
#pragma omp parallel for num_threads((int)n_workers) if (n_par > 1)            \
    schedule(static)
  for (i64 p = 0; p < n_par; ++p) {
    sdsge_gaussian_kde1d(&x[p * S], S, m, factor, &grid[p * m],
                         &density[p * m]);
  }
}
