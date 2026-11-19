#ifndef SDSGE_MCSE_H
#define SDSGE_MCSE_H

#include "../_common/sdsge_common.h"

#define PHINEG1 0.15865525393145707
#define PHI1 0.8413447460685429

/* Monte Carlo Standard Error (MCSE) of (n_par, n_chains * N) pooled samples at
 * a quantile. Reads are parameter major, cannot raise. Failures present as NAN
 * outputs. */

arena_size sdsge_mcse_quantile_arena_size(const i64 N, const i64 n_chains,
                                          const i64 n_workers);

void sdsge_mcse_quantile(f64 *SDSGE_RESTRICT x, const i64 N, const i64 n_par,
                         const i64 n_chains, const i64 n_workers, const f64 q,
                         f64 *SDSGE_RESTRICT mcse, f64 *SDSGE_RESTRICT arena);
#endif // !SDSGE_MCSE_H
