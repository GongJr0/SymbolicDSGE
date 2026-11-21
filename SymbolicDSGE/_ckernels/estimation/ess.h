#ifndef SDSGE_ESS_H
#define SDSGE_ESS_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"

#define SDSGE_ESS_OK 0
#define SDSGE_ESS_EBADSHAPE -701
#define SDSGE_ESS_EUDEFVARIANCE -702
#define SDSGE_ESS_ENODECAY -703
#define SDSGE_ESS_ENONPOSTAU -704

/* Pooled split-chain ESS and split R-hat per parameter. `x` is param-major and
 * pooled (n_par, n_chains * N). Diagnostics come off pooled 2 * n_chains
 * subchains, each one length N/2.
 *
 * ess_rhat1d: single parameter, single worker. Returns the status and writes
 * into {`ess`, `rhat`}. Passing `rhat=NULL` skips the R-hat calculation.
 *
 * ess_rhat: multiple parameters, multiple workers. Writes `status` per
 * parameter and results into {`ess`, `rhat`}. Passing `rhat=NULL` skips the
 * R-hat calculation.
 *
 * Both functions can produce a valid R-hat and fail on ESS, a nonzero status
 * doesn't automatically invalidate every output. */
arena_size sdsge_ess_rhat1d_arena_size(const i64 N, const i64 n_chains);
int sdsge_ess_rhat1d(const f64 *SDSGE_RESTRICT x, const i64 N,
                     const i64 n_chains, f64 *SDSGE_RESTRICT arena,
                     f64 *SDSGE_RESTRICT ess, f64 *SDSGE_RESTRICT rhat);

arena_size sdsge_ess_rhat_arena_size(const i64 N, const i64 n_chains,
                                     const i64 n_workers);
void sdsge_ess_rhat(const f64 *SDSGE_RESTRICT x, const i64 N, const i64 n_par,
                    const i64 n_chains, const i64 n_workers,
                    f64 *SDSGE_RESTRICT arena, f64 *SDSGE_RESTRICT ess,
                    f64 *SDSGE_RESTRICT rhat, int *SDSGE_RESTRICT status);

arena_size sdsge_ess_quantile1d_arena_size(const i64 N, const i64 n_chains);
int sdsge_ess_quantile1d(f64 *SDSGE_RESTRICT x, const i64 N, const i64 n_chains,
                         const f64 q, f64 *SDSGE_RESTRICT arena,
                         f64 *SDSGE_RESTRICT ess);

arena_size sdsge_ess_quantile_arena_size(const i64 N, const i64 n_chains,
                                         const i64 n_workers);
void sdsge_ess_quantile(f64 *SDSGE_RESTRICT x, const i64 N, const i64 n_par,
                        const i64 n_chains, const i64 n_workers, const f64 q,
                        f64 *SDSGE_RESTRICT arena, f64 *SDSGE_RESTRICT ess,
                        int *SDSGE_RESTRICT status);

#endif // !SDSGE_ESS_H
