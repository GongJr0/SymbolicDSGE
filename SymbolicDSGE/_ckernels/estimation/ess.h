#ifndef SDSGE_ESS_H
#define SDSGE_ESS_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"

#define SDSGE_ESS_OK 0
#define SDSGE_ESS_EBADSHAPE -701
#define SDSGE_ESS_EUDEFVARIANCE -702
#define SDSGE_ESS_ENODECAY -703
#define SDSGE_ESS_ENONPOSTAU -704

arena_size sdsge_split_acov_arena_size(const i64 N);
int sdsge_split_acov(const f64 *SDSGE_RESTRICT x, const i64 N, const i64 stride,
                     f64 *SDSGE_RESTRICT arena, f64 *SDSGE_RESTRICT accum,
                     f64 *SDSGE_RESTRICT mu_out);

/* Pooled split-chain ESS and split R-hat per parameter. `x` is chain-major
 * (n_chains, N, n_par). diagnostics come off pooled 2 * n_chains subchains,
 * each one length N/2. Writes NaN into the relevant diagnostic buffer on error.
 * R-hat survives a tau failure. */
arena_size sdsge_ess_rhat_arena_size(const i64 N, const i64 n_par,
                                     const i64 n_chains);
void sdsge_ess_rhat(const f64 *SDSGE_RESTRICT x, const i64 N, const i64 n_par,
                    const i64 n_chains, const i64 n_workers,
                    f64 *SDSGE_RESTRICT arena, f64 *SDSGE_RESTRICT ess,
                    f64 *SDSGE_RESTRICT rhat, int *SDSGE_RESTRICT status);

#endif // !SDSGE_ESS_H
