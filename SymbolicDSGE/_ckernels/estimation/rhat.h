#ifndef SDSGE_RHAT_H
#define SDSGE_RHAT_H

#include "../_common/sdsge_common.h"

arena_size sdsge_rank_normalized_rhat_arena_size(const i64 N,
                                                 const i64 n_chains,
                                                 const i64 n_workers);
void sdsge_rank_normalized_rhat(f64 *SDSGE_RESTRICT x, const i64 N,
                                const i64 n_par, const i64 n_chains,
                                const i64 n_workers, f64 *SDSGE_RESTRICT arena,
                                i64 *SDSGE_RESTRICT iarena,
                                f64 *SDSGE_RESTRICT folded_rhat,
                                f64 *SDSGE_RESTRICT bulk_rhat,
                                f64 *SDSGE_RESTRICT bulk_ess);

#endif // !SDSGE_RHAT_H
