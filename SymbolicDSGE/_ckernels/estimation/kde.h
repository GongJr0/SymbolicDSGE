#ifndef SDSGE_ESTIM_KDE_H
#define SDSGE_ESTIM_KDE_H

#include "../_common/sdsge_common.h"

/* Parallel 1D KDE over flat array of (n_par, N * n_chains) samples.
 * Returns NAN when variance is undefined, does not shape check. */
void sdsge_marginal_kde(const f64 *SDSGE_RESTRICT x, const i64 N,
                        const i64 n_par, const i64 n_chains, const i64 m,
                        const i64 n_workers, const f64 factor,
                        f64 *SDSGE_RESTRICT grid, f64 *SDSGE_RESTRICT density);

#endif // !SDSGE_ESTIM_KDE_H
