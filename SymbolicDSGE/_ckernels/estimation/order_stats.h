#ifndef SDSGE_ORDER_STATS_H
#define SDSGE_ORDER_STATS_H

#include "../_common/sdsge_common.h"

/* HPD intervals over pooled posterior draws. Destructively sorts the pooled
 * draws. Instead of an arena and copy, samples are copy-on-pool and shaped
 * (n_par, S) before being passed to the C kernel */
void sdsge_hpd_intervals(f64 *SDSGE_RESTRICT pooled, const f64 alpha,
                         const i64 S, const i64 n_par, const i64 n_workers,
                         f64 *SDSGE_RESTRICT lo, f64 *SDSGE_RESTRICT hi);

#endif // !SDSGE_ORDER_STATS_H
