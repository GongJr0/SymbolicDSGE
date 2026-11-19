#ifndef SDSGE_ORDER_STATS_H
#define SDSGE_ORDER_STATS_H

#include "../_common/sdsge_common.h"

/* HPD intervals over pooled posterior draws. Destructively sorts the pooled
 * draws. Instead of an arena and copy, samples are copy-on-pool and shaped
 * (n_par, S) before being passed to the C kernel */
void sdsge_hpd_intervals(f64 *SDSGE_RESTRICT pooled, const f64 alpha,
                         const i64 S, const i64 n_par, const i64 n_workers,
                         f64 *SDSGE_RESTRICT lo, f64 *SDSGE_RESTRICT hi);

/* helpers at mcse.c */
void sdsge_select_kth(f64 *SDSGE_RESTRICT a, i64 lo, i64 hi, const i64 k);
f64 sdsge_quantile1d(f64 *SDSGE_RESTRICT buf, const i64 S, const f64 q);

#endif // !SDSGE_ORDER_STATS_H
