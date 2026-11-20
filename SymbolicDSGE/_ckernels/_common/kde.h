#ifndef SDSGE_KDE_H
#define SDSGE_KDE_H

#include "sdsge_common.h"

/* Kernel terms below this, against a per-sample peak of at most 1, are not
 * accumulated. */
#define SDSGE_KDE_CUTOFF 0x1p-60

void sdsge_gaussian_kde1d(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 m,
                          const f64 factor, f64 *SDSGE_RESTRICT grid,
                          f64 *SDSGE_RESTRICT density);

#endif // !SDSGE_KDE_H
