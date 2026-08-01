#ifndef SDSGE_DIAG_CUSUM_H
#define SDSGE_DIAG_CUSUM_H

#include "../_common/sdsge_common.h"

/* Durbin (1969) reference distribution for the recursive-residual CUSUM
 * statistic, a parameter-free closed form. Parity oracle: the numba CusumDist
 * kernels in SymbolicDSGE/_diag_tests/cusum.py. The recursion/series/stat
 * kernels live in diag.c; this file is the p-value layer only.
 *
 * The raw Durbin form (2*(Phi_sf(2a) + exp(-4 a^2) Phi_cdf(a))) exceeds 1 for
 * small statistics, so the survival function is clamped to <= 1. */

/* Clamped Durbin survival function of the CUSUM statistic ``a`` (a >= 0). */
f64 sdsge_cusum_sf(f64 a);

/* Elementwise sf over a length-n buffer. out must not alias a. */
void sdsge_cusum_sf_into(const f64 *a, i64 n, f64 *out);

#endif /* SDSGE_DIAG_CUSUM_H */
