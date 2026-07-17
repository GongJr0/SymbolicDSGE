#ifndef SDSGE_DIAG_CUSUM_H
#define SDSGE_DIAG_CUSUM_H

#include "../_common/sdsge_common.h"

/* Durbin (1969) reference distribution for the recursive-residual CUSUM
 * statistic. The test is parameter-free: the boundary-crossing probability is a
 * closed form of the statistic itself, so the survival function is a closed form
 * of the statistic. Mirrors the numba CusumDist kernels in
 * SymbolicDSGE/_diag_tests/cusum.py.
 *
 * The recursion/series/stat kernels for CUSUM live in diag.c; this file holds
 * only the reference-distribution (p-value) layer.
 *
 * The raw Durbin form (2*(Phi_sf(2a) + exp(-4 a^2) Phi_cdf(a))) can exceed 1 for
 * small statistics, so the survival function is clamped to <= 1 here in C -- the
 * clamp is the whole reason CusumDist.sf existed as a Python wrapper, and it
 * belongs on the native side. The isf Newton solve (once ported) will re-expose
 * the unclamped monotone form it needs. */

/* Clamped Durbin survival function of the CUSUM statistic ``a`` (a >= 0). */
f64 sdsge_cusum_sf(f64 a);

/* Elementwise sf over a length-n buffer. out must not alias a. */
void sdsge_cusum_sf_into(const f64 *a, i64 n, f64 *out);

#endif /* SDSGE_DIAG_CUSUM_H */
