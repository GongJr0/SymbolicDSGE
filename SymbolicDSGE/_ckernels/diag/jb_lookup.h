#ifndef SDSGE_DIAG_JB_LOOKUP_H
#define SDSGE_DIAG_JB_LOOKUP_H

#include "../_common/sdsge_common.h"

/* Small-N Jarque-Bera reference distribution: table lookup + bilinear
 * interpolation over the Wuertz-Keller (2004) finite-sample critical-value
 * grid. Mirrors the numba kernels that used to live in
 * SymbolicDSGE/_diag_tests/jb_lookup.py.
 *
 * The (constant) N grid, p-value grid, and critical-value matrix are compiled
 * in as static const tables in jb_lookup.c; the Python module keeps its own
 * numpy copies purely for the distribution's small-N boundary check and the
 * parity tests. There is no rank-deficiency / fallback path here: the kernels
 * are pure interpolation, so the Python side is hard-native (no numba mirror).
 *
 * ``isf`` maps (n, p) -> critical value; ``pval`` maps (n, x) -> p-value. Each
 * has a scalar and an ``_into`` array form. The ``find_hilo`` helpers are the
 * bracketing primitives (ascending grid / descending critical-value column),
 * exposed so the parity tests can pin them directly. */

/* Bracket ``val`` in the ascending array ``arr`` (length n). Writes the pair of
 * indices (lo, hi): (0,0) below the grid, (n-1,n-1) above it, (idx,idx) on an
 * exact hit, else the straddling (idx-1, idx). Matches numpy searchsorted with
 * side='left'. */
void sdsge_jb_find_hilo_ascending(f64 val, const f64 *SDSGE_RESTRICT arr, i64 n,
                                  i64 *SDSGE_RESTRICT lo,
                                  i64 *SDSGE_RESTRICT hi);

/* As above for a strictly descending array (critical values shrink as the
 * p-value grows). Mirrors searchsorted(-arr, -val). */
void sdsge_jb_find_hilo_descending(f64 val, const f64 *SDSGE_RESTRICT arr,
                                   i64 n, i64 *SDSGE_RESTRICT lo,
                                   i64 *SDSGE_RESTRICT hi);

/* Inverse survival function: critical value at sample size ``n`` and upper-tail
 * probability ``p``. Returns NaN for NaN p, +inf for p<=0, 0 for p>=1. */
f64 sdsge_jb_isf_interp(i64 n, f64 p);

/* Survival function: upper-tail probability at sample size ``n`` and statistic
 * ``x``. Returns NaN for NaN x, 1 for x<=0, 0 for x=+inf. */
f64 sdsge_jb_pval_interp(i64 n, f64 x);

/* Elementwise isf over p(m) -> out(m). out must not alias p. */
void sdsge_jb_isf_interp_into(i64 n, const f64 *SDSGE_RESTRICT p, i64 m,
                              f64 *SDSGE_RESTRICT out);

/* Elementwise pval over x(m) -> out(m). out must not alias x. */
void sdsge_jb_pval_interp_into(i64 n, const f64 *SDSGE_RESTRICT x, i64 m,
                               f64 *SDSGE_RESTRICT out);

#endif /* SDSGE_DIAG_JB_LOOKUP_H */
