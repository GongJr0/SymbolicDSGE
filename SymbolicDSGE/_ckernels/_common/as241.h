#ifndef SDSGE_AS241_H
#define SDSGE_AS241_H

#include "sdsge_common.h"

/* Wichura's AS 241 algorithm: the inverse standard-normal CDF (probit / ndtri)
 * to full double precision, plus the erf inverse derived from it.
 *
 * Self-contained: the six rational-approximation coefficient sets (A/B, C/D,
 * E/F) are baked into as241.c, not passed in. The numba reference in
 * SymbolicDSGE/bayesian/distributions/_as241.py moved to the test oracles, so
 * this C is the sole production implementation; the parity test pins them
 * together. */

/* Inverse standard-normal CDF. Returns -INFINITY for p <= 0 and +INFINITY for
 * p >= 1, matching the numba reference. */
f64 sdsge_ndtri_as241(f64 p);

/* Elementwise ndtri over a length-n buffer (out may not alias p). Drives the
 * truncated-normal inverse-transform sampler. */
void sdsge_ndtri_as241_into(const f64 *p, i64 n, f64 *out);

/* Inverse error function, via ndtri: erfinv(y) = ndtri(0.5*(y+1)/sqrt(2)). */
f64 sdsge_erfinv_from_as241(f64 y);

#endif /* SDSGE_AS241_H */
