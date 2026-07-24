#ifndef SDSGE_RNG_H
#define SDSGE_RNG_H

#include "../_common/sdsge_common.h"

/* numpy owns this ABI type (typedef struct bitgen { ... } bitgen_t;). We only
 * ever pass the pointer through to numpy's fill functions and never dereference
 * it, so an incomplete forward declaration is sufficient. Keeping numpy's header
 * out of ours means every consumer (mcmc.h, the estimation driver) stays free of
 * numpy's build-time-only include path and resolves under static analysis. The
 * definition is completed in rng.c alone, via <numpy/random/distributions.h>. */
typedef struct bitgen bitgen_t;

/* Native RNG bridge (issue #328). Shared subsystem: any consumer that draws
 * randoms links rng via _EXTRA_DEPS, which also pulls the numpy include path and
 * the `npyrandom` link (setup.py gates both on "rng" appearing in the deps).
 *
 * The design deliberately splits the ENGINE from the TRANSFORM:
 *   - Engine (PCG64 state advance): numpy owns it. We never allocate or seed it;
 *     the `bitgen_t*` is BORROWED from a live numpy Generator's capsule, unwrapped
 *     on the Cython side. Its `next_*` function pointers advance numpy's own state,
 *     so draws share one stream with `rng.standard_normal()` / `rng.random()`.
 *   - Transform (raw bits -> uniform / ziggurat normal): numpy's compiled
 *     implementation in `npyrandom`, called here via `distributions.h`. Linking it
 *     (not reimplementing the ziggurat) is what buys bit-parity.
 *
 * Everything below is `nogil`-safe plain C over the borrowed pointer: no CPython
 * API, and no numpy PYTHON object (only the C ABI struct + linked functions). The
 * borrowed-pointer lifetime (hold the Python `rng` alive across the call) is the
 * Cython caller's responsibility, not this layer's. */

/* Fill `out[0..n)` with standard normal draws (mean 0, var 1) advancing `bg`'s
 * state; bit-identical to numpy's `random_standard_normal_fill`. */
void sdsge_rng_standard_normal_fill(bitgen_t *bg, i64 n, f64 *SDSGE_RESTRICT out);

/* Fill `out[0..n)` with standard uniform draws in [0, 1) advancing `bg`'s state;
 * bit-identical to numpy's `random_standard_uniform_fill`. */
void sdsge_rng_standard_uniform_fill(bitgen_t *bg, i64 n,
                                     f64 *SDSGE_RESTRICT out);

#endif /* SDSGE_RNG_H */
