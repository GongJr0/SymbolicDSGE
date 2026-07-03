#include "spike.h"

/* The whole point: a plain C call through a function pointer that happens to be
 * a numba @cfunc address. No numba headers, no Python C-API, no runtime setup --
 * exactly the constraint the native preproc driver operates under. */
void spike_call(spike_residual_fn fn, const c128 *a, const c128 *b, c128 *out,
                i64 n) {
  fn(a, b, out, n);
}
