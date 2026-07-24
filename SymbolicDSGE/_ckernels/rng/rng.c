#include "rng.h"

/* Direct include (not transitive via rng.h): the fill functions and npy_intp are
 * declared here, and native-include hygiene wants the header that declares each
 * used symbol included at its use site. Resolved at link time against
 * `npyrandom.lib`. */
#include "numpy/random/distributions.h"

void sdsge_rng_standard_normal_fill(bitgen_t *bg, i64 n, f64 *SDSGE_RESTRICT out) {
  if (n <= 0) {
    return;
  }
  random_standard_normal_fill(bg, (npy_intp)n, out);
}

void sdsge_rng_standard_uniform_fill(bitgen_t *bg, i64 n,
                                     f64 *SDSGE_RESTRICT out) {
  if (n <= 0) {
    return;
  }
  random_standard_uniform_fill(bg, (npy_intp)n, out);
}
