#include "residual_path.h"
#include <stdlib.h>

i64 sdsge_residual_path(sdsge_residual_fn resid, const c128 *SDSGE_RESTRICT cur,
                        const c128 *SDSGE_RESTRICT fwd,
                        const c128 *SDSGE_RESTRICT par, i64 n_steps, i64 n_var,
                        i64 n_eq, f64 *SDSGE_RESTRICT residuals) {
  /* One scratch output vector, reused across the path (owned here). */
  c128 *out = (c128 *)malloc((size_t)(n_eq > 0 ? n_eq : 1) * sizeof(c128));
  if (!out) {
    return SDSGE_RESIDUAL_PATH_ALLOC_FAIL;
  }

  for (i64 t = 0; t < n_steps; ++t) {
    resid(&fwd[t * n_var], &cur[t * n_var], par, out);
    for (i64 k = 0; k < n_eq; ++k) {
      residuals[t * n_eq + k] = c128_real(out[k]);
    }
  }

  free(out);
  return SDSGE_RESIDUAL_PATH_OK;
}
