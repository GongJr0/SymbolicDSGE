#include "occbin.h"

i64 sdsge_constraint_path(sdsge_constraint_fn cond, f64 *SDSGE_RESTRICT path,
                          f64 *SDSGE_RESTRICT par, const i8 *regime_in,
                          i8 *regime_out, i64 T, i64 n_var, i64 n_constraint) {
  i8 flags[4]; // 2 * MAX_CONSTRAINTS (a constraint is a relax/bind pair)
  i64 changed = 0;

  for (i64 t = 0; t < T; ++t) {
    cond(&path[t * n_var], par, flags);
    const i8 prev = regime_in[t];
    i8 next = 0;
    for (i64 i = 0; i < n_constraint; ++i) {
      // If the previous regime was binding, check if it should relax; if it was
      // relaxing, check if it should bind
      next |= (((prev >> i) & 1) ? !flags[2 * i + 1] : flags[2 * i]) << i;
    }
    regime_out[t] = next;
    changed += (next != prev);
  }
  return changed; // No status code, call can't fail
}
