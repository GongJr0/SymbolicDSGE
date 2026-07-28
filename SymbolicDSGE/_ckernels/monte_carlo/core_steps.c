#include "core_steps.h"
#include "../core/core.h"

void sdsge_simulate_order1_step(
    const f64 *SDSGE_RESTRICT A, const f64 *SDSGE_RESTRICT B,
    const f64 *SDSGE_RESTRICT C, const f64 *SDSGE_RESTRICT d,
    const f64 *SDSGE_RESTRICT x0, const f64 *SDSGE_RESTRICT shock, const i64 T,
    const i64 n, const i64 k, const i64 m, f64 *SDSGE_RESTRICT simout) {
  f64 *states = simout;
  f64 *observables = simout + T * n;

  for (i64 t = 0; t < T; ++t) {
    const f64 *xt = t == 0 ? x0 : states + (t - 1) * n;
    const f64 *shock_t = shock + t * k;
    f64 *state_t = states + t * n;
    f64 *observable_t = observables + t * m;

    for (i64 i = 0; i < n; ++i) {
      const f64 *Ai = A + i * n;
      const f64 *Bi = B + i * k;
      f64 value = 0.0;
      for (i64 j = 0; j < n; ++j)
        value += Ai[j] * xt[j];
      for (i64 j = 0; j < k; ++j)
        value += Bi[j] * shock_t[j];
      state_t[i] = value;
    }

    for (i64 i = 0; i < m; ++i) {
      const f64 *Ci = C + i * n;
      f64 value = d[i];
      for (i64 j = 0; j < n; ++j)
        value += Ci[j] * state_t[j];
      observable_t[i] = value;
    }
  }
}
