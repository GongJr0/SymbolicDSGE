#include "interface_helpers.h"
#include <math.h>

void sdsge_cov_from_unconstrained(const f64 *SDSGE_RESTRICT z,
                                  const f64 *SDSGE_RESTRICT std, const i64 K,
                                  f64 *SDSGE_RESTRICT L,
                                  f64 *SDSGE_RESTRICT out) {
  i64 idx = 0;
  for (i64 i = 0; i < K; ++i) {
    const i64 ri = i * K;
    const f64 si = std[i];

    f64 rem = 1.0;
    for (i64 j = 0; j < i; ++j) {
      const f64 v = sqrt(max_f64(1e-14, rem)) * tanh(z[idx++]);
      L[ri + j] = v;
      rem -= v * v;
    }
    L[ri + i] = sqrt(max_f64(1e-14, rem));

    for (i64 j = 0; j < i; ++j) {
      const i64 rj = j * K;
      f64 s = 0.0;
      for (i64 c = 0; c <= j; ++c)
        s += L[ri + c] * L[rj + c];
      const f64 v = si * std[j] * s;
      out[ri + j] = v;
      out[rj + i] = v;
    }
    out[ri + i] = si * si;
  }
}
