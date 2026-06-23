#include "core.h"

void sdsge_simulate_linear_states(const f64 *SDSGE_RESTRICT A,
                                  const f64 *SDSGE_RESTRICT B,
                                  const f64 *SDSGE_RESTRICT x0,
                                  const f64 *SDSGE_RESTRICT shock,
                                  f64 *SDSGE_RESTRICT out,
                                  i64 T, i64 n, i64 k)
{
    for (i64 i = 0; i < n; ++i)
        out[i] = x0[i];

    for (i64 t = 0; t < T; ++t) {
        const f64 *xt = out + t * n;
        const f64 *st = shock + t * k;
        f64 *xn = out + (t + 1) * n;
        for (i64 i = 0; i < n; ++i) {
            const f64 *Ai = A + i * n;
            const f64 *Bi = B + i * k;
            f64 s = 0.0;
            for (i64 j = 0; j < n; ++j)
                s += Ai[j] * xt[j];
            for (i64 j = 0; j < k; ++j)
                s += Bi[j] * st[j];
            xn[i] = s;
        }
    }
}

void sdsge_affine_observations(const f64 *SDSGE_RESTRICT states,
                               const f64 *SDSGE_RESTRICT C,
                               const f64 *SDSGE_RESTRICT d,
                               i64 state_start,
                               f64 *SDSGE_RESTRICT out,
                               i64 T, i64 m, i64 n)
{
    for (i64 t = 0; t < T; ++t) {
        const f64 *row = states + (state_start + t) * n;
        f64 *ot = out + t * m;
        for (i64 i = 0; i < m; ++i) {
            const f64 *Ci = C + i * n;
            f64 s = d[i];
            for (i64 j = 0; j < n; ++j)
                s += Ci[j] * row[j];
            ot[i] = s;
        }
    }
}
