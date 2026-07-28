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

void sdsge_simulate_order2_step(
    const f64 *SDSGE_RESTRICT hx, const f64 *SDSGE_RESTRICT gx,
    const f64 *SDSGE_RESTRICT bx, const f64 *SDSGE_RESTRICT hxx,
    const f64 *SDSGE_RESTRICT gxx, const f64 *SDSGE_RESTRICT hss,
    const f64 *SDSGE_RESTRICT gss, const f64 *SDSGE_RESTRICT steady_state,
    const f64 *SDSGE_RESTRICT x0, const f64 *SDSGE_RESTRICT shock,
    sdsge_measurement_fn measurement, f64 *SDSGE_RESTRICT params, i64 T, i64 nx,
    i64 ny, i64 n_exog, i64 m, f64 *SDSGE_RESTRICT simout,
    f64 *SDSGE_RESTRICT scratch) {
  const i64 n = nx + ny;
  f64 *SDSGE_RESTRICT states = simout;
  f64 *SDSGE_RESTRICT observables = simout + T * n;
  f64 *SDSGE_RESTRICT x1_cur = scratch;
  f64 *SDSGE_RESTRICT x1_next = scratch + nx;
  f64 *SDSGE_RESTRICT x2_cur = scratch + 2 * nx;
  f64 *SDSGE_RESTRICT x2_next = scratch + 3 * nx;
  f64 *SDSGE_RESTRICT x1_outer = scratch + 4 * nx;

  for (i64 i = 0; i < nx; ++i) {
    x1_cur[i] = x0[i];
    x2_cur[i] = 0.0;
  }

  for (i64 t = 0; t < T; ++t) {
    for (i64 j = 0; j < nx; ++j) {
      const f64 xj = x1_cur[j];
      f64 *SDSGE_RESTRICT row = x1_outer + j * nx;
      for (i64 k = 0; k < nx; ++k) {
        row[k] = xj * x1_cur[k];
      }
    }

    const f64 *SDSGE_RESTRICT shock_t = n_exog > 0 ? shock + t * n_exog : NULL;
    for (i64 i = 0; i < nx; ++i) {
      const f64 *SDSGE_RESTRICT hxi = hx + i * nx;
      const f64 *SDSGE_RESTRICT bxi = n_exog > 0 ? bx + i * n_exog : NULL;
      const f64 *SDSGE_RESTRICT hxxi = hxx + i * nx * nx;
      f64 s1 = 0.0;
      f64 s2 = 0.5 * hss[i];

      for (i64 j = 0; j < nx; ++j) {
        s1 += hxi[j] * x1_cur[j];
        s2 += hxi[j] * x2_cur[j];
      }
      for (i64 j = 0; j < n_exog; ++j) {
        s1 += bxi[j] * shock_t[j];
      }
      for (i64 j = 0; j < nx; ++j) {
        const f64 *SDSGE_RESTRICT hxxij = hxxi + j * nx;
        const f64 *SDSGE_RESTRICT outerj = x1_outer + j * nx;
        for (i64 k = 0; k < nx; ++k) {
          s2 += 0.5 * hxxij[k] * outerj[k];
        }
      }

      x1_next[i] = s1;
      x2_next[i] = s2;
    }

    f64 *tmp = x1_cur;
    x1_cur = x1_next;
    x1_next = tmp;
    tmp = x2_cur;
    x2_cur = x2_next;
    x2_next = tmp;

    f64 *SDSGE_RESTRICT state_t = states + t * n;
    for (i64 i = 0; i < nx; ++i) {
      state_t[i] = x1_cur[i] + x2_cur[i];
    }

    if (ny > 0) {
      for (i64 j = 0; j < nx; ++j) {
        const f64 xj = x1_cur[j];
        f64 *SDSGE_RESTRICT row = x1_outer + j * nx;
        for (i64 k = 0; k < nx; ++k) {
          row[k] = xj * x1_cur[k];
        }
      }

      for (i64 i = 0; i < ny; ++i) {
        const f64 *SDSGE_RESTRICT gxi = gx + i * nx;
        const f64 *SDSGE_RESTRICT gxxi = gxx + i * nx * nx;
        f64 value = 0.5 * gss[i];

        for (i64 j = 0; j < nx; ++j) {
          value += gxi[j] * state_t[j];
        }
        for (i64 j = 0; j < nx; ++j) {
          const f64 *SDSGE_RESTRICT gxxij = gxxi + j * nx;
          const f64 *SDSGE_RESTRICT outerj = x1_outer + j * nx;
          for (i64 k = 0; k < nx; ++k) {
            value += 0.5 * gxxij[k] * outerj[k];
          }
        }
        state_t[nx + i] = value;
      }
    }

    for (i64 i = 0; i < n; ++i) {
      state_t[i] += steady_state[i];
    }
    if (m > 0) {
      measurement(state_t, params, observables + t * m);
    }
  }
}
