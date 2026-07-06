#include "core.h"
#include <stdlib.h>

void sdsge_simulate_linear_states(const f64 *SDSGE_RESTRICT A,
                                  const f64 *SDSGE_RESTRICT B,
                                  const f64 *SDSGE_RESTRICT x0,
                                  const f64 *SDSGE_RESTRICT shock,
                                  f64 *SDSGE_RESTRICT out, i64 T, i64 n,
                                  i64 k) {
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
                               const f64 *SDSGE_RESTRICT d, i64 state_start,
                               f64 *SDSGE_RESTRICT out, i64 T, i64 m, i64 n) {
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

i64 sdsge_simulate_second_order_pruned(
    const f64 *SDSGE_RESTRICT hx, const f64 *SDSGE_RESTRICT gx,
    const f64 *SDSGE_RESTRICT bx, const f64 *SDSGE_RESTRICT hxx,
    const f64 *SDSGE_RESTRICT gxx, const f64 *SDSGE_RESTRICT hss,
    const f64 *SDSGE_RESTRICT gss, const f64 *SDSGE_RESTRICT x0,
    const f64 *SDSGE_RESTRICT shock, const i64 T, const i64 nx, const i64 ny,
    const i64 n_exog, f64 *SDSGE_RESTRICT x_out, f64 *SDSGE_RESTRICT y_out) {

  /* One arena holds x1_cur, x1_next, x2_cur, x2_next, and x1_outer. */
  const size_t arena_count = (size_t)(4 * nx + nx * nx);
  f64 *SDSGE_RESTRICT arena =
      (f64 *)malloc((arena_count > 0 ? arena_count : 1) * sizeof(f64));
  if (!arena) {
    return SDSGE_CORE_ALLOC_FAIL;
  }

  f64 *SDSGE_RESTRICT x1_cur = arena;
  f64 *SDSGE_RESTRICT x1_next = arena + nx;
  f64 *SDSGE_RESTRICT x2_cur = arena + 2 * nx;
  f64 *SDSGE_RESTRICT x2_next = arena + 3 * nx;
  f64 *SDSGE_RESTRICT x1_outer = arena + 4 * nx;

  for (i64 i = 0; i < nx; ++i) {
    x1_cur[i] = x0[i];
    x2_cur[i] = 0.0;
  }

  for (i64 t = 0; t <= T; ++t) {
    f64 *SDSGE_RESTRICT xt = x_out + t * nx;

    for (i64 i = 0; i < nx; ++i) {
      xt[i] = x1_cur[i] + x2_cur[i];
    }

    for (i64 j = 0; j < nx; ++j) {
      const f64 xj = x1_cur[j];
      f64 *SDSGE_RESTRICT row = x1_outer + j * nx;
      for (i64 k = 0; k < nx; ++k) {
        row[k] = xj * x1_cur[k];
      }
    }

    if (ny > 0) {
      f64 *SDSGE_RESTRICT yt = y_out + t * ny;
      for (i64 i = 0; i < ny; ++i) {
        const f64 *SDSGE_RESTRICT gxi = gx + i * nx;
        const f64 *SDSGE_RESTRICT gxxi = gxx + i * nx * nx;
        f64 s = 0.5 * gss[i];

        for (i64 j = 0; j < nx; ++j) {
          s += gxi[j] * xt[j];
        }
        for (i64 j = 0; j < nx; ++j) {
          const f64 *SDSGE_RESTRICT gxxij = gxxi + j * nx;
          const f64 *SDSGE_RESTRICT outerj = x1_outer + j * nx;
          for (i64 k = 0; k < nx; ++k) {
            s += 0.5 * gxxij[k] * outerj[k];
          }
        }
        yt[i] = s;
      }
    }

    if (t == T) {
      break;
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
  }

  free(arena);
  return SDSGE_CORE_SUCCESS;
}
