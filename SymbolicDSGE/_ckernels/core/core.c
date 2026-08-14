#include "core.h"

void sdsge_assemble_transition(const f64 *SDSGE_RESTRICT p,
                               const f64 *SDSGE_RESTRICT f, const i64 n_state,
                               const i64 n_control, f64 *SDSGE_RESTRICT A) {
  const i64 n_total = n_state + n_control;

  /*
   * A = [[p, 0],
   *      [f, 0]]
   *
   * A state is a variable occurring at t-1, so the rule it carries already maps
   * y_{t-1} to y_t and every row of A is a row of that rule. There is no
   * product here: a control does not respond to the state at t, it responds to
   * the same lagged state the transition does. The control columns are empty
   * because nothing reads a control one period on.
   */
  for (i64 i = 0; i < n_total; ++i) {
    for (i64 j = 0; j < n_total; ++j) {
      A[i * n_total + j] = 0.0;
    }
  }
  for (i64 i = 0; i < n_state; ++i) {
    for (i64 j = 0; j < n_state; ++j) {
      A[i * n_total + j] = p[i * n_state + j];
    }
  }
  for (i64 i = 0; i < n_control; ++i) {
    for (i64 j = 0; j < n_state; ++j) {
      A[(n_state + i) * n_total + j] = f[i * n_state + j];
    }
  }
}

arena_size sdsge_simulate_linear_states_arena_size(const i64 n) {
  return make_sizer(2 * n, 0);
}

void sdsge_simulate_linear_states(const f64 *SDSGE_RESTRICT A,
                                  const f64 *SDSGE_RESTRICT B,
                                  const f64 *SDSGE_RESTRICT x0,
                                  const f64 *SDSGE_RESTRICT shock,
                                  const f64 *SDSGE_RESTRICT ss,
                                  f64 *SDSGE_RESTRICT out,
                                  f64 *SDSGE_RESTRICT arena, const i64 T,
                                  const i64 n, const i64 k) {
  f64 *SDSGE_RESTRICT cur = arena;
  f64 *SDSGE_RESTRICT next = arena + n;

  for (i64 i = 0; i < n; ++i) {
    cur[i] = x0[i];
  }

  for (i64 t = 0; t < T; ++t) {
    const f64 *SDSGE_RESTRICT st = shock + t * k;
    f64 *SDSGE_RESTRICT row = out + t * n;
    for (i64 i = 0; i < n; ++i) {
      const f64 *SDSGE_RESTRICT Ai = A + i * n;
      const f64 *SDSGE_RESTRICT Bi = B + i * k;
      f64 s = 0.0;
      for (i64 j = 0; j < n; ++j)
        s += Ai[j] * cur[j];
      for (i64 j = 0; j < k; ++j)
        s += Bi[j] * st[j];
      next[i] = s;
      row[i] = ss != NULL ? s + ss[i] : s;
    }
    f64 *tmp = cur;
    cur = next;
    next = tmp;
  }
}

void sdsge_affine_observations(const f64 *SDSGE_RESTRICT states,
                               const f64 *SDSGE_RESTRICT C,
                               const f64 *SDSGE_RESTRICT d,
                               f64 *SDSGE_RESTRICT out, i64 T, i64 m, i64 n) {
  for (i64 t = 0; t < T; ++t) {
    const f64 *row = states + t * n;
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

arena_size sdsge_simulate_second_order_pruned_arena_size(const i64 nx,
                                                        const i64 n_exog) {
  return make_sizer(4 * nx + nx * nx + nx * n_exog + n_exog * n_exog, 0);
}

void sdsge_simulate_second_order_pruned(
    const f64 *SDSGE_RESTRICT hx, const f64 *SDSGE_RESTRICT gx,
    const f64 *SDSGE_RESTRICT bu, const f64 *SDSGE_RESTRICT hxx,
    const f64 *SDSGE_RESTRICT gxx, const f64 *SDSGE_RESTRICT hxu,
    const f64 *SDSGE_RESTRICT gxu, const f64 *SDSGE_RESTRICT huu,
    const f64 *SDSGE_RESTRICT guu, const f64 *SDSGE_RESTRICT hss,
    const f64 *SDSGE_RESTRICT gss, const f64 *SDSGE_RESTRICT x0,
    const f64 *SDSGE_RESTRICT shock, const f64 *SDSGE_RESTRICT ss,
    f64 *SDSGE_RESTRICT out, f64 *SDSGE_RESTRICT arena, const i64 T,
    const i64 nx, const i64 ny, const i64 n_exog) {

  f64 *SDSGE_RESTRICT x1_cur = arena;
  f64 *SDSGE_RESTRICT x1_next = arena + nx;
  f64 *SDSGE_RESTRICT x2_cur = arena + 2 * nx;
  f64 *SDSGE_RESTRICT x2_next = arena + 3 * nx;
  f64 *SDSGE_RESTRICT xx = arena + 4 * nx;
  f64 *SDSGE_RESTRICT xu = xx + nx * nx;
  f64 *SDSGE_RESTRICT uu = xu + nx * n_exog;

  for (i64 i = 0; i < nx; ++i) {
    x1_cur[i] = x0[i];
    x2_cur[i] = 0.0;
  }

  for (i64 t = 0; t < T; ++t) {
    const f64 *SDSGE_RESTRICT u = n_exog > 0 ? shock + t * n_exog : NULL;

    for (i64 j = 0; j < nx; ++j) {
      const f64 xj = x1_cur[j];
      for (i64 k = 0; k < nx; ++k) {
        xx[j * nx + k] = xj * x1_cur[k];
      }
      for (i64 l = 0; l < n_exog; ++l) {
        xu[j * n_exog + l] = xj * u[l];
      }
    }
    for (i64 l = 0; l < n_exog; ++l) {
      for (i64 m = 0; m < n_exog; ++m) {
        uu[l * n_exog + m] = u[l] * u[m];
      }
    }

    /* States. The first-order path is what the quadratic terms read, so it is
     * carried alongside the full one rather than recovered from it. */
    for (i64 i = 0; i < nx; ++i) {
      f64 s1 = 0.0;
      f64 s2 = 0.5 * hss[i];
      for (i64 j = 0; j < nx; ++j) {
        const f64 hxij = hx[i * nx + j];
        s1 += hxij * x1_cur[j];
        s2 += hxij * x2_cur[j];
      }
      for (i64 l = 0; l < n_exog; ++l) {
        s1 += bu[i * n_exog + l] * u[l];
      }
      const f64 *SDSGE_RESTRICT hxxi = hxx + i * nx * nx;
      for (i64 j = 0; j < nx * nx; ++j) {
        s2 += 0.5 * hxxi[j] * xx[j];
      }
      const f64 *SDSGE_RESTRICT hxui = hxu + i * nx * n_exog;
      for (i64 j = 0; j < nx * n_exog; ++j) {
        s2 += hxui[j] * xu[j];
      }
      const f64 *SDSGE_RESTRICT huui = huu + i * n_exog * n_exog;
      for (i64 j = 0; j < n_exog * n_exog; ++j) {
        s2 += 0.5 * huui[j] * uu[j];
      }
      x1_next[i] = s1;
      x2_next[i] = s2;
    }

    /* Controls, off the same previous state and the same innovation. Their
     * first-order shock response is the control rows of bu, which no product of
     * gx with a state loading can stand in for. */
    f64 *SDSGE_RESTRICT row = out + t * (nx + ny);
    for (i64 i = 0; i < ny; ++i) {
      const f64 *SDSGE_RESTRICT gxi = gx + i * nx;
      f64 s = 0.5 * gss[i];
      for (i64 j = 0; j < nx; ++j) {
        s += gxi[j] * (x1_cur[j] + x2_cur[j]);
      }
      for (i64 l = 0; l < n_exog; ++l) {
        s += bu[(nx + i) * n_exog + l] * u[l];
      }
      const f64 *SDSGE_RESTRICT gxxi = gxx + i * nx * nx;
      for (i64 j = 0; j < nx * nx; ++j) {
        s += 0.5 * gxxi[j] * xx[j];
      }
      const f64 *SDSGE_RESTRICT gxui = gxu + i * nx * n_exog;
      for (i64 j = 0; j < nx * n_exog; ++j) {
        s += gxui[j] * xu[j];
      }
      const f64 *SDSGE_RESTRICT guui = guu + i * n_exog * n_exog;
      for (i64 j = 0; j < n_exog * n_exog; ++j) {
        s += 0.5 * guui[j] * uu[j];
      }
      row[nx + i] = ss != NULL ? s + ss[nx + i] : s;
    }

    f64 *tmp = x1_cur;
    x1_cur = x1_next;
    x1_next = tmp;

    tmp = x2_cur;
    x2_cur = x2_next;
    x2_next = tmp;

    for (i64 i = 0; i < nx; ++i) {
      const f64 s = x1_cur[i] + x2_cur[i];
      row[i] = ss != NULL ? s + ss[i] : s;
    }
  }
}
