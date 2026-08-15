#ifndef SDSGE_PERTURBATION_H
#define SDSGE_PERTURBATION_H

#include "sdsge_common.h"
#include <stddef.h>

/* One period of the pruned second-order law of motion, as Dynare's
 * local_state_space_iteration_2 writes it: from the previous pruned state and
 * this period's innovation to the whole variable vector and the next pruned
 * state.
 *
 * The simulator sweeps it over time and the unscented filter sweeps it over
 * sigma points. They differ in what they sweep, not in the recursion, so it
 * lives here rather than in either. It is header-only and `static inline`
 * deliberately: a shared definition in `core` would make every filter link
 * `core`, and a copy in each would be free to drift. Neither happens if there
 * is one source and no symbol.
 *
 * `ss` denominates `vars`; NULL leaves it a deviation. The pruned state is
 * always a deviation. `arena` is sdsge_second_order_step_scratch wide.
 *
 * `ghxu` carries no 1/2: the cross pair is counted once. The quadratic terms
 * read the first-order path alone, which is the pruning. */

static inline i64 sdsge_second_order_step_scratch(const i64 nx,
                                                  const i64 n_exog) {
  return nx * nx + nx * n_exog + n_exog * n_exog;
}

static inline void sdsge_second_order_step(
    const f64 *SDSGE_RESTRICT hx, const f64 *SDSGE_RESTRICT gx,
    const f64 *SDSGE_RESTRICT bu, const f64 *SDSGE_RESTRICT hxx,
    const f64 *SDSGE_RESTRICT gxx, const f64 *SDSGE_RESTRICT hxu,
    const f64 *SDSGE_RESTRICT gxu, const f64 *SDSGE_RESTRICT huu,
    const f64 *SDSGE_RESTRICT guu, const f64 *SDSGE_RESTRICT hss,
    const f64 *SDSGE_RESTRICT gss, const f64 *SDSGE_RESTRICT x1,
    const f64 *SDSGE_RESTRICT x2, const f64 *SDSGE_RESTRICT u,
    const f64 *SDSGE_RESTRICT ss, f64 *SDSGE_RESTRICT x1_next,
    f64 *SDSGE_RESTRICT x2_next, f64 *SDSGE_RESTRICT vars,
    f64 *SDSGE_RESTRICT arena, const i64 nx, const i64 ny, const i64 n_exog) {

  f64 *SDSGE_RESTRICT xx = arena;
  f64 *SDSGE_RESTRICT xu = xx + nx * nx;
  f64 *SDSGE_RESTRICT uu = xu + nx * n_exog;

  for (i64 j = 0; j < nx; ++j) {
    const f64 xj = x1[j];
    for (i64 k = 0; k < nx; ++k) {
      xx[j * nx + k] = xj * x1[k];
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
      s1 += hxij * x1[j];
      s2 += hxij * x2[j];
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
    vars[i] = ss != NULL ? s1 + s2 + ss[i] : s1 + s2;
  }

  /* Controls, off the same previous state and the same innovation. Their
   * first-order shock response is the control rows of bu, which no product of
   * gx with a state loading can stand in for. */
  for (i64 i = 0; i < ny; ++i) {
    const f64 *SDSGE_RESTRICT gxi = gx + i * nx;
    f64 s = 0.5 * gss[i];
    for (i64 j = 0; j < nx; ++j) {
      s += gxi[j] * (x1[j] + x2[j]);
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
    vars[nx + i] = ss != NULL ? s + ss[nx + i] : s;
  }
}

#endif /* SDSGE_PERTURBATION_H */
