#include "core.h"
#include "../_common/sdsge_linalg.h" /* sdsge_lu_factor_inplace, sdsge_lu_solve */
#include <math.h>                    /* exp */
#include <stdlib.h>                  /* malloc, free */

arena_size sdsge_assemble_arena_size(const i64 n_state, const i64 n_control,
                                     const i64 n_exog) {
  if (n_exog <= 0) {
    return make_sizer(0, 0);
  }
  return make_sizer(2 * (n_state + n_control) + 3 * n_exog * n_exog, n_exog);
}

i64 sdsge_assemble_state_space_into(
    const c128 *SDSGE_RESTRICT p, const c128 *SDSGE_RESTRICT f,
    const sdsge_shock_ctx *shock, const i64 n_state, const i64 n_control,
    const i64 n_exog, f64 *SDSGE_RESTRICT arena, i64 *SDSGE_RESTRICT pivot,
    f64 *SDSGE_RESTRICT A, f64 *SDSGE_RESTRICT B) {
  const i64 n_total = n_state + n_control;

  /*
   * A = [[p,   0],
   *      [f@p, 0]]
   *
   * Shape: n_total × n_total
   */
  for (i64 i = 0; i < n_total; ++i) {
    for (i64 j = 0; j < n_total; ++j) {
      A[i * n_total + j] = 0.0;
    }
  }

  /* Top-left block: p. */
  for (i64 i = 0; i < n_state; ++i) {
    for (i64 j = 0; j < n_state; ++j) {
      A[i * n_total + j] = c128_real(p[i * n_state + j]);
    }
  }

  /* Bottom-left block: f @ p. */
  for (i64 i = 0; i < n_control; ++i) {
    for (i64 j = 0; j < n_state; ++j) {
      c128 value = c128_mul(f[i * n_state], p[j]);

      for (i64 k = 1; k < n_state; ++k) {
        value =
            c128_add(value, c128_mul(f[i * n_state + k], p[k * n_state + j]));
      }

      A[(n_state + i) * n_total + j] = c128_real(value);
    }
  }

  /*
   * B_state = [Rex]  Rex is the within-period response to the innovations;
   *           [ 0 ]  the endogenous states below it are predetermined.
   *
   * B = [B_state]
   *     [f @ B_state]
   *
   * Shape: n_total × n_exog
   */
  for (i64 i = 0; i < n_total; ++i) {
    for (i64 j = 0; j < n_exog; ++j) {
      B[i * n_exog + j] = 0.0;
    }
  }

  if (n_exog == 0) {
    return SDSGE_CORE_SUCCESS;
  }

  const size_t n_sq = (size_t)n_exog * (size_t)n_exog;
  f64 *fwd = arena;
  f64 *cur = arena + n_total;
  f64 *rhs = arena + 2 * n_total; /* -d(resid)/d(shock), the shock rows */
  f64 *M = rhs + n_sq;
  f64 *Rex = M + n_sq;

  /* fwd and cur are both restrict, so the evaluation point goes in twice. */
  for (i64 i = 0; i < n_total; ++i) {
    fwd[i] = shock->log_linear ? exp(shock->ss[i]) : shock->ss[i];
    cur[i] = fwd[i];
  }
  shock->fn(fwd, cur, shock->par, rhs);
  for (size_t i = 0; i < n_sq; ++i) {
    rhs[i] = -rhs[i];
  }

  /* At date t the states move by B_state and the controls by f @ B_state, so
   * the shock rows read a[rows, :n_state] + a[rows, n_state:] @ f. B_state is
   * zero past n_exog, so only its leading columns enter and the system is
   * square. */
  for (i64 k = 0; k < n_exog; ++k) {
    const f64 *arow = shock->a + shock->rows[k] * n_total;
    for (i64 j = 0; j < n_exog; ++j) {
      f64 s = arow[j];
      for (i64 c = 0; c < n_control; ++c) {
        s += arow[n_state + c] * c128_real(f[c * n_state + j]);
      }
      M[k * n_exog + j] = s;
    }
  }

  /* M is scratch, so it takes the factorization in place. */
  const i64 lu_err = sdsge_lu_factor_inplace(M, pivot, n_exog);
  if (lu_err != SDSGE_LU_SUCCESS) {
    return lu_err == SDSGE_LU_SINGULAR ? SDSGE_CORE_SINGULAR
                                       : SDSGE_CORE_ALLOC_FAIL;
  }
  sdsge_lu_solve(M, pivot, rhs, Rex, n_exog, n_exog);

  for (i64 i = 0; i < n_exog; ++i) {
    for (i64 j = 0; j < n_exog; ++j) {
      B[i * n_exog + j] = Rex[i * n_exog + j];
    }
  }

  /* Bottom block: f @ B_state == f[:, :n_exog] @ Rex. */
  for (i64 i = 0; i < n_control; ++i) {
    for (i64 j = 0; j < n_exog; ++j) {
      f64 s = 0.0;
      for (i64 k = 0; k < n_exog; ++k) {
        s += c128_real(f[i * n_state + k]) * Rex[k * n_exog + j];
      }
      B[(n_state + i) * n_exog + j] = s;
    }
  }

  return SDSGE_CORE_SUCCESS;
}

i64 sdsge_assemble_state_space(const c128 *SDSGE_RESTRICT p,
                               const c128 *SDSGE_RESTRICT f,
                               const sdsge_shock_ctx *shock, const i64 n_state,
                               const i64 n_control, const i64 n_exog,
                               f64 *SDSGE_RESTRICT A, f64 *SDSGE_RESTRICT B) {
  const arena_size want =
      sdsge_assemble_arena_size(n_state, n_control, n_exog);
  f64 *arena = NULL;
  i64 *pivot = NULL;

  if (want.n_float > 0) {
    arena = (f64 *)malloc((size_t)want.n_float * sizeof(f64));
    pivot = (i64 *)malloc((size_t)want.n_int * sizeof(i64));
    if (!arena || !pivot) {
      free(arena);
      free(pivot);
      return SDSGE_CORE_ALLOC_FAIL;
    }
  }

  const i64 err = sdsge_assemble_state_space_into(
      p, f, shock, n_state, n_control, n_exog, arena, pivot, A, B);
  free(arena);
  free(pivot);
  return err;
}

void sdsge_simulate_linear_states(const f64 *SDSGE_RESTRICT A,
                                  const f64 *SDSGE_RESTRICT B,
                                  const f64 *SDSGE_RESTRICT x0,
                                  const f64 *SDSGE_RESTRICT shock,
                                  f64 *SDSGE_RESTRICT out, i64 T, i64 n,
                                  i64 k) {
  for (i64 t = 0; t < T; ++t) {
    const f64 *xt = t == 0 ? x0 : out + (t - 1) * n;
    const f64 *st = shock + t * k;
    f64 *xn = out + t * n;
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

    f64 *SDSGE_RESTRICT xt = x_out + t * nx;
    for (i64 i = 0; i < nx; ++i) {
      xt[i] = x1_cur[i] + x2_cur[i];
    }

    if (ny > 0) {
      for (i64 j = 0; j < nx; ++j) {
        const f64 xj = x1_cur[j];
        f64 *SDSGE_RESTRICT row = x1_outer + j * nx;
        for (i64 k = 0; k < nx; ++k) {
          row[k] = xj * x1_cur[k];
        }
      }

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
  }

  free(arena);
  return SDSGE_CORE_SUCCESS;
}
