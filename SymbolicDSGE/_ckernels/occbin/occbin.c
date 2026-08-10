#include "occbin.h"
#include "../_common/sdsge_linalg.h"
#include <string.h>

i64 sdsge_constraint_path(sdsge_constraint_fn cond, f64 *SDSGE_RESTRICT path,
                          f64 *SDSGE_RESTRICT par, const i8 *regime_in,
                          i8 *regime_out, i64 inclusive,
                          f64 *SDSGE_RESTRICT max_err, i64 T, i64 n_var,
                          i64 n_constraint) {
  f64 err[4]; // 2 * MAX_CONSTRAINTS (a constraint is a relax/bind pair)
  i64 changed = 0;

  for (i64 t = 0; t < T; ++t) {
    cond(&path[t * n_var], par, err);
    const i8 prev = regime_in[t];
    i8 next = 0;
    for (i64 i = 0; i < n_constraint; ++i) {
      // If the previous regime was binding, check if it should relax; if it was
      // relaxing, check if it should bind
      const i64 slot = ((prev >> i) & 1) ? 2 * i + 1 : 2 * i;
      const f64 e = err[slot];
      const int fired = (e > 0.0) || (e == 0.0 && ((inclusive >> slot) & 1));
      next |= (((prev >> i) & 1) ? !fired : fired) << i;
      if (fired && e > *max_err) {
        *max_err = e;
      }
    }
    regime_out[t] = next;
    changed += (next != prev);
  }
  return changed; // No status code, call can't fail
}

arena_size sdsge_occbin_recursion_arena_size(i64 n_var, i64 n_state,
                                             i64 n_ctrl) {
  const i64 n_rhs = n_state + 1;
  return make_sizer(n_var * n_var + n_var * n_rhs + n_ctrl * n_rhs, n_var);
}

i64 sdsge_occbin_recursion(const occbin_ctx *ctx, const i8 *SDSGE_RESTRICT mask,
                           i64 T, f64 *SDSGE_RESTRICT out,
                           i64 *SDSGE_RESTRICT singular_date,
                           f64 *SDSGE_RESTRICT arena,
                           i64 *SDSGE_RESTRICT iarena) {
  const i64 n_var = ctx->n_var;
  const i64 n_state = ctx->n_state;
  const i64 n_ctrl = ctx->n_ctrl;
  const i64 n_rhs = n_state + 1;

  f64 *m = arena;
  f64 *rhs = m + n_var * n_var;
  f64 *seed = rhs + n_var * n_rhs;
  i64 *piv = iarena;

  *singular_date = -1;

  /* Date T closes on the reference rule, restrided to match a solved block. */
  for (i64 l = 0; l < n_ctrl; ++l) {
    for (i64 j = 0; j < n_state; ++j) {
      seed[l * n_rhs + j] = ctx->f_ref[l * n_state + j];
    }
    seed[l * n_rhs + n_state] = 0.0;
  }
  const f64 *u_next = seed;

  for (i64 t = T - 1; t >= 0; --t) {
    const regime_ctx *reg = &ctx->table[mask[t]];

    for (i64 i = 0; i < n_var; ++i) {
      const f64 *a_row = reg->a + i * n_var;
      const f64 *b_row = reg->b + i * n_var;
      f64 *m_row = m + i * n_var;
      f64 *r_row = rhs + i * n_rhs;
      f64 rc = reg->c[i];

      for (i64 j = 0; j < n_state; ++j) {
        m_row[j] = a_row[j];
        r_row[j] = b_row[j];
      }
      for (i64 j = 0; j < n_ctrl; ++j) {
        m_row[n_state + j] = -b_row[n_state + j];
      }

      for (i64 l = 0; l < n_ctrl; ++l) {
        const f64 a_ul = a_row[n_state + l];
        const f64 *u_row = u_next + l * n_rhs;
        for (i64 j = 0; j < n_state; ++j) {
          m_row[j] += a_ul * u_row[j];
        }
        rc += a_ul * u_row[n_state];
      }
      r_row[n_state] = -rc;
    }

    if (sdsge_lu_factor_inplace(m, piv, n_var) != SDSGE_LU_SUCCESS) {
      *singular_date = t;
      return SDSGE_OCCBIN_RECURSION_SINGULAR;
    }

    f64 *rule = out + t * n_var * n_rhs;
    sdsge_lu_solve(m, piv, rhs, rule, n_var, n_rhs);
    u_next = rule + n_state * n_rhs;
  }
  return SDSGE_OCCBIN_RECURSION_OK;
}

void sdsge_occbin_forward(const f64 *SDSGE_RESTRICT rule,
                          const f64 *SDSGE_RESTRICT x0, i64 T, i64 n_var,
                          i64 n_state, f64 *SDSGE_RESTRICT path) {
  const i64 n_rhs = n_state + 1;

  for (i64 j = 0; j < n_state; ++j) {
    path[j] = x0[j];
  }

  for (i64 t = 0; t < T; ++t) {
    const f64 *r = rule + t * n_var * n_rhs;
    const f64 *x = path + t * n_var;
    f64 *u = path + t * n_var + n_state;

    for (i64 i = n_state; i < n_var; ++i) { // n_ctrl
      f64 acc = r[i * n_rhs + n_state];
      for (i64 j = 0; j < n_state; ++j) {
        acc += r[i * n_rhs + j] * x[j];
      }
      u[i - n_state] = acc;
    }
    if (t == T - 1) {
      break;
    }

    f64 *xn = path + (t + 1) * n_var;
    for (i64 i = 0; i < n_state; ++i) {
      f64 acc = r[i * n_rhs + n_state];
      for (i64 j = 0; j < n_state; ++j) {
        acc += r[i * n_rhs + j] * x[j];
      }
      xn[i] = acc;
    }
  }
}

arena_size sdsge_occbin_period_arena_size(i64 n_var, i64 n_state, i64 n_ctrl,
                                          i64 T_cap, i64 max_iter) {
  const arena_size rec =
      sdsge_occbin_recursion_arena_size(n_var, n_state, n_ctrl);
  const i64 bytes = (max_iter + 1) * T_cap; // hist and next
  return make_sizer(T_cap * n_var + rec.n_float,
                    rec.n_int + max_iter + (bytes + 7) / 8);
}

i64 sdsge_occbin_period(const occbin_run_ctx *run,
                        const f64 *SDSGE_RESTRICT x0, i8 *mask, i64 *T,
                        f64 *rule, f64 *path, occbin_diag *diag, i64 s,
                        f64 *arena, i64 *iarena) {
  const i64 n_var = run->model->n_var;
  const i64 n_state = run->model->n_state;

  f64 *lev = arena;                    // (T_cap, n_var) levels
  f64 *rec = lev + run->T_cap * n_var; // recursion scratch
  i64 *hist_len = iarena + n_var;      // (max_iter,)
  i8 *next = (i8 *)(hist_len + run->max_iter); // (T_cap,)
  i8 *hist = next + run->T_cap;                // (max_iter, T_cap)

  f64 *max_err = &diag->max_err[s];
  i64 changed = 1;
  i64 iter = 0;

  while (changed && iter < run->max_iter) {
    if (mask[*T - 1] != 0 && *T < run->T_cap) {
      // last period is binding, append a relaxing regime at the end.
      mask[*T] = 0;
      ++(*T);
    } else if (mask[*T - 1] != 0) {
      // binding at the cap, clip the last period to a relaxing regime.
      mask[*T - 1] = 0;
    }

    memcpy(hist + iter * run->T_cap, mask, *T);
    hist_len[iter] = *T;
    ++iter;

    i64 status = sdsge_occbin_recursion(run->model, mask, *T, rule,
                                        &diag->singular_date, rec, iarena);
    if (status != SDSGE_OCCBIN_RECURSION_OK) {
      diag->iters[s] = iter;
      return status;
    }

    sdsge_occbin_forward(rule, x0, *T, n_var, n_state, path);

    for (i64 t = 0; t < *T; ++t) {
      for (i64 i = 0; i < n_var; ++i) {
        lev[t * n_var + i] = path[t * n_var + i] + run->ss[i];
      }
    }

    *max_err = 0.0;
    changed = sdsge_constraint_path(run->cond, lev, run->par, mask, next,
                                    run->inclusive, max_err, *T, n_var,
                                    run->n_constraint);
    memcpy(mask, next, *T);

    // A guess seen before is a cycle; the newest match dates its length. The
    // guess entering this iteration cannot match, `changed` says it moved.
    if (changed) {
      for (i64 k = iter - 2; k >= 0; --k) {
        if (hist_len[k] == *T && memcmp(hist + k * run->T_cap, mask, *T) == 0) {
          diag->iters[s] = iter;
          return (k == iter - 2) ? SDSGE_OCCBIN_PERIODIC
                                 : SDSGE_OCCBIN_PERIODIC_LOOP;
        }
      }
    }
  }
  diag->iters[s] = iter;
  return changed ? SDSGE_OCCBIN_MAXITER : SDSGE_OCCBIN_PERIOD_OK;
}

/* Extend a relaxed path by one date: x_T = p x_{T-1}, then u_T = f x_T. Kept
   as two loops so the control rows read `next` through `next` itself. */
static void sdsge_occbin_reference_step(const f64 *SDSGE_RESTRICT ref,
                                        const f64 *SDSGE_RESTRICT prev,
                                        i64 n_var, i64 n_state,
                                        f64 *SDSGE_RESTRICT next) {
  const i64 n_rhs = n_state + 1;

  for (i64 i = 0; i < n_state; ++i) {
    f64 acc = ref[i * n_rhs + n_state];
    for (i64 j = 0; j < n_state; ++j) {
      acc += ref[i * n_rhs + j] * prev[j];
    }
    next[i] = acc;
  }
  for (i64 i = n_state; i < n_var; ++i) {
    f64 acc = ref[i * n_rhs + n_state];
    for (i64 j = 0; j < n_state; ++j) {
      acc += ref[i * n_rhs + j] * next[j];
    }
    next[i] = acc;
  }
}

arena_size sdsge_occbin_solve_arena_size(i64 n_var, i64 n_state, i64 n_ctrl,
                                         i64 T_cap, i64 max_iter) {
  const arena_size per = sdsge_occbin_period_arena_size(n_var, n_state, n_ctrl,
                                                        T_cap, max_iter);
  return make_sizer(n_state + per.n_float, per.n_int);
}

i64 sdsge_occbin_solve(const occbin_run_ctx *run, const f64 *shocks, i64 S,
                       i64 n_periods, const f64 *x_init, f64 *out, i8 *regimes,
                       i64 *T_used, occbin_diag *diag, f64 *rule, f64 *path,
                       i8 *mask, f64 *arena, i64 *iarena) {
  const i64 n_var = run->model->n_var;
  const i64 n_state = run->model->n_state;
  const i64 n_rhs = n_state + 1;
  i64 T = run->T0;

  f64 *x = arena; // (n_state,) state entering the current shock period
  f64 *period_arena = x + n_state;

  diag->fail_period = -1;
  diag->singular_date = -1;

  memset(mask, 0, T);
  memset(path, 0, T * n_var * sizeof(f64));
  memcpy(x, x_init, n_state * sizeof(f64));

  for (i64 s = 0; s < S; ++s) {
    for (i64 j = 0; j < n_state; ++j) {
      x[j] += shocks[s * n_state + j];
    }

    if (s == 0 || sdsge_any_nonzero(shocks + s * n_state, n_state)) {
      i64 status = sdsge_occbin_period(run, x, mask, &T, rule, path, diag, s,
                                       period_arena, iarena);
      if (status != SDSGE_OCCBIN_PERIOD_OK) {
        diag->fail_period = s;
        return status;
      }
    } else {
      // No shock: the previous guess and the path it implies, both already
      // shifted onto this period, still describe it.
      diag->iters[s] = 0;
      diag->max_err[s] = 0.0;
    }

    memcpy(out + s * n_var, path, n_var * sizeof(f64));
    memcpy(regimes + s * run->T_cap, mask, T);
    T_used[s] = T;

    // Advance a period: guess, path and carry shift together. A converged
    // guess is relaxed at its last date, so that block is the reference rule.
    memmove(path, path + n_var, (T - 1) * n_var * sizeof(f64));
    sdsge_occbin_reference_step(rule + (T - 1) * n_var * n_rhs,
                                path + (T - 2) * n_var, n_var, n_state,
                                path + (T - 1) * n_var);
    memmove(mask, mask + 1, T - 1);
    mask[T - 1] = 0; // last period is relaxing by default
    memcpy(x, path, n_state * sizeof(f64));
  }

  // The tail is what the last solved path still had left to run.
  for (i64 t = S; t < n_periods; ++t) {
    memcpy(out + t * n_var, path + (t - S) * n_var, n_var * sizeof(f64));
  }
  return SDSGE_OCCBIN_SOLVE_OK;
}
