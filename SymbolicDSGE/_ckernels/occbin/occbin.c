#include "occbin.h"
#include "../_common/sdsge_linalg.h"
#include "../core/klein_solve.h"
#include <stddef.h> // NULL
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
    const f64 *reg_a = ctx->a + mask[t] * n_var * n_var;
    const f64 *reg_b = ctx->b + mask[t] * n_var * n_var;
    const f64 *reg_c = ctx->c + mask[t] * n_var;

    for (i64 i = 0; i < n_var; ++i) {
      const f64 *a_row = reg_a + i * n_var;
      const f64 *b_row = reg_b + i * n_var;
      f64 *m_row = m + i * n_var;
      f64 *r_row = rhs + i * n_rhs;
      f64 rc = reg_c[i];

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

  if (T < 1) {
    return; // the seed below already needs a row to write into
  }
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

/* Largest signed move any one constraint makes between two guesses: dates that
   left the binding regime less dates that entered it. Dynare ranks a guess by
   this, per constraint, and weighs the worst one. */
static i64 sdsge_occbin_regime_move(const i8 *prev, const i8 *next, i64 T,
                                    i64 n_constraint) {
  i64 worst = 0;

  for (i64 i = 0; i < n_constraint; ++i) {
    i64 move = 0;
    for (i64 t = 0; t < T; ++t) {
      move += ((prev[t] >> i) & 1) - ((next[t] >> i) & 1);
    }
    if (i == 0 || move > worst) {
      worst = move;
    }
  }
  return worst;
}

/* Whether any date wants to enter the binding regime, on any constraint. */
static i64 sdsge_occbin_binds_more(const i8 *prev, const i8 *next, i64 T) {
  for (i64 t = 0; t < T; ++t) {
    if (~prev[t] & next[t]) {
      return 1;
    }
  }
  return 0;
}

/* Gauss-Seidel damping: adopt every date the latch wants binding, but let only
   the last date that wants to relax actually relax, and only when no date the
   guess had binding outlasts the relax condition. `stay` carries the relax
   condition at every date, with a set bit meaning the date stays binding. */
static void sdsge_occbin_retrench(i8 *mask, const i8 *next, const i8 *stay,
                                  i64 T, i64 n_constraint) {
  for (i64 i = 0; i < n_constraint; ++i) {
    const i8 bit = (i8)(1 << i);
    i64 last_relax = -1;
    i64 last_bind = -1;
    i64 target = -1;

    for (i64 t = 0; t < T; ++t) {
      const int relaxes = !((stay[t] >> i) & 1);
      if (relaxes) {
        last_relax = t;
      }
      if (mask[t] & bit) {
        last_bind = t;
        if (relaxes) {
          target = t;
        }
      }
    }
    if (last_relax < last_bind) {
      target = -1;
    }

    for (i64 t = 0; t < T; ++t) {
      const int keep = (next[t] & bit) || ((mask[t] & bit) && t != target);
      mask[t] = (i8)((mask[t] & ~bit) | (keep ? bit : 0));
    }
  }
}

/* Dynare's gate for treating a repeated guess as periodic anyway: some pass
   that wanted no new binding date moved the guess by little enough. The
   comparison is `<=` in the one-constraint file and `<` in the two-constraint
   one, and both are kept. */
static i64 sdsge_occbin_promotable(const i64 *move, const i8 *binds, i64 iter,
                                   i64 threshold, i64 n_constraint) {
  for (i64 k = 0; k < iter; ++k) {
    if (binds[k]) {
      continue;
    }
    if (n_constraint > 1 ? move[k] < threshold : move[k] <= threshold) {
      return 1;
    }
  }
  return 0;
}

arena_size sdsge_occbin_period_arena_size(i64 n_var, i64 n_state, i64 n_ctrl,
                                          i64 T_cap, i64 max_iter) {
  const arena_size rec =
      sdsge_occbin_recursion_arena_size(n_var, n_state, n_ctrl);
  // hist, and next, all_bind and stay beside it
  const i64 bytes = (max_iter + 3) * T_cap + max_iter;
  return make_sizer(T_cap * n_var + max_iter + rec.n_float,
                    rec.n_int + 2 * max_iter + (bytes + 7) / 8);
}

i64 sdsge_occbin_period(const occbin_run_ctx *run, const f64 *SDSGE_RESTRICT x0,
                        i8 *mask, i64 *T, f64 *rule, f64 *path,
                        occbin_diag *diag, i64 s, f64 *arena, i64 *iarena) {
  const i64 n_var = run->model->n_var;
  const i64 n_state = run->model->n_state;

  const i64 T_cap = run->T_cap;
  const i64 max_iter = run->max_iter;

  f64 *lev = arena;                         // (T_cap, n_var) levels
  f64 *err_hist = lev + T_cap * n_var;      // (max_iter,)
  f64 *rec = err_hist + max_iter;           // recursion scratch
  i64 *hist_len = iarena + n_var;           // (max_iter,)
  i64 *move_hist = hist_len + max_iter;     // (max_iter,)
  i8 *next = (i8 *)(move_hist + max_iter);  // (T_cap,)
  i8 *hist = next + T_cap;                  // (max_iter, T_cap)
  i8 *binds_hist = hist + max_iter * T_cap; // (max_iter,)
  i8 *all_bind = binds_hist + max_iter;     // (T_cap,)
  i8 *stay = all_bind + T_cap;              // (T_cap,)

  i64 changed = 1;
  i64 iter = 0;
  i64 grew = 0;
  i64 cycle = -1; // iteration whose guess the updated one repeats

  if (run->opts.reset_regime) {
    if (run->opts.reset_check_ahead) {
      *T = run->T0;
    }
    memset(mask, 0, *T);
  }
  if (run->opts.curb_retrench) {
    memset(all_bind, (1 << run->n_constraint) - 1, T_cap);
  }

  while (changed && iter < max_iter) {
    if (mask[*T - 1] != 0 && *T < T_cap) {
      // last period is binding, append a relaxing regime at the end.
      mask[*T] = 0;
      ++(*T);
      grew = 1; // Dynare stops looking for cycles in a period that grew
    } else if (mask[*T - 1] != 0) {
      // binding at the cap, clip the last period to a relaxing regime.
      mask[*T - 1] = 0;
    }

    memcpy(hist + iter * T_cap, mask, *T);
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

    err_hist[iter - 1] = 0.0;
    changed = sdsge_constraint_path(run->cond, lev, run->par, mask, next,
                                    run->inclusive, &err_hist[iter - 1], *T,
                                    n_var, run->n_constraint);
    move_hist[iter - 1] =
        sdsge_occbin_regime_move(mask, next, *T, run->n_constraint);
    binds_hist[iter - 1] = (i8)sdsge_occbin_binds_more(mask, next, *T);

    if (run->opts.curb_retrench) {
      // Reading every date as binding makes the latch report the relax
      // condition everywhere, which the real sweep never sees at a date the
      // guess already has relaxed.
      f64 ignored = 0.0;
      sdsge_constraint_path(run->cond, lev, run->par, all_bind, stay,
                            run->inclusive, &ignored, *T, n_var,
                            run->n_constraint);
      sdsge_occbin_retrench(mask, next, stay, *T, run->n_constraint);
    } else {
      memcpy(mask, next, *T);
    }

    // A guess seen before is a cycle; the newest match dates its length. The
    // guess entering this iteration cannot match, `changed` says it moved.
    if (changed && !grew) {
      for (i64 k = iter - 2; k >= 0; --k) {
        if (hist_len[k] == *T && memcmp(hist + k * T_cap, mask, *T) == 0) {
          cycle = k;
          break;
        }
      }
      if (cycle >= 0) {
        break;
      }
    }
  }

  diag->iters[s] = iter;
  diag->max_err[s] = err_hist[iter - 1];
  diag->periodic[s] = 0;

  i64 code = SDSGE_OCCBIN_PERIOD_OK;
  if (cycle >= 0) {
    // Only a two-cycle whose guess barely moved is periodic; every other
    // repeat is an ordinary loop over guesses.
    const i64 two = cycle == iter - 2 &&
                    sdsge_occbin_regime_move(hist + (iter - 1) * T_cap, mask,
                                             *T, run->n_constraint) <=
                        run->opts.periodic_threshold;
    code = two ? SDSGE_OCCBIN_PERIODIC : SDSGE_OCCBIN_PERIODIC_LOOP;
  } else if (changed) {
    code = SDSGE_OCCBIN_MAXITER;
  }
  if (code == SDSGE_OCCBIN_PERIOD_OK) {
    return code;
  }

  if (max_iter <= run->opts.algo_truncation) {
    // Truncated on purpose: rule and path already belong to the last guess.
    *T = hist_len[iter - 1];
    memcpy(mask, hist + (iter - 1) * T_cap, *T);
    return SDSGE_OCCBIN_PERIOD_OK;
  }
  if (!run->opts.periodic_solution) {
    return code;
  }

  // A strict two-cycle weighs only the pair that repeats. A loop is weighed
  // over every guess, and only when the strictness flag lets it through.
  i64 lo;
  if (code == SDSGE_OCCBIN_PERIODIC) {
    lo = iter - 2;
  } else if (!run->opts.periodic_strict &&
             sdsge_occbin_promotable(move_hist, binds_hist, iter,
                                     run->opts.periodic_threshold,
                                     run->n_constraint)) {
    lo = 0;
  } else {
    return code;
  }

  // Passes that wanted no new binding date are preferred outright, and a
  // promoted loop then ranks them by how little the guess moved.
  i64 clean = 0;
  for (i64 k = lo; k < iter; ++k) {
    clean |= !binds_hist[k];
  }
  const i64 by_move = code != SDSGE_OCCBIN_PERIODIC && clean;

  i64 best = -1;
  for (i64 k = lo; k < iter; ++k) {
    if (clean && binds_hist[k]) {
      continue;
    }
    if (best < 0 || (by_move ? move_hist[k] < move_hist[best]
                             : err_hist[k] < err_hist[best])) {
      best = k;
    }
  }

  *T = hist_len[best];
  memcpy(mask, hist + best * T_cap, *T);
  if (best != iter - 1) {
    // rule and path still describe the last guess tried, not this one.
    i64 status = sdsge_occbin_recursion(run->model, mask, *T, rule,
                                        &diag->singular_date, rec, iarena);
    if (status != SDSGE_OCCBIN_RECURSION_OK) {
      return status;
    }
    sdsge_occbin_forward(rule, x0, *T, n_var, n_state, path);
  }
  diag->max_err[s] = err_hist[best];
  diag->periodic[s] = (i8)code;
  return SDSGE_OCCBIN_PERIOD_OK;
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

arena_size sdsge_occbin_sim_arena_size(i64 n_var, i64 n_state, i64 n_ctrl,
                                       i64 T_cap, i64 max_iter) {
  const arena_size per =
      sdsge_occbin_period_arena_size(n_var, n_state, n_ctrl, T_cap, max_iter);
  return make_sizer(n_state + per.n_float, per.n_int);
}

i64 sdsge_occbin_sim(const occbin_run_ctx *run, const f64 *shocks, i64 S,
                     i64 n_periods, const f64 *x_init, const i8 *init_mask,
                     i64 n_init, f64 *out, i8 *regimes, i64 *T_used,
                     occbin_diag *diag, f64 *rule, f64 *path, i8 *mask,
                     f64 *arena, i64 *iarena) {
  const i64 n_var = run->model->n_var;
  const i64 n_state = run->model->n_state;
  const i64 n_rhs = n_state + 1;
  i64 T = run->T0;

  f64 *x = arena; // (n_state,) state entering the current shock period
  f64 *period_arena = x + n_state;

  diag->fail_period = -1;
  diag->singular_date = -1;

  memset(mask, 0, run->T_cap);
  memset(path, 0, T * n_var * sizeof(f64));
  memcpy(x, x_init, n_state * sizeof(f64));

  for (i64 s = 0; s < S; ++s) {
    // A supplied guess replaces the carried one. Past `n_init` the previous
    // period's guess stands, shifted, which is where Dynare's history ends too.
    if (s < n_init) {
      memcpy(mask, init_mask + s * run->T_cap, run->T_cap);
    }

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
      diag->periodic[s] = 0;
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
  return SDSGE_OCCBIN_SIM_OK;
}

arena_size sdsge_occbin_solve1_arena_size(i64 n_var, i64 n_state, i64 n_ctrl,
                                          i64 n_par, i64 max_n_row) {
  arena_size pencil = sdsge_regime_pencil_arena_size(n_var, max_n_row);
  arena_size klein =
      sdsge_klein_solve1_arena_size(n_var, n_state, n_ctrl, n_par);

  // Klein and pencil are sequential, maximum of both is enough for the arena to
  // hold either.
  return sdsge_max_arena(klein, pencil);
}

// "Solve" means build the reference regime and the pencils for every other
// regime. Transitions per-path are not predetermined, making "sim" the actual
// solver.
i64 sdsge_occbin_solve1(const occbin_solve1_spec *spec, sdsge_occbin1 *out,
                        f64 *SDSGE_RESTRICT arena, i64 *SDSGE_RESTRICT iarena) {
  const i64 rc = sdsge_klein_solve1(&spec->first, &out->ref, arena, iarena);
  if (rc != SDSGE_KLEIN_SOLVE_OK) {
    return rc;
  }

  const i64 n_var = spec->first.n_var;
  const i64 blk = n_var * n_var;

  i64 n_regime = 1 << spec->n_constraint;

  regime_ctx table[SDSGE_MAX_REGIME];
  for (i64 m = 0; m < n_regime; ++m) {
    table[m].pencil = spec->pencil[m];
    table[m].n_row = spec->n_row[m];
    table[m].rows = spec->rows[m];
    table[m].a = out->a + m * blk;
    table[m].b = out->b + m * blk;
    table[m].c = out->c + m * n_var;
  }

  regime_table(table, n_regime, out->ref.ss, spec->first.params,
               out->ref.a_real, out->ref.b_real, n_var, arena);
  return rc;
}
