#ifndef SDSGE_OCCBIN_H
#define SDSGE_OCCBIN_H

#include "../_common/sdsge_common.h"
#include "regime_pencil.h"

typedef void (*sdsge_constraint_fn)(f64 *cur, f64 *par, f64 *err);

i64 sdsge_constraint_path(sdsge_constraint_fn cond,
                          f64 *SDSGE_RESTRICT path, // (T, n_var)
                          f64 *SDSGE_RESTRICT par,  // (n_par,)
                          const i8 *regime_in,      // (T,)
                          i8 *regime_out,           // (T,)
                          i64 inclusive, // Bitmask for inequality strictness
                          f64 *SDSGE_RESTRICT max_err, i64 T, i64 n_var,
                          i64 n_constraint);
typedef struct {
  const regime_ctx *table;
  const f64 *f_ref;
  i64 n_var;
  i64 n_state;
  i64 n_ctrl;
} occbin_ctx;

typedef struct {
  const occbin_ctx *model;
  sdsge_constraint_fn cond;
  f64 *par;
  const f64 *ss; // (n_var,) reference steady state; the cfunc reads levels
  i64 inclusive;
  i64 n_constraint;
  i64 max_iter; // Dynare default == 30
  i64 T0; // Check ahead + 1 (additional period to append a relaxing regime if T
          // terminates in a binding regime)
  i64 T_cap;
} occbin_run_ctx;

typedef struct {
  i64 *iters;
  f64 *max_err;
  i64 fail_period;
  i64 singular_date;
} occbin_diag;

arena_size sdsge_occbin_recursion_arena_size(i64 n_var, i64 n_state,
                                             i64 n_ctrl);

i64 sdsge_occbin_recursion(
    const occbin_ctx *ctx, const i8 *SDSGE_RESTRICT mask, i64 T,
    f64 *SDSGE_RESTRICT out, // (T, n_var, n_state + 1)
    i64 *SDSGE_RESTRICT
        singular_date, // `t` if the recursion is singular, -1 otherwise
    f64 *SDSGE_RESTRICT arena, i64 *SDSGE_RESTRICT iarena);

void sdsge_occbin_forward(const f64 *SDSGE_RESTRICT rule,
                          const f64 *SDSGE_RESTRICT x0, i64 T, i64 n_var,
                          i64 n_state, f64 *SDSGE_RESTRICT path);

arena_size sdsge_occbin_period_arena_size(i64 n_var, i64 n_state, i64 n_ctrl,
                                          i64 T_cap, i64 max_iter);

// `mask`, `T` and `path` are updated in place; `diag` is written at slot `s`.
i64 sdsge_occbin_period(const occbin_run_ctx *run,
                        const f64 *SDSGE_RESTRICT x0, // (n_state,)
                        i8 *mask,                     // (T_cap,)
                        i64 *T, f64 *rule, f64 *path, occbin_diag *diag, i64 s,
                        f64 *arena, i64 *iarena);

arena_size sdsge_occbin_solve_arena_size(i64 n_var, i64 n_state, i64 n_ctrl,
                                         i64 T_cap, i64 max_iter);

// Requires `T0 >= 2` and `n_periods - S <= T0`.
i64 sdsge_occbin_solve(const occbin_run_ctx *run,
                       const f64 *shocks, // (S, n_state)
                       i64 S, i64 n_periods,
                       const f64 *x_init, // (n_state,)
                       f64 *out,          // (n_periods, n_var)
                       i8 *regimes,       // (S, T_cap)
                       i64 *T_used,       // (S,)
                       occbin_diag *diag, // iters and max_err are (S,)
                       f64 *rule,         // (T_cap, n_var, n_state + 1) scratch
                       f64 *path,         // (T_cap, n_var) scratch
                       i8 *mask,          // (T_cap,) scratch
                       f64 *arena, i64 *iarena);

#define SDSGE_OCCBIN_RECURSION_OK 0
#define SDSGE_OCCBIN_RECURSION_SINGULAR -2 // match code to LU factorization

// return codes matching Dynare's
#define SDSGE_OCCBIN_PERIOD_OK 0
#define SDSGE_OCCBIN_PERIODIC 310
#define SDSGE_OCCBIN_PERIODIC_LOOP 313
#define SDSGE_OCCBIN_MAXITER 311

#define SDSGE_OCCBIN_SOLVE_OK 0

#endif // SDSGE_OCCBIN_H
