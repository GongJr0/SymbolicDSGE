#ifndef SDSGE_OCCBIN_H
#define SDSGE_OCCBIN_H

#include "../_common/sdsge_common.h"
#include "../core/klein_solve.h"
#include "regime_pencil.h"

// regime upper bound: 2^n_constraint with n_constraint <= 2
// by definition for OccBin.
#define SDSGE_MAX_CONSTRAINT 2
#define SDSGE_MAX_REGIME (1 << SDSGE_MAX_CONSTRAINT)

typedef void (*sdsge_constraint_fn)(f64 *cur, f64 *par, f64 *err);

i64 sdsge_constraint_path(sdsge_constraint_fn cond,
                          f64 *SDSGE_RESTRICT path, // (T, n_var)
                          f64 *SDSGE_RESTRICT par,  // (n_par,)
                          const i8 *regime_in,      // (T,)
                          i8 *regime_out,           // (T,)
                          i64 inclusive, // Bitmask for inequality strictness
                          f64 *SDSGE_RESTRICT max_err, i64 T, i64 n_var,
                          i64 n_constraint);
// Pencils stacked by binding bitmask, slot 0 the reference. Flat rather than a
// regime_ctx table: those carry the build role (cfunc, rows) that is spent once
// in regime_table, and a caller holding arrays would have to lower them again.
typedef struct {
  const f64 *a;   // (n_regime, n_var, n_var)
  const f64 *b;   // (n_regime, n_var, n_var)
  const f64 *c;   // (n_regime, n_var, n_var)  lag block
  const f64 *d;   // (n_regime, n_var, n_exog) shock block
  const f64 *cst; // (n_regime, n_var)         constant
  // (n_var, n_state) the reference regime's whole rule, y_t = ghx y_{t-1}.
  // The recursion closes on it past the horizon, where the guess is relaxed.
  // Full width rather than f_ref: the unknown is the whole vector now, so the
  // state/control split has nothing to say here.
  const f64 *ghx_ref;
  i64 n_var;
  i64 n_state;
  i64 n_ctrl;
  i64 n_exog;
} occbin_ctx;

typedef struct {
  i64 periodic_solution;  // accept the best iteration instead of failing
  i64 periodic_threshold; // largest signed move in binding dates still a cycle
  i64 periodic_strict;    // a repeat that is not a two-cycle is a failure
  i64 curb_retrench;      // relax one date per pass instead of all of them
  i64 reset_regime;       // start each shock period from the relaxed guess
  i64 reset_check_ahead;  // a reset drops an endogenously grown horizon to T0
  i64 algo_truncation;    // a max_iter at or below this accepts the last guess
} occbin_opts;

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
  occbin_opts opts;
} occbin_run_ctx;

typedef struct {
  i64 *iters;
  f64 *max_err;
  i8 *periodic;
  i64 fail_period;
  i64 singular_date;
} occbin_diag;

typedef struct {
  klein_spec first;
  const sdsge_regime_pencil_fn *pencil; // (n_regime,) NULL for reference regime
  const i64 *const *rows; // (n_regime,) each points to (n_row,) indices of the
                          // rows a regime swaps
  const i64 *n_row;       // (n_regime,) number of rows a regime swaps
  i64 n_constraint;       // n_regime = 2^n_constraint
} occbin_solve1_spec;

typedef struct {
  sdsge_solve1 ref;
  f64 *a;   // (n_regime, n_var, n_var)
  f64 *b;   // (n_regime, n_var, n_var)
  f64 *c;   // (n_regime, n_var, n_var)  lag block
  f64 *d;   // (n_regime, n_var, n_exog) shock block
  f64 *cst; // (n_regime, n_var)         constant
} sdsge_occbin1;

arena_size sdsge_occbin_recursion_arena_size(i64 n_var, i64 n_state,
                                             i64 n_exog);

i64 sdsge_occbin_recursion(
    const occbin_ctx *ctx, const i8 *SDSGE_RESTRICT mask, i64 T,
    f64 *SDSGE_RESTRICT out, // (T, n_var, n_state + n_exog + 1)
    i64 *SDSGE_RESTRICT
        singular_date, // `t` if the recursion is singular, -1 otherwise
    f64 *SDSGE_RESTRICT arena, i64 *SDSGE_RESTRICT iarena);

// `x0` is y_{t-1} at the first date, so row t of `path` is y_t and the seed is
// not itself a row. `eps0` is (n_exog,) and lands on date 0 alone: a shock
// later in the projection is unforeseen, which is what makes it zero here.
// NULL for a period with no innovation.
void sdsge_occbin_forward(const f64 *SDSGE_RESTRICT rule,
                          const f64 *SDSGE_RESTRICT x0,
                          const f64 *SDSGE_RESTRICT eps0, i64 T, i64 n_var,
                          i64 n_state, i64 n_exog, f64 *SDSGE_RESTRICT path);

arena_size sdsge_occbin_period_arena_size(i64 n_var, i64 n_state, i64 n_ctrl,
                                          i64 n_exog, i64 T_cap, i64 max_iter);

// `mask`, `T` and `path` are updated in place; `diag` is written at slot `s`.
// Requires `max_iter >= 1`: the guess is read once before anything is weighed.
i64 sdsge_occbin_period(const occbin_run_ctx *run,
                        const f64 *SDSGE_RESTRICT x0,   // (n_state,)
                        const f64 *SDSGE_RESTRICT eps0, // (n_exog,)
                        i8 *mask,                       // (T_cap,)
                        i64 *T, f64 *rule, f64 *path, occbin_diag *diag, i64 s,
                        f64 *arena, i64 *iarena);

arena_size sdsge_occbin_sim_arena_size(i64 n_var, i64 n_state, i64 n_ctrl,
                                       i64 n_exog, i64 T_cap, i64 max_iter);

// Requires `T0 >= 2` and `n_periods - S <= T0`. `init_mask` holds one guess per
// shock period, in the layout `regimes` comes back in, for the first `n_init`
// of them; pass NULL with `n_init` 0 to start relaxed. Periods past `n_init`
// take the previous period's guess, shifted.
i64 sdsge_occbin_sim(const occbin_run_ctx *run,
                     const f64 *shocks, // (S, n_exog)
                     i64 S, i64 n_periods,
                     const f64 *x_init,   // (n_state,)
                     const i8 *init_mask, // (n_init, T_cap)
                     i64 n_init,          // 0 <= n_init <= S
                     f64 *out,            // (n_periods, n_var)
                     i8 *regimes,         // (S, T_cap)
                     i64 *T_used,         // (S,)
                     occbin_diag *diag,   // per-period arrays are (S,)
                     f64 *rule, // (T_cap, n_var, n_state + n_exog + 1) scratch
                     f64 *path,           // (T_cap, n_var) scratch
                     i8 *mask,            // (T_cap,) scratch
                     f64 *arena, i64 *iarena);

arena_size sdsge_occbin_solve1_arena_size(i64 n_var, i64 n_state, i64 n_ctrl,
                                          i64 n_par, i64 n_exog, i64 nd,
                                          i64 max_n_row);

i64 sdsge_occbin_solve1(const occbin_solve1_spec *spec, sdsge_occbin1 *out,
                        f64 *arena, i64 *iarena);

#define SDSGE_OCCBIN_RECURSION_OK 0
#define SDSGE_OCCBIN_RECURSION_SINGULAR -1401 // the period's pencil is singular

// return codes (mapped to Dynare's occbin codes)
#define SDSGE_OCCBIN_PERIOD_OK 0
#define SDSGE_OCCBIN_PERIODIC 1      // 310 in Dynare
#define SDSGE_OCCBIN_PERIODIC_LOOP 2 // 313 in Dynare
#define SDSGE_OCCBIN_MAXITER 3       // 311 in Dynare

#define SDSGE_OCCBIN_SIM_OK 0

#endif // SDSGE_OCCBIN_H
