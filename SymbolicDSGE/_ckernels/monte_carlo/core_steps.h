#ifndef SDSGE_MC_CORE_STEPS
#define SDSGE_MC_CORE_STEPS

#include "../_common/sdsge_common.h"
#include "../core/core.h"
#include "../kalman/kalman.h"
#include "runner.h"
#include "shocks.h"

/* Static configuration for generic native MC step dispatch. Dynamic numeric
 * inputs, scratch, and outputs live in caller-owned arenas. */
typedef struct {
  const f64 *input;
  i64 n;
  int input_batched;
} sdsge_mc_payload_step_ctx;

typedef struct {
  const f64 *states_input;
  i64 n_states;
  int states_batched;
  const f64 *observables_input;
  i64 n_observables;
  int observables_batched;
} sdsge_mc_raw_model_data_step_ctx;

/* `shocks` selects how the (T, k) shock block in the input arena is populated.
 * NULL means the runner already copied it in through a static binding (the
 * Python prematerialization route). Otherwise the step draws its own shocks
 * from `rep_idx` before simulating, and `shock_scratch_offset` locates the
 * draw's scratch just past the arena the simulation itself needs. */
typedef struct {
  meas_fn measurement;
  i64 T;
  i64 n;
  i64 k;
  i64 n_par;
  i64 m;
  const sdsge_mc_shock_plan *shocks;
  i64 shock_scratch_offset;
} sdsge_mc_simulate_order1_step_ctx;

typedef struct {
  meas_fn measurement;
  i64 T;
  i64 n_state;
  i64 n_ctrl;
  i64 n_exog;
  i64 n_par;
  i64 m;
  const sdsge_mc_shock_plan *shocks;
  i64 shock_scratch_offset;
} sdsge_mc_simulate_order2_step_ctx;

typedef struct {
  i64 T;
  i64 n;
  i64 m;
  i64 k;
  int symmetrize;
  f64 jitter;
  int return_shocks;
} sdsge_mc_filter_linear_step_ctx;

typedef struct {
  meas_fn measurement;
  meas_fn jacobian;
  i64 T;
  i64 n;
  i64 m;
  i64 k;
  i64 n_par;
  int symmetrize;
  f64 jitter;
  int return_shocks;
} sdsge_mc_filter_extended_step_ctx;

typedef struct {
  meas_fn measurement;
  i64 T;
  i64 n_state;
  i64 n_ctrl;
  i64 n_exog;
  i64 n_obs;
  i64 n_par;
  f64 alpha;
  f64 beta;
  f64 kappa;
  int symmetrize;
  f64 jitter;
} sdsge_mc_filter_unscented_step_ctx;

/* Generic-runner adapters. They preserve the canonical arena layouts of the
 * direct kernels below and write one status code when ``int_out`` is non-NULL.
 */
int sdsge_mc_payload_runner(i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
                            f64 *SDSGE_RESTRICT float_out,
                            i64 *SDSGE_RESTRICT int_work,
                            i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_raw_model_data_runner(
    i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_simulate_order1_runner(
    i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_simulate_order2_runner(
    i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_filter_linear_runner(
    i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_filter_extended_runner(
    i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_filter_unscented_runner(
    i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx);

/* Payload materialization. ``input_batched`` selects input[rep_idx] from a
 * leading replication axis; otherwise the same input span is copied each time. */
void sdsge_add_payload_step(const f64 *SDSGE_RESTRICT input, i64 n,
                            int input_batched, i64 rep_idx,
                            f64 *SDSGE_RESTRICT output);

/* Raw model-data materialization. ``*_batched`` selects input[rep_idx] from a
 * leading replication axis; otherwise the same input span is copied each time.
 * Either input/output pair may be NULL when its element count is zero. */
void sdsge_raw_model_data_step(const f64 *SDSGE_RESTRICT states_input,
                               i64 n_states, int states_batched,
                               f64 *SDSGE_RESTRICT states_output,
                               const f64 *SDSGE_RESTRICT observables_input,
                               i64 n_observables, int observables_batched,
                               i64 rep_idx,
                               f64 *SDSGE_RESTRICT observables_output);

/* ``input`` is [A(n,n), B(n,k), x0(n), shock(T,k), params(n_par)].
 * ``simout`` is [states(T,n), observables(T,m)]. */
i64 sdsge_simulate_order1_arena_size(i64 n, i64 k, i64 T, i64 n_par);
void sdsge_simulate_order1_step(f64 *SDSGE_RESTRICT arena,
                                meas_fn measurement, i64 T, i64 n,
                                i64 k, i64 n_par, i64 m,
                                f64 *SDSGE_RESTRICT simout);

/* ``input`` is [hx(nx,nx), gx(ny,nx), bx(nx,n_exog), hxx(nx,nx,nx),
 * gxx(ny,nx,nx), hss(nx), gss(ny), steady_state(nx+ny), x0(nx),
 * shock(T,n_exog), params(n_par), scratch(4*nx + nx*nx)]. ``simout`` is
 * [states(T,nx+ny), observables(T,m)]. */
i64 sdsge_simulate_order2_arena_size(i64 n_state, i64 n_var, i64 n_exog, i64 T,
                                     i64 n_par);
void sdsge_simulate_order2_step(f64 *SDSGE_RESTRICT arena,
                                meas_fn measurement, i64 T, i64 nx,
                                i64 ny, i64 n_exog, i64 n_par, i64 m,
                                f64 *SDSGE_RESTRICT simout);

/* Linear filter input is [A(n,n), B(n,k), C(m,n), d(m), Q(k,k), R(m,m),
 * y(T,m), x0(n), P0(n,n)]. Output follows FilterRawResult field order:
 * [x_pred, x_filt, P_pred, P_filt, y_pred, y_filt, innov, std_innov, S,
 * eps_hat (when return_shocks), loglik]. */
i64 sdsge_filter_linear_input_arena_size(i64 n, i64 m, i64 k, i64 T);
i64 sdsge_filter_linear_output_arena_size(i64 n, i64 m, i64 k, i64 T,
                                           int return_shocks);
int sdsge_filter_linear_step(const f64 *SDSGE_RESTRICT input_arena,
                             f64 *SDSGE_RESTRICT scratch_arena, i64 T, i64 n,
                             i64 m, i64 k, int symmetrize, f64 jitter,
                             int return_shocks,
                             f64 *SDSGE_RESTRICT output_arena);

/* Extended filter input is [A(n,n), B(n,k), params(n_par), Q(k,k), R(m,m),
 * y(T,m), x0(n), P0(n,n)]. Output has the linear filter layout. */
i64 sdsge_filter_extended_input_arena_size(i64 n, i64 m, i64 k, i64 T,
                                            i64 n_par);
i64 sdsge_filter_extended_output_arena_size(i64 n, i64 m, i64 k, i64 T,
                                             int return_shocks);
int sdsge_filter_extended_step(const f64 *SDSGE_RESTRICT input_arena,
                               f64 *SDSGE_RESTRICT scratch_arena, meas_fn meas,
                               meas_fn jac, i64 T, i64 n, i64 m, i64 k,
                               i64 n_par, int symmetrize, f64 jitter,
                               int return_shocks,
                               f64 *SDSGE_RESTRICT output_arena);

/* Unscented input is [hx(ns,ns), gx(nc,ns), bx(ns,ne), hxx(ns,ns,ns),
 * gxx(nc,ns,ns), hss(ns), gss(nc), steady_state(ns+nc), params(n_par),
 * Q(ne,ne), R(no,no), obs(T,no), z0(2*ns), P0(2*ns,2*ns)]. Output follows
 * UnscentedFilterRawResult field order, omitting eps_hat (always None). */
i64 sdsge_filter_unscented_input_arena_size(i64 n_state, i64 n_ctrl,
                                             i64 n_exog, i64 n_obs, i64 T,
                                             i64 n_par);
i64 sdsge_filter_unscented_output_arena_size(i64 n_state, i64 n_ctrl,
                                              i64 n_obs, i64 T);
i64 sdsge_filter_unscented_step(const f64 *SDSGE_RESTRICT input_arena,
                                f64 *SDSGE_RESTRICT scratch_arena, meas_fn meas,
                                i64 T, i64 n_state, i64 n_ctrl, i64 n_exog,
                                i64 n_obs, i64 n_par, f64 alpha, f64 beta,
                                f64 kappa, int symmetrize, f64 jitter,
                                f64 *SDSGE_RESTRICT output_arena);

#endif /* SDSGE_MC_CORE_STEPS */
