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
  i64 n;
  i64 p;
} sdsge_mc_passthrough_step_ctx;

typedef struct {
  const f64 *states_input;
  const f64 *shocks_input;
  const f64 *observables_input;
  int states_batched;
  int shocks_batched;
  int observables_batched;
  i64 n_states;
  i64 n_shocks;
  i64 n_observables;
} sdsge_mc_raw_model_data_step_ctx;

/* `shocks` selects how the (T, k) shock block in the input arena is populated.
 * NULL means the runner already copied it in through a static binding (the
 * Python prematerialization route). Otherwise the step draws its own shocks
 * from `rep_idx` before simulating, and `shock_scratch_offset` locates the
 * draw's scratch just past the arena the simulation itself needs. */
typedef struct {
  sdsge_measurement_fn measurement;
  i64 T;
  i64 n;
  i64 k;
  i64 n_par;
  i64 m;
  const sdsge_mc_shock_plan *shocks;
  i64 shock_scratch_offset;
} sdsge_mc_simulate_order1_step_ctx;

typedef struct {
  sdsge_measurement_fn measurement;
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
  int joseph_cov;
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
  int joseph_cov;
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
int sdsge_mc_passthrough_runner(i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
                                f64 *SDSGE_RESTRICT float_out,
                                i64 *SDSGE_RESTRICT int_work,
                                i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_raw_model_data_runner(i64 rep_idx,
                                   f64 *SDSGE_RESTRICT float_in_work,
                                   f64 *SDSGE_RESTRICT float_out,
                                   i64 *SDSGE_RESTRICT int_work,
                                   i64 *SDSGE_RESTRICT int_out,
                                   const void *ctx);
int sdsge_mc_simulate_order1_runner(i64 rep_idx,
                                    f64 *SDSGE_RESTRICT float_in_work,
                                    f64 *SDSGE_RESTRICT float_out,
                                    i64 *SDSGE_RESTRICT int_work,
                                    i64 *SDSGE_RESTRICT int_out,
                                    const void *ctx);
int sdsge_mc_simulate_order2_runner(i64 rep_idx,
                                    f64 *SDSGE_RESTRICT float_in_work,
                                    f64 *SDSGE_RESTRICT float_out,
                                    i64 *SDSGE_RESTRICT int_work,
                                    i64 *SDSGE_RESTRICT int_out,
                                    const void *ctx);
int sdsge_mc_filter_linear_runner(i64 rep_idx,
                                  f64 *SDSGE_RESTRICT float_in_work,
                                  f64 *SDSGE_RESTRICT float_out,
                                  i64 *SDSGE_RESTRICT int_work,
                                  i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_filter_extended_runner(i64 rep_idx,
                                    f64 *SDSGE_RESTRICT float_in_work,
                                    f64 *SDSGE_RESTRICT float_out,
                                    i64 *SDSGE_RESTRICT int_work,
                                    i64 *SDSGE_RESTRICT int_out,
                                    const void *ctx);
int sdsge_mc_filter_unscented_runner(i64 rep_idx,
                                     f64 *SDSGE_RESTRICT float_in_work,
                                     f64 *SDSGE_RESTRICT float_out,
                                     i64 *SDSGE_RESTRICT int_work,
                                     i64 *SDSGE_RESTRICT int_out,
                                     const void *ctx);

/* Payload materialization. ``input_batched`` selects input[rep_idx] from a
 * leading replication axis; otherwise the same input span is copied each time.
 */
void sdsge_add_payload_step(const f64 *SDSGE_RESTRICT input, i64 n,
                            int input_batched, i64 rep_idx,
                            f64 *SDSGE_RESTRICT output);

void sdsge_passthrough_step(const f64 *SDSGE_RESTRICT input, i64 n, i64 p,
                            f64 *SDSGE_RESTRICT output);

void sdsge_raw_model_data_step(
    const i64 rep_idx, const f64 *SDSGE_RESTRICT states_input,
    const f64 *SDSGE_RESTRICT shocks_input,
    const f64 *SDSGE_RESTRICT observables_input, const int states_batched,
    const int shocks_batched, const int observables_batched, const i64 n_states,
    const i64 n_shocks, const i64 n_observables,
    f64 *SDSGE_RESTRICT states_output, f64 *SDSGE_RESTRICT shocks_output,
    f64 *SDSGE_RESTRICT observables_output);

/* ``input`` is [A(n,n), B(n,k), steady_state(n), x0(n), shock(T,k),
 * params(n_par), scratch].
 *
 * The step splits that arena and calls core's parameterized kernel; the flat
 * layout is this pipeline's own requirement and stops here. */
void sdsge_simulate_order1_step(f64 *SDSGE_RESTRICT arena,
                                sdsge_measurement_fn measurement, i64 T, i64 n,
                                i64 k, i64 n_par, i64 m,
                                f64 *SDSGE_RESTRICT simout);

void sdsge_simulate_order2_step(f64 *SDSGE_RESTRICT arena,
                                sdsge_measurement_fn measurement, i64 T, i64 nx,
                                i64 ny, i64 n_exog, i64 n_par, i64 m,
                                f64 *SDSGE_RESTRICT simout);

int sdsge_filter_linear_step(const f64 *SDSGE_RESTRICT input_arena,
                             f64 *SDSGE_RESTRICT scratch_arena, i64 T, i64 n,
                             i64 m, i64 k, int symmetrize, int joseph_cov,
                             f64 jitter, int return_shocks,
                             f64 *SDSGE_RESTRICT output_arena);

int sdsge_filter_extended_step(const f64 *SDSGE_RESTRICT input_arena,
                               f64 *SDSGE_RESTRICT scratch_arena, meas_fn meas,
                               meas_fn jac, i64 T, i64 n, i64 m, i64 k,
                               i64 n_par, int symmetrize, int joseph_cov,
                               f64 jitter, int return_shocks,
                               f64 *SDSGE_RESTRICT output_arena);

i64 sdsge_filter_unscented_step(const f64 *SDSGE_RESTRICT input_arena,
                                f64 *SDSGE_RESTRICT scratch_arena, meas_fn meas,
                                i64 T, i64 n_state, i64 n_ctrl, i64 n_exog,
                                i64 n_obs, i64 n_par, f64 alpha, f64 beta,
                                f64 kappa, int symmetrize, f64 jitter,
                                f64 *SDSGE_RESTRICT output_arena);

#endif /* SDSGE_MC_CORE_STEPS */
