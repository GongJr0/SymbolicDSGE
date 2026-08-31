#include "core_steps.h"
#include "../core/core.h"
#include "layout.h"
#include <string.h>

static int sdsge_mc_finish_status(const int status,
                                  i64 *SDSGE_RESTRICT int_out) {
  if (int_out != NULL)
    int_out[0] = (i64)status;
  return status;
}

int sdsge_mc_payload_runner(const i64 rep_idx,
                            f64 *SDSGE_RESTRICT float_in_work,
                            f64 *SDSGE_RESTRICT float_out,
                            i64 *SDSGE_RESTRICT int_work,
                            i64 *SDSGE_RESTRICT int_out, const void *ctx_ptr) {
  const sdsge_mc_payload_step_ctx *ctx = ctx_ptr;
  (void)float_in_work;
  (void)int_work;
  sdsge_add_payload_step(ctx->input, ctx->n, ctx->input_batched, rep_idx,
                         float_out);
  return sdsge_mc_finish_status(SDSGE_OK, int_out);
}

int sdsge_mc_passthrough_runner(const i64 rep_idx,
                                f64 *SDSGE_RESTRICT float_in_work,
                                f64 *SDSGE_RESTRICT float_out,
                                i64 *SDSGE_RESTRICT int_work,
                                i64 *SDSGE_RESTRICT int_out,
                                const void *ctx_ptr) {
  const sdsge_mc_passthrough_step_ctx *ctx = ctx_ptr;
  (void)rep_idx;
  (void)int_work;
  sdsge_passthrough_step(float_in_work, ctx->n, ctx->p, float_out);
  return sdsge_mc_finish_status(SDSGE_OK, int_out);
}

int sdsge_mc_raw_model_data_runner(const i64 rep_idx,
                                   f64 *SDSGE_RESTRICT float_in_work,
                                   f64 *SDSGE_RESTRICT float_out,
                                   i64 *SDSGE_RESTRICT int_work,
                                   i64 *SDSGE_RESTRICT int_out,
                                   const void *ctx_ptr) {
  const sdsge_mc_raw_model_data_step_ctx *ctx = ctx_ptr;
  (void)float_in_work;
  (void)int_work;
  const arena_offset out_off = sdsge_raw_model_data_output_arena_offset(
      ctx->n_states, ctx->n_shocks, ctx->n_observables);
  sdsge_raw_model_data_step(
      rep_idx, ctx->states_input, ctx->shocks_input, ctx->observables_input,
      ctx->states_batched, ctx->shocks_batched, ctx->observables_batched,
      ctx->n_states, ctx->n_shocks, ctx->n_observables, float_out,
      float_out + out_off.foffset[0], float_out + out_off.foffset[1]);
  return sdsge_mc_finish_status(SDSGE_OK, int_out);
}

int sdsge_mc_simulate_order1_runner(const i64 rep_idx,
                                    f64 *SDSGE_RESTRICT float_in_work,
                                    f64 *SDSGE_RESTRICT float_out,
                                    i64 *SDSGE_RESTRICT int_work,
                                    i64 *SDSGE_RESTRICT int_out,
                                    const void *ctx_ptr) {
  const sdsge_mc_simulate_order1_step_ctx *ctx = ctx_ptr;
  (void)int_work;
  if (ctx->shocks != NULL) {
    const arena_offset in_off =
        sdsge_simulate_order1_arena_offset(ctx->n, ctx->k, ctx->T, ctx->n_par);
    sdsge_mc_shock_draw(ctx->shocks, rep_idx,
                        float_in_work + ctx->shock_scratch_offset,
                        float_in_work + in_off.foffset[3]); // shock opens on x0
  }
  sdsge_simulate_order1_step(float_in_work, ctx->measurement, ctx->T, ctx->n,
                             ctx->k, ctx->n_par, ctx->m, float_out);
  return sdsge_mc_finish_status(SDSGE_OK, int_out);
}

int sdsge_mc_simulate_order2_runner(const i64 rep_idx,
                                    f64 *SDSGE_RESTRICT float_in_work,
                                    f64 *SDSGE_RESTRICT float_out,
                                    i64 *SDSGE_RESTRICT int_work,
                                    i64 *SDSGE_RESTRICT int_out,
                                    const void *ctx_ptr) {
  const sdsge_mc_simulate_order2_step_ctx *ctx = ctx_ptr;
  (void)int_work;
  if (ctx->shocks != NULL) {
    const arena_offset in_off = sdsge_simulate_order2_arena_offset(
        ctx->n_state, ctx->n_state + ctx->n_ctrl, ctx->n_exog, ctx->T,
        ctx->n_par);
    sdsge_mc_shock_draw(
        ctx->shocks, rep_idx, float_in_work + ctx->shock_scratch_offset,
        float_in_work + in_off.foffset[12]); // shock opens on x0
  }
  sdsge_simulate_order2_step(float_in_work, ctx->measurement, ctx->T,
                             ctx->n_state, ctx->n_ctrl, ctx->n_exog, ctx->n_par,
                             ctx->m, float_out);
  return sdsge_mc_finish_status(SDSGE_OK, int_out);
}

int sdsge_mc_filter_linear_runner(const i64 rep_idx,
                                  f64 *SDSGE_RESTRICT float_in_work,
                                  f64 *SDSGE_RESTRICT float_out,
                                  i64 *SDSGE_RESTRICT int_work,
                                  i64 *SDSGE_RESTRICT int_out,
                                  const void *ctx_ptr) {
  const sdsge_mc_filter_linear_step_ctx *ctx = ctx_ptr;
  const i64 input_size =
      sdsge_filter_linear_input_arena_size(ctx->n, ctx->m, ctx->k, ctx->T)
          .n_float;
  (void)rep_idx;
  (void)int_work;
  const int status = sdsge_filter_linear_step(
      float_in_work, float_in_work + input_size, ctx->T, ctx->n, ctx->m, ctx->k,
      ctx->symmetrize, ctx->joseph_cov, ctx->jitter, ctx->return_shocks,
      float_out);
  return sdsge_mc_finish_status(status, int_out);
}

int sdsge_mc_filter_extended_runner(const i64 rep_idx,
                                    f64 *SDSGE_RESTRICT float_in_work,
                                    f64 *SDSGE_RESTRICT float_out,
                                    i64 *SDSGE_RESTRICT int_work,
                                    i64 *SDSGE_RESTRICT int_out,
                                    const void *ctx_ptr) {
  const sdsge_mc_filter_extended_step_ctx *ctx = ctx_ptr;
  const i64 input_size = sdsge_filter_extended_input_arena_size(
                             ctx->n, ctx->m, ctx->k, ctx->T, ctx->n_par)
                             .n_float;
  (void)rep_idx;
  (void)int_work;
  const int status = sdsge_filter_extended_step(
      float_in_work, float_in_work + input_size, ctx->measurement,
      ctx->jacobian, ctx->T, ctx->n, ctx->m, ctx->k, ctx->n_par,
      ctx->symmetrize, ctx->joseph_cov, ctx->jitter, ctx->return_shocks,
      float_out);
  return sdsge_mc_finish_status(status, int_out);
}

int sdsge_mc_filter_unscented_runner(const i64 rep_idx,
                                     f64 *SDSGE_RESTRICT float_in_work,
                                     f64 *SDSGE_RESTRICT float_out,
                                     i64 *SDSGE_RESTRICT int_work,
                                     i64 *SDSGE_RESTRICT int_out,
                                     const void *ctx_ptr) {
  const sdsge_mc_filter_unscented_step_ctx *ctx = ctx_ptr;
  const i64 input_size = sdsge_filter_unscented_input_arena_size(
                             ctx->n_state, ctx->n_ctrl, ctx->n_exog, ctx->n_obs,
                             ctx->T, ctx->n_par)
                             .n_float;
  (void)rep_idx;
  (void)int_work;
  const int status = (int)sdsge_filter_unscented_step(
      float_in_work, float_in_work + input_size, ctx->measurement, ctx->T,
      ctx->n_state, ctx->n_ctrl, ctx->n_exog, ctx->n_obs, ctx->n_par,
      ctx->alpha, ctx->beta, ctx->kappa, ctx->symmetrize, ctx->jitter,
      float_out);
  return sdsge_mc_finish_status(status, int_out);
}

void sdsge_add_payload_step(const f64 *SDSGE_RESTRICT input, const i64 n,
                            const int input_batched, const i64 rep_idx,
                            f64 *SDSGE_RESTRICT output) {
  if (n > 0)
    memcpy(output, input + (input_batched ? rep_idx * n : 0),
           (size_t)n * sizeof(f64));
}

void sdsge_passthrough_step(const f64 *SDSGE_RESTRICT input, const i64 n,
                            const i64 p, f64 *SDSGE_RESTRICT output) {
  if (n > 0 && p > 0)
    memcpy(output, input, (size_t)(n * p) * sizeof(f64));
}

void sdsge_raw_model_data_step(
    const i64 rep_idx, const f64 *SDSGE_RESTRICT states_input,
    const f64 *SDSGE_RESTRICT shocks_input,
    const f64 *SDSGE_RESTRICT observables_input, const int states_batched,
    const int shocks_batched, const int observables_batched, const i64 n_states,
    const i64 n_shocks, const i64 n_observables,
    f64 *SDSGE_RESTRICT states_output, f64 *SDSGE_RESTRICT shocks_output,
    f64 *SDSGE_RESTRICT observables_output) {
  if (n_states > 0) {
    memcpy(states_output,
           states_input + (states_batched ? rep_idx * n_states : 0),
           (size_t)n_states * sizeof(f64));
  }
  if (n_shocks > 0) {
    memcpy(shocks_output,
           shocks_input + (shocks_batched ? rep_idx * n_shocks : 0),
           (size_t)n_shocks * sizeof(f64));
  }
  if (n_observables > 0) {
    memcpy(observables_output,
           observables_input +
               (observables_batched ? rep_idx * n_observables : 0),
           (size_t)n_observables * sizeof(f64));
  }
}

void sdsge_simulate_order1_step(f64 *SDSGE_RESTRICT arena,
                                sdsge_measurement_fn measurement, const i64 T,
                                const i64 n, const i64 k, const i64 n_par,
                                const i64 m, f64 *SDSGE_RESTRICT simout) {
  const arena_offset in_off =
      sdsge_simulate_order1_arena_offset(n, k, T, n_par);
  const f64 *SDSGE_RESTRICT A = arena;
  const f64 *SDSGE_RESTRICT B = arena + in_off.foffset[0];
  const f64 *SDSGE_RESTRICT steady_state = arena + in_off.foffset[1];
  const f64 *SDSGE_RESTRICT x0 = arena + in_off.foffset[2];
  const f64 *SDSGE_RESTRICT shock = arena + in_off.foffset[3];
  f64 *SDSGE_RESTRICT params = arena + in_off.foffset[4];
  f64 *SDSGE_RESTRICT scratch = arena + in_off.foffset[5];

  const arena_offset out_off =
      sdsge_simulate_order1_output_arena_offset(n, k, T, m);
  f64 *SDSGE_RESTRICT states = simout;
  f64 *SDSGE_RESTRICT shock_out = simout + out_off.foffset[0];
  f64 *SDSGE_RESTRICT observables = simout + out_off.foffset[1];

  sdsge_simulate_linear_states(A, B, x0, shock, steady_state, states, scratch,
                               T, n, k);

  memcpy(shock_out, shock, (size_t)(T * k) * sizeof(f64));
  if (m > 0) {
    for (i64 t = 0; t < T; ++t) {
      measurement(states + t * n, params, observables + t * m);
    }
  }
}

void sdsge_simulate_order2_step(f64 *SDSGE_RESTRICT arena,
                                sdsge_measurement_fn measurement, i64 T, i64 nx,
                                i64 ny, i64 n_exog, i64 n_par, i64 m,
                                f64 *SDSGE_RESTRICT simout) {
  const i64 n = nx + ny;
  const arena_offset in_off =
      sdsge_simulate_order2_arena_offset(nx, n, n_exog, T, n_par);
  const f64 *SDSGE_RESTRICT hx = arena;
  const f64 *SDSGE_RESTRICT gx = arena + in_off.foffset[0];
  const f64 *SDSGE_RESTRICT bu = arena + in_off.foffset[1];
  const f64 *SDSGE_RESTRICT hxx = arena + in_off.foffset[2];
  const f64 *SDSGE_RESTRICT gxx = arena + in_off.foffset[3];
  const f64 *SDSGE_RESTRICT hxu = arena + in_off.foffset[4];
  const f64 *SDSGE_RESTRICT gxu = arena + in_off.foffset[5];
  const f64 *SDSGE_RESTRICT huu = arena + in_off.foffset[6];
  const f64 *SDSGE_RESTRICT guu = arena + in_off.foffset[7];
  const f64 *SDSGE_RESTRICT hss = arena + in_off.foffset[8];
  const f64 *SDSGE_RESTRICT gss = arena + in_off.foffset[9];
  const f64 *SDSGE_RESTRICT steady_state = arena + in_off.foffset[10];
  const f64 *SDSGE_RESTRICT x0 = arena + in_off.foffset[11];
  const f64 *SDSGE_RESTRICT shock = arena + in_off.foffset[12];
  f64 *SDSGE_RESTRICT params = arena + in_off.foffset[13];
  f64 *SDSGE_RESTRICT scratch = arena + in_off.foffset[14];

  const arena_offset out_off =
      sdsge_simulate_order2_output_arena_offset(n, n_exog, T, m);
  f64 *SDSGE_RESTRICT states = simout;
  f64 *SDSGE_RESTRICT shock_out = simout + out_off.foffset[0];
  f64 *SDSGE_RESTRICT observables = simout + out_off.foffset[1];

  sdsge_simulate_second_order_pruned(hx, gx, bu, hxx, gxx, hxu, gxu, huu, guu,
                                     hss, gss, x0, shock, steady_state, states,
                                     scratch, T, nx, ny, n_exog);

  memcpy(shock_out, shock, (size_t)(T * n_exog) * sizeof(f64));
  if (m > 0) {
    for (i64 t = 0; t < T; ++t) {
      measurement(states + t * n, params, observables + t * m);
    }
  }
}

int sdsge_filter_linear_step(const f64 *SDSGE_RESTRICT input_arena,
                             f64 *SDSGE_RESTRICT scratch_arena, const i64 T,
                             const i64 n, const i64 m, const i64 k,
                             const int symmetrize, const int joseph_cov,
                             const f64 jitter, const int return_shocks,
                             f64 *SDSGE_RESTRICT output_arena) {
  const arena_offset in_off =
      sdsge_filter_linear_input_arena_offset(n, m, k, T);
  const f64 *A = input_arena;
  const f64 *B = input_arena + in_off.foffset[0];
  const f64 *C = input_arena + in_off.foffset[1];
  const f64 *d = input_arena + in_off.foffset[2];
  const f64 *Q = input_arena + in_off.foffset[3];
  const f64 *R = input_arena + in_off.foffset[4];
  const f64 *y = input_arena + in_off.foffset[5];
  const f64 *x0 = input_arena + in_off.foffset[6];
  const f64 *P0 = input_arena + in_off.foffset[7];
  const arena_offset out_off =
      sdsge_filter_linear_output_arena_offset(n, m, k, T, return_shocks);
  kf_outputs out = {.x_pred = output_arena};
  out.x_filt = output_arena + out_off.foffset[0];
  out.P_pred = output_arena + out_off.foffset[1];
  out.P_filt = output_arena + out_off.foffset[2];
  out.y_pred = output_arena + out_off.foffset[3];
  out.y_filt = output_arena + out_off.foffset[4];
  out.innov = output_arena + out_off.foffset[5];
  out.std_innov = output_arena + out_off.foffset[6];
  out.S = output_arena + out_off.foffset[7];
  out.eps_hat = return_shocks ? output_arena + out_off.foffset[8] : NULL;
  out.loglik = output_arena + out_off.foffset[9];

  const kf_inputs in = {.n = n,
                        .m = m,
                        .k = k,
                        .T = T,
                        .A = A,
                        .B = B,
                        .C = C,
                        .d = d,
                        .Q = Q,
                        .R = R,
                        .y = y,
                        .x0 = x0,
                        .P0 = P0,
                        .symmetrize = symmetrize,
                        .joseph_cov = joseph_cov,
                        .jitter = jitter,
                        .return_shocks = return_shocks,
                        .store_history = 1};
  return kf_hot_loop(&in, scratch_arena, &out);
}

int sdsge_filter_extended_step(const f64 *SDSGE_RESTRICT input_arena,
                               f64 *SDSGE_RESTRICT scratch_arena,
                               const meas_fn meas, const meas_fn jac,
                               const i64 T, const i64 n, const i64 m,
                               const i64 k, const i64 n_par,
                               const int symmetrize, const int joseph_cov,
                               const f64 jitter, const int return_shocks,
                               f64 *SDSGE_RESTRICT output_arena) {
  const arena_offset in_off =
      sdsge_filter_extended_input_arena_offset(n, m, k, T, n_par);
  const f64 *A = input_arena;
  const f64 *B = input_arena + in_off.foffset[0];
  const f64 *params = input_arena + in_off.foffset[1];
  const f64 *Q = input_arena + in_off.foffset[2];
  const f64 *R = input_arena + in_off.foffset[3];
  const f64 *y = input_arena + in_off.foffset[4];
  const f64 *x0 = input_arena + in_off.foffset[5];
  const f64 *P0 = input_arena + in_off.foffset[6];
  const arena_offset out_off =
      sdsge_filter_extended_output_arena_offset(n, m, k, T, return_shocks);
  ekf_outputs out = {.x_pred = output_arena};
  out.x_filt = output_arena + out_off.foffset[0];
  out.P_pred = output_arena + out_off.foffset[1];
  out.P_filt = output_arena + out_off.foffset[2];
  out.y_pred = output_arena + out_off.foffset[3];
  out.y_filt = output_arena + out_off.foffset[4];
  out.innov = output_arena + out_off.foffset[5];
  out.std_innov = output_arena + out_off.foffset[6];
  out.S = output_arena + out_off.foffset[7];
  out.eps_hat = return_shocks ? output_arena + out_off.foffset[8] : NULL;
  out.loglik = output_arena + out_off.foffset[9];

  const ekf_inputs in = {.meas = meas,
                         .jac = jac,
                         .A = A,
                         .B = B,
                         .calib_params = params,
                         .Q = Q,
                         .R = R,
                         .y = y,
                         .x0 = x0,
                         .P0 = P0,
                         .T = T,
                         .n = n,
                         .m = m,
                         .k = k,
                         .n_par = n_par,
                         .jitter = jitter,
                         .symmetrize = symmetrize,
                         .joseph_cov = joseph_cov,
                         .compute_y_filt = 1,
                         .return_shocks = return_shocks,
                         .store_history = 1};
  return ekf_hot_loop(&in, scratch_arena, &out);
}

i64 sdsge_filter_unscented_step(
    const f64 *SDSGE_RESTRICT input_arena, f64 *SDSGE_RESTRICT scratch_arena,
    const meas_fn meas, const i64 T, const i64 n_state, const i64 n_ctrl,
    const i64 n_exog, const i64 n_obs, const i64 n_par, const f64 alpha,
    const f64 beta, const f64 kappa, const int symmetrize, const f64 jitter,
    f64 *SDSGE_RESTRICT output_arena) {
  const arena_offset in_off = sdsge_filter_unscented_input_arena_offset(
      n_state, n_ctrl, n_exog, n_obs, T, n_par);
  const f64 *hx = input_arena;
  const f64 *gx = input_arena + in_off.foffset[0];
  const f64 *bu = input_arena + in_off.foffset[1];
  const f64 *hxx = input_arena + in_off.foffset[2];
  const f64 *gxx = input_arena + in_off.foffset[3];
  const f64 *hxu = input_arena + in_off.foffset[4];
  const f64 *gxu = input_arena + in_off.foffset[5];
  const f64 *huu = input_arena + in_off.foffset[6];
  const f64 *guu = input_arena + in_off.foffset[7];
  const f64 *hss = input_arena + in_off.foffset[8];
  const f64 *gss = input_arena + in_off.foffset[9];
  const f64 *steady_state = input_arena + in_off.foffset[10];
  const f64 *params = input_arena + in_off.foffset[11];
  const f64 *Q = input_arena + in_off.foffset[12];
  const f64 *R = input_arena + in_off.foffset[13];
  const f64 *obs = input_arena + in_off.foffset[14];
  const f64 *z0 = input_arena + in_off.foffset[15];
  const f64 *P0 = input_arena + in_off.foffset[16];

  const arena_offset out_off =
      sdsge_filter_unscented_output_arena_offset(n_state, n_ctrl, n_obs, T);
  ukf_outputs out = {.x_pred = output_arena};
  out.x_filt = output_arena + out_off.foffset[0];
  out.P_pred = output_arena + out_off.foffset[1];
  out.P_filt = output_arena + out_off.foffset[2];
  out.y_pred = output_arena + out_off.foffset[3];
  out.y_filt = output_arena + out_off.foffset[4];
  out.innov = output_arena + out_off.foffset[5];
  out.std_innov = output_arena + out_off.foffset[6];
  out.S = output_arena + out_off.foffset[7];
  out.loglik = output_arena + out_off.foffset[8];
  out.x1_pred = output_arena + out_off.foffset[9];
  out.x2_pred = output_arena + out_off.foffset[10];
  out.x1_filt = output_arena + out_off.foffset[11];
  out.x2_filt = output_arena + out_off.foffset[12];

  const ukf_inputs in = {.meas = meas,
                         .hx = hx,
                         .gx = gx,
                         .bu = bu,
                         .hxx = hxx,
                         .gxx = gxx,
                         .hxu = hxu,
                         .gxu = gxu,
                         .huu = huu,
                         .guu = guu,
                         .hss = hss,
                         .gss = gss,
                         .steady_state = steady_state,
                         .params = params,
                         .Q = Q,
                         .R = R,
                         .obs = obs,
                         .z0 = z0,
                         .P0 = P0,
                         .T = T,
                         .n_state = n_state,
                         .n_ctrl = n_ctrl,
                         .n_exog = n_exog,
                         .n_obs = n_obs,
                         .n_params = n_par,
                         .alpha = alpha,
                         .beta = beta,
                         .kappa = kappa,
                         .jitter = jitter,
                         .symmetrize = symmetrize,
                         .store_history = 1};
  return ukf_hot_loop(&in, scratch_arena, &out);
}
