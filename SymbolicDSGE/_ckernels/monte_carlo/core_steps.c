#include "core_steps.h"
#include "../core/core.h"
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

int sdsge_mc_raw_model_data_runner(
    const i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx_ptr) {
  const sdsge_mc_raw_model_data_step_ctx *ctx = ctx_ptr;
  (void)float_in_work;
  (void)int_work;
  sdsge_raw_model_data_step(
      ctx->states_input, ctx->n_states, ctx->states_batched, float_out,
      ctx->observables_input, ctx->n_observables, ctx->observables_batched,
      rep_idx, float_out + ctx->n_states);
  return sdsge_mc_finish_status(SDSGE_OK, int_out);
}

/* Where the (T, k) shock block sits in each simulation input arena. Derived
 * from the layouts documented in core_steps.h so the offset stays with the
 * layout it describes rather than being recomputed by the caller. */
static inline i64 sdsge_simulate_order1_shock_offset(const i64 n, const i64 k) {
  return n * n + n * k + n;
}

static inline i64 sdsge_simulate_order2_shock_offset(const i64 nx, const i64 ny,
                                                     const i64 n_exog) {
  return nx * nx + ny * nx + nx * n_exog + nx * nx * nx + ny * nx * nx + nx +
         ny + (nx + ny) + nx;
}

int sdsge_mc_simulate_order1_runner(
    const i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx_ptr) {
  const sdsge_mc_simulate_order1_step_ctx *ctx = ctx_ptr;
  (void)int_work;
  if (ctx->shocks != NULL) {
    sdsge_mc_shock_draw(
        ctx->shocks, rep_idx, float_in_work + ctx->shock_scratch_offset,
        float_in_work + sdsge_simulate_order1_shock_offset(ctx->n, ctx->k));
  }
  sdsge_simulate_order1_step(float_in_work, ctx->measurement, ctx->T, ctx->n,
                             ctx->k, ctx->n_par, ctx->m, float_out);
  return sdsge_mc_finish_status(SDSGE_OK, int_out);
}

int sdsge_mc_simulate_order2_runner(
    const i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx_ptr) {
  const sdsge_mc_simulate_order2_step_ctx *ctx = ctx_ptr;
  (void)int_work;
  if (ctx->shocks != NULL) {
    sdsge_mc_shock_draw(ctx->shocks, rep_idx,
                        float_in_work + ctx->shock_scratch_offset,
                        float_in_work + sdsge_simulate_order2_shock_offset(
                                            ctx->n_state, ctx->n_ctrl,
                                            ctx->n_exog));
  }
  sdsge_simulate_order2_step(float_in_work, ctx->measurement, ctx->T,
                             ctx->n_state, ctx->n_ctrl, ctx->n_exog,
                             ctx->n_par, ctx->m, float_out);
  return sdsge_mc_finish_status(SDSGE_OK, int_out);
}

int sdsge_mc_filter_linear_runner(
    const i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx_ptr) {
  const sdsge_mc_filter_linear_step_ctx *ctx = ctx_ptr;
  const i64 input_size =
      sdsge_filter_linear_input_arena_size(ctx->n, ctx->m, ctx->k, ctx->T);
  (void)rep_idx;
  (void)int_work;
  const int status = sdsge_filter_linear_step(
      float_in_work, float_in_work + input_size, ctx->T, ctx->n, ctx->m,
      ctx->k, ctx->symmetrize, ctx->jitter, ctx->return_shocks, float_out);
  return sdsge_mc_finish_status(status, int_out);
}

int sdsge_mc_filter_extended_runner(
    const i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx_ptr) {
  const sdsge_mc_filter_extended_step_ctx *ctx = ctx_ptr;
  const i64 input_size = sdsge_filter_extended_input_arena_size(
      ctx->n, ctx->m, ctx->k, ctx->T, ctx->n_par);
  (void)rep_idx;
  (void)int_work;
  const int status = sdsge_filter_extended_step(
      float_in_work, float_in_work + input_size, ctx->measurement,
      ctx->jacobian, ctx->T, ctx->n, ctx->m, ctx->k, ctx->n_par,
      ctx->symmetrize, ctx->jitter, ctx->return_shocks, float_out);
  return sdsge_mc_finish_status(status, int_out);
}

int sdsge_mc_filter_unscented_runner(
    const i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx_ptr) {
  const sdsge_mc_filter_unscented_step_ctx *ctx = ctx_ptr;
  const i64 input_size = sdsge_filter_unscented_input_arena_size(
      ctx->n_state, ctx->n_ctrl, ctx->n_exog, ctx->n_obs, ctx->T,
      ctx->n_par);
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

void sdsge_raw_model_data_step(
    const f64 *SDSGE_RESTRICT states_input, const i64 n_states,
    const int states_batched,
    f64 *SDSGE_RESTRICT states_output,
    const f64 *SDSGE_RESTRICT observables_input, const i64 n_observables,
    const int observables_batched, const i64 rep_idx,
    f64 *SDSGE_RESTRICT observables_output) {
  if (n_states > 0)
    memcpy(states_output,
           states_input + (states_batched ? rep_idx * n_states : 0),
           (size_t)n_states * sizeof(f64));
  if (n_observables > 0)
    memcpy(observables_output,
           observables_input +
               (observables_batched ? rep_idx * n_observables : 0),
           (size_t)n_observables * sizeof(f64));
}

i64 sdsge_simulate_order1_arena_size(const i64 n, const i64 k, const i64 T,
                                     const i64 n_par) {
  return n * n + n * k + n + T * k + n_par;
}
void sdsge_simulate_order1_step(f64 *SDSGE_RESTRICT arena,
                                sdsge_measurement_fn measurement, const i64 T,
                                const i64 n, const i64 k, const i64 n_par,
                                const i64 m, f64 *SDSGE_RESTRICT simout) {
  const f64 *SDSGE_RESTRICT A = arena;
  const f64 *SDSGE_RESTRICT B = A + n * n;
  const f64 *SDSGE_RESTRICT x0 = B + n * k;
  const f64 *SDSGE_RESTRICT shock = x0 + n;
  f64 *SDSGE_RESTRICT params = (f64 *)(shock + T * k);
  (void)n_par;
  f64 *states = simout;
  f64 *observables = simout + T * n;

  for (i64 t = 0; t < T; ++t) {
    const f64 *xt = t == 0 ? x0 : states + (t - 1) * n;
    const f64 *shock_t = shock + t * k;
    f64 *state_t = states + t * n;
    f64 *observable_t = observables + t * m;

    for (i64 i = 0; i < n; ++i) {
      const f64 *Ai = A + i * n;
      const f64 *Bi = B + i * k;
      f64 value = 0.0;
      for (i64 j = 0; j < n; ++j)
        value += Ai[j] * xt[j];
      for (i64 j = 0; j < k; ++j)
        value += Bi[j] * shock_t[j];
      state_t[i] = value;
    }

    if (m > 0) {
      measurement(state_t, params, observable_t);
    }
  }
}

i64 sdsge_simulate_order2_arena_size(const i64 n_state, const i64 n_var,
                                     const i64 n_exog, const i64 T,
                                     const i64 n_par) {
  i64 nx = n_state;
  i64 ny = n_var - n_state;
  return nx * nx + ny * nx + nx * n_exog + nx * nx * nx + ny * nx * nx + nx +
         ny + (nx + ny) + nx + T * n_exog + n_par + 4 * nx + nx * nx;
}
void sdsge_simulate_order2_step(f64 *SDSGE_RESTRICT arena,
                                sdsge_measurement_fn measurement, i64 T, i64 nx,
                                i64 ny, i64 n_exog, i64 n_par, i64 m,
                                f64 *SDSGE_RESTRICT simout) {
  const i64 n = nx + ny;
  const f64 *SDSGE_RESTRICT hx = arena;
  const f64 *SDSGE_RESTRICT gx = hx + nx * nx;
  const f64 *SDSGE_RESTRICT bx = gx + ny * nx;
  const f64 *SDSGE_RESTRICT hxx = bx + nx * n_exog;
  const f64 *SDSGE_RESTRICT gxx = hxx + nx * nx * nx;
  const f64 *SDSGE_RESTRICT hss = gxx + ny * nx * nx;
  const f64 *SDSGE_RESTRICT gss = hss + nx;
  const f64 *SDSGE_RESTRICT steady_state = gss + ny;
  const f64 *SDSGE_RESTRICT x0 = steady_state + n;
  const f64 *SDSGE_RESTRICT shock = x0 + nx;
  f64 *SDSGE_RESTRICT params = (f64 *)(shock + T * n_exog);
  f64 *SDSGE_RESTRICT scratch = params + n_par;
  f64 *SDSGE_RESTRICT states = simout;
  f64 *SDSGE_RESTRICT observables = simout + T * n;
  f64 *SDSGE_RESTRICT x1_cur = scratch;
  f64 *SDSGE_RESTRICT x1_next = scratch + nx;
  f64 *SDSGE_RESTRICT x2_cur = scratch + 2 * nx;
  f64 *SDSGE_RESTRICT x2_next = scratch + 3 * nx;
  f64 *SDSGE_RESTRICT x1_outer = scratch + 4 * nx;

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

    f64 *SDSGE_RESTRICT state_t = states + t * n;
    for (i64 i = 0; i < nx; ++i) {
      state_t[i] = x1_cur[i] + x2_cur[i];
    }

    if (ny > 0) {
      for (i64 j = 0; j < nx; ++j) {
        const f64 xj = x1_cur[j];
        f64 *SDSGE_RESTRICT row = x1_outer + j * nx;
        for (i64 k = 0; k < nx; ++k) {
          row[k] = xj * x1_cur[k];
        }
      }

      for (i64 i = 0; i < ny; ++i) {
        const f64 *SDSGE_RESTRICT gxi = gx + i * nx;
        const f64 *SDSGE_RESTRICT gxxi = gxx + i * nx * nx;
        f64 value = 0.5 * gss[i];

        for (i64 j = 0; j < nx; ++j) {
          value += gxi[j] * state_t[j];
        }
        for (i64 j = 0; j < nx; ++j) {
          const f64 *SDSGE_RESTRICT gxxij = gxxi + j * nx;
          const f64 *SDSGE_RESTRICT outerj = x1_outer + j * nx;
          for (i64 k = 0; k < nx; ++k) {
            value += 0.5 * gxxij[k] * outerj[k];
          }
        }
        state_t[nx + i] = value;
      }
    }

    for (i64 i = 0; i < n; ++i) {
      state_t[i] += steady_state[i];
    }
    if (m > 0) {
      measurement(state_t, params, observables + t * m);
    }
  }
}

i64 sdsge_filter_linear_input_arena_size(const i64 n, const i64 m,
                                          const i64 k, const i64 T) {
  return n * n + n * k + m * n + m + k * k + m * m + T * m + n + n * n;
}

i64 sdsge_filter_linear_output_arena_size(const i64 n, const i64 m,
                                           const i64 k, const i64 T,
                                           const int return_shocks) {
  return 2 * T * n + 2 * T * n * n + 4 * T * m + T * m * m +
         (return_shocks ? T * k : 0) + 1;
}

int sdsge_filter_linear_step(const f64 *SDSGE_RESTRICT input_arena,
                             f64 *SDSGE_RESTRICT scratch_arena, const i64 T,
                             const i64 n, const i64 m, const i64 k,
                             const int symmetrize, const f64 jitter,
                             const int return_shocks,
                             f64 *SDSGE_RESTRICT output_arena) {
  const f64 *A = input_arena;
  const f64 *B = A + n * n;
  const f64 *C = B + n * k;
  const f64 *d = C + m * n;
  const f64 *Q = d + m;
  const f64 *R = Q + k * k;
  const f64 *y = R + m * m;
  const f64 *x0 = y + T * m;
  const f64 *P0 = x0 + n;
  f64 *p = output_arena;
  kf_outputs out = {.x_pred = p};
  p += T * n;
  out.x_filt = p;
  p += T * n;
  out.P_pred = p;
  p += T * n * n;
  out.P_filt = p;
  p += T * n * n;
  out.y_pred = p;
  p += T * m;
  out.y_filt = p;
  p += T * m;
  out.innov = p;
  p += T * m;
  out.std_innov = p;
  p += T * m;
  out.S = p;
  p += T * m * m;
  out.eps_hat = return_shocks ? p : NULL;
  p += return_shocks ? T * k : 0;
  out.loglik = p;

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
                        .jitter = jitter,
                        .return_shocks = return_shocks,
                        .store_history = 1};
  return kf_hot_loop(&in, scratch_arena, &out);
}

i64 sdsge_filter_extended_input_arena_size(const i64 n, const i64 m,
                                            const i64 k, const i64 T,
                                            const i64 n_par) {
  return n * n + n * k + n_par + k * k + m * m + T * m + n + n * n;
}

i64 sdsge_filter_extended_output_arena_size(const i64 n, const i64 m,
                                             const i64 k, const i64 T,
                                             const int return_shocks) {
  return sdsge_filter_linear_output_arena_size(n, m, k, T, return_shocks);
}

int sdsge_filter_extended_step(const f64 *SDSGE_RESTRICT input_arena,
                               f64 *SDSGE_RESTRICT scratch_arena,
                               const meas_fn meas, const meas_fn jac,
                               const i64 T, const i64 n, const i64 m,
                               const i64 k, const i64 n_par,
                               const int symmetrize, const f64 jitter,
                               const int return_shocks,
                               f64 *SDSGE_RESTRICT output_arena) {
  const f64 *A = input_arena;
  const f64 *B = A + n * n;
  const f64 *params = B + n * k;
  const f64 *Q = params + n_par;
  const f64 *R = Q + k * k;
  const f64 *y = R + m * m;
  const f64 *x0 = y + T * m;
  const f64 *P0 = x0 + n;
  f64 *p = output_arena;
  ekf_outputs out = {.x_pred = p};
  p += T * n;
  out.x_filt = p;
  p += T * n;
  out.P_pred = p;
  p += T * n * n;
  out.P_filt = p;
  p += T * n * n;
  out.y_pred = p;
  p += T * m;
  out.y_filt = p;
  p += T * m;
  out.innov = p;
  p += T * m;
  out.std_innov = p;
  p += T * m;
  out.S = p;
  p += T * m * m;
  out.eps_hat = return_shocks ? p : NULL;
  p += return_shocks ? T * k : 0;
  out.loglik = p;

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
                         .compute_y_filt = 1,
                         .return_shocks = return_shocks,
                         .store_history = 1};
  return ekf_hot_loop(&in, scratch_arena, &out);
}

i64 sdsge_filter_unscented_input_arena_size(const i64 n_state,
                                             const i64 n_ctrl,
                                             const i64 n_exog,
                                             const i64 n_obs, const i64 T,
                                             const i64 n_par) {
  const i64 nz = 2 * n_state;
  return n_state * n_state + n_ctrl * n_state + n_state * n_exog +
         n_state * n_state * n_state + n_ctrl * n_state * n_state + n_state +
         n_ctrl + n_state + n_ctrl + n_par + n_exog * n_exog + n_obs * n_obs +
         T * n_obs + nz + nz * nz;
}

i64 sdsge_filter_unscented_output_arena_size(const i64 n_state,
                                              const i64 n_ctrl,
                                              const i64 n_obs, const i64 T) {
  const i64 n_var = n_state + n_ctrl;
  const i64 nz = 2 * n_state;
  return 2 * T * n_var + 2 * T * nz * nz + 4 * T * n_obs + T * n_obs * n_obs +
         4 * T * n_state + 1;
}

i64 sdsge_filter_unscented_step(const f64 *SDSGE_RESTRICT input_arena,
                                f64 *SDSGE_RESTRICT scratch_arena,
                                const meas_fn meas, const i64 T,
                                const i64 n_state, const i64 n_ctrl,
                                const i64 n_exog, const i64 n_obs,
                                const i64 n_par, const f64 alpha,
                                const f64 beta, const f64 kappa,
                                const int symmetrize, const f64 jitter,
                                f64 *SDSGE_RESTRICT output_arena) {
  const i64 n_var = n_state + n_ctrl;
  const i64 nz = 2 * n_state;
  const f64 *hx = input_arena;
  const f64 *gx = hx + n_state * n_state;
  const f64 *bx = gx + n_ctrl * n_state;
  const f64 *hxx = bx + n_state * n_exog;
  const f64 *gxx = hxx + n_state * n_state * n_state;
  const f64 *hss = gxx + n_ctrl * n_state * n_state;
  const f64 *gss = hss + n_state;
  const f64 *steady_state = gss + n_ctrl;
  const f64 *params = steady_state + n_var;
  const f64 *Q = params + n_par;
  const f64 *R = Q + n_exog * n_exog;
  const f64 *obs = R + n_obs * n_obs;
  const f64 *z0 = obs + T * n_obs;
  const f64 *P0 = z0 + nz;
  f64 *p = output_arena;
  ukf_outputs out = {.x_pred = p};
  p += T * n_var;
  out.x_filt = p;
  p += T * n_var;
  out.P_pred = p;
  p += T * nz * nz;
  out.P_filt = p;
  p += T * nz * nz;
  out.y_pred = p;
  p += T * n_obs;
  out.y_filt = p;
  p += T * n_obs;
  out.innov = p;
  p += T * n_obs;
  out.std_innov = p;
  p += T * n_obs;
  out.S = p;
  p += T * n_obs * n_obs;
  out.loglik = p;
  p += 1;
  out.x1_pred = p;
  p += T * n_state;
  out.x2_pred = p;
  p += T * n_state;
  out.x1_filt = p;
  p += T * n_state;
  out.x2_filt = p;

  const ukf_inputs in = {.meas = meas,
                         .hx = hx,
                         .gx = gx,
                         .bx = bx,
                         .hxx = hxx,
                         .gxx = gxx,
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
