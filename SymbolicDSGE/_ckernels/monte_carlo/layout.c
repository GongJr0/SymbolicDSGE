/* Arena sizes and offsets for every Monte Carlo step kind.
 *
 * Collected out of the step files so a layout is described in one place and the
 * kernels beside them only consume it.
 *
 * A kind places its buffers in an offset function, each entry the one before it
 * plus the width of the buffer it closes. Its sizer reads the total off the
 * last entry, so the two cannot describe different layouts.
 */

#include "layout.h"

#include "../core/core.h"
#include "../kalman/kalman.h"
#include "core_steps.h"
#include "regression.h"
#include "transforms.h"

/* from core_steps.c */
arena_offset sdsge_passthrough_arena_offset(const i64 n, const i64 p) {
  arena_offset off = make_offset(1, 0);
  off.foffset[0] = n * p; // payload(n, p)
  return off;
}

arena_size sdsge_passthrough_arena_size(const i64 n, const i64 p) {
  const arena_offset off = sdsge_passthrough_arena_offset(n, p);
  return make_sizer(off.foffset[off.n_fbuf - 1], 0);
}

arena_offset sdsge_raw_model_data_output_arena_offset(const i64 n_states,
                                                      const i64 n_shocks,
                                                      const i64 n_observables) {
  arena_offset off = make_offset(3, 0);
  off.foffset[0] = n_states;                       // states
  off.foffset[1] = off.foffset[0] + n_shocks;      // shocks
  off.foffset[2] = off.foffset[1] + n_observables; // observables
  return off;
}

arena_size sdsge_raw_model_data_output_arena_size(const i64 n_states,
                                                  const i64 n_shocks,
                                                  const i64 n_observables) {
  const arena_offset off = sdsge_raw_model_data_output_arena_offset(
      n_states, n_shocks, n_observables);
  return make_sizer(off.foffset[off.n_fbuf - 1], 0);
}

arena_offset sdsge_simulate_order1_arena_offset(const i64 n, const i64 k,
                                                const i64 T, const i64 n_par) {
  arena_offset off = make_offset(7, 0);
  off.foffset[0] = n * n;                  // A(n, n)
  off.foffset[1] = off.foffset[0] + n * k; // B(n, k)
  off.foffset[2] = off.foffset[1] + n;     // steady_state(n)
  off.foffset[3] = off.foffset[2] + n;     // x0(n)
  off.foffset[4] = off.foffset[3] + T * k; // shock(T, k)
  off.foffset[5] = off.foffset[4] + n_par; // params(n_par)
  off.foffset[6] =
      off.foffset[5] +
      sdsge_simulate_linear_states_arena_size(n).n_float; // scratch
  return off;
}

arena_size sdsge_simulate_order1_arena_size(const i64 n, const i64 k,
                                            const i64 T, const i64 n_par) {
  const arena_offset off = sdsge_simulate_order1_arena_offset(n, k, T, n_par);
  return make_sizer(off.foffset[off.n_fbuf - 1], 0);
}

arena_offset sdsge_simulate_order1_output_arena_offset(const i64 n, const i64 k,
                                                       const i64 T,
                                                       const i64 m) {
  arena_offset off = make_offset(3, 0);
  off.foffset[0] = T * n;                  // states(T, n)
  off.foffset[1] = off.foffset[0] + T * k; // shocks(T, k)
  off.foffset[2] =
      off.foffset[1] + T * m; // observables(T, m), zero when unbuilt
  return off;
}

arena_size sdsge_simulate_order1_output_arena_size(const i64 n, const i64 k,
                                                   const i64 T, const i64 m) {
  const arena_offset off =
      sdsge_simulate_order1_output_arena_offset(n, k, T, m);
  return make_sizer(off.foffset[off.n_fbuf - 1], 0);
}

arena_offset sdsge_simulate_order2_arena_offset(const i64 n_state,
                                                const i64 n_var,
                                                const i64 n_exog, const i64 T,
                                                const i64 n_par) {
  arena_offset off = make_offset(16, 0);
  const i64 nx = n_state;
  const i64 ny = n_var - n_state;
  off.foffset[0] = nx * nx;                           // hx(nx, nx)
  off.foffset[1] = off.foffset[0] + ny * nx;          // gx(ny, nx)
  off.foffset[2] = off.foffset[1] + n_var * n_exog;   // bu(n_var, n_exog)
  off.foffset[3] = off.foffset[2] + nx * nx * nx;     // hxx(nx, nx, nx)
  off.foffset[4] = off.foffset[3] + ny * nx * nx;     // gxx(ny, nx, nx)
  off.foffset[5] = off.foffset[4] + nx * nx * n_exog; // hxu(nx, nx, n_exog)
  off.foffset[6] = off.foffset[5] + ny * nx * n_exog; // gxu(ny, nx, n_exog)
  off.foffset[7] =
      off.foffset[6] + nx * n_exog * n_exog; // huu(nx, n_exog, n_exog)
  off.foffset[8] =
      off.foffset[7] + ny * n_exog * n_exog;      // guu(ny, n_exog, n_exog)
  off.foffset[9] = off.foffset[8] + nx;           // hss(nx)
  off.foffset[10] = off.foffset[9] + ny;          // gss(ny)
  off.foffset[11] = off.foffset[10] + n_var;      // steady_state(n_var)
  off.foffset[12] = off.foffset[11] + nx;         // x0(nx)
  off.foffset[13] = off.foffset[12] + T * n_exog; // shock(T, n_exog)
  off.foffset[14] = off.foffset[13] + n_par;      // params(n_par)
  off.foffset[15] = off.foffset[14] +
                    sdsge_simulate_second_order_pruned_arena_size(nx, n_exog)
                        .n_float; // scratch
  return off;
}

arena_size sdsge_simulate_order2_arena_size(const i64 n_state, const i64 n_var,
                                            const i64 n_exog, const i64 T,
                                            const i64 n_par) {
  const arena_offset off =
      sdsge_simulate_order2_arena_offset(n_state, n_var, n_exog, T, n_par);
  return make_sizer(off.foffset[off.n_fbuf - 1], 0);
}

arena_offset sdsge_simulate_order2_output_arena_offset(const i64 n_var,
                                                       const i64 n_exog,
                                                       const i64 T,
                                                       const i64 m) {
  return sdsge_simulate_order1_output_arena_offset(n_var, n_exog, T, m);
}

arena_size sdsge_simulate_order2_output_arena_size(const i64 n_var,
                                                   const i64 n_exog,
                                                   const i64 T, const i64 m) {
  return sdsge_simulate_order1_output_arena_size(n_var, n_exog, T, m);
}

arena_offset sdsge_filter_linear_input_arena_offset(const i64 n, const i64 m,
                                                    const i64 k, const i64 T) {
  arena_offset off = make_offset(9, 0);
  off.foffset[0] = n * n;                  // A(n, n)
  off.foffset[1] = off.foffset[0] + n * k; // B(n, k)
  off.foffset[2] = off.foffset[1] + m * n; // C(m, n)
  off.foffset[3] = off.foffset[2] + m;     // d(m)
  off.foffset[4] = off.foffset[3] + k * k; // Q(k, k)
  off.foffset[5] = off.foffset[4] + m * m; // R(m, m)
  off.foffset[6] = off.foffset[5] + T * m; // y(T, m)
  off.foffset[7] = off.foffset[6] + n;     // x0(n)
  off.foffset[8] = off.foffset[7] + n * n; // P0(n, n)
  return off;
}

arena_size sdsge_filter_linear_input_arena_size(const i64 n, const i64 m,
                                                const i64 k, const i64 T) {
  const arena_offset off = sdsge_filter_linear_input_arena_offset(n, m, k, T);
  return make_sizer(off.foffset[off.n_fbuf - 1], 0);
}

arena_offset sdsge_filter_linear_output_arena_offset(const i64 n, const i64 m,
                                                     const i64 k, const i64 T,
                                                     const int return_shocks) {
  arena_offset off = make_offset(11, 0);
  off.foffset[0] = T * n;                      // x_pred(T, n)
  off.foffset[1] = off.foffset[0] + T * n;     // x_filt(T, n)
  off.foffset[2] = off.foffset[1] + T * n * n; // P_pred(T, n, n)
  off.foffset[3] = off.foffset[2] + T * n * n; // P_filt(T, n, n)
  off.foffset[4] = off.foffset[3] + T * m;     // y_pred(T, m)
  off.foffset[5] = off.foffset[4] + T * m;     // y_filt(T, m)
  off.foffset[6] = off.foffset[5] + T * m;     // innov(T, m)
  off.foffset[7] = off.foffset[6] + T * m;     // std_innov(T, m)
  off.foffset[8] = off.foffset[7] + T * m * m; // S(T, m, m)
  off.foffset[9] =
      off.foffset[8] + (return_shocks ? T * k : 0); // eps_hat(T, k) or nothing
  off.foffset[10] = off.foffset[9] + 1;             // loglik
  return off;
}

arena_size sdsge_filter_linear_output_arena_size(const i64 n, const i64 m,
                                                 const i64 k, const i64 T,
                                                 const int return_shocks) {
  const arena_offset off =
      sdsge_filter_linear_output_arena_offset(n, m, k, T, return_shocks);
  return make_sizer(off.foffset[off.n_fbuf - 1], 0);
}

arena_offset sdsge_filter_extended_input_arena_offset(const i64 n, const i64 m,
                                                      const i64 k, const i64 T,
                                                      const i64 n_par) {
  arena_offset off = make_offset(8, 0);
  off.foffset[0] = n * n;                  // A(n, n)
  off.foffset[1] = off.foffset[0] + n * k; // B(n, k)
  off.foffset[2] = off.foffset[1] + n_par; // params(n_par)
  off.foffset[3] = off.foffset[2] + k * k; // Q(k, k)
  off.foffset[4] = off.foffset[3] + m * m; // R(m, m)
  off.foffset[5] = off.foffset[4] + T * m; // y(T, m)
  off.foffset[6] = off.foffset[5] + n;     // x0(n)
  off.foffset[7] = off.foffset[6] + n * n; // P0(n, n)
  return off;
}

arena_size sdsge_filter_extended_input_arena_size(const i64 n, const i64 m,
                                                  const i64 k, const i64 T,
                                                  const i64 n_par) {
  const arena_offset off =
      sdsge_filter_extended_input_arena_offset(n, m, k, T, n_par);
  return make_sizer(off.foffset[off.n_fbuf - 1], 0);
}

arena_offset
sdsge_filter_extended_output_arena_offset(const i64 n, const i64 m, const i64 k,
                                          const i64 T,
                                          const int return_shocks) {
  return sdsge_filter_linear_output_arena_offset(n, m, k, T, return_shocks);
}

arena_size sdsge_filter_extended_output_arena_size(const i64 n, const i64 m,
                                                   const i64 k, const i64 T,
                                                   const int return_shocks) {
  return sdsge_filter_linear_output_arena_size(n, m, k, T, return_shocks);
}

arena_offset
sdsge_filter_unscented_input_arena_offset(const i64 n_state, const i64 n_ctrl,
                                          const i64 n_exog, const i64 n_obs,
                                          const i64 T, const i64 n_par) {
  arena_offset off = make_offset(18, 0);
  const i64 n_var = n_state + n_ctrl;
  const i64 nz = 2 * n_state;
  off.foffset[0] = n_state * n_state;                 // hx(n_state, n_state)
  off.foffset[1] = off.foffset[0] + n_ctrl * n_state; // gx(n_ctrl, n_state)
  off.foffset[2] = off.foffset[1] + n_var * n_exog;   // bu(n_var, n_exog)
  off.foffset[3] =
      off.foffset[2] +
      n_state * n_state * n_state; // hxx(n_state, n_state, n_state)
  off.foffset[4] = off.foffset[3] +
                   n_ctrl * n_state * n_state; // gxx(n_ctrl, n_state, n_state)
  off.foffset[5] = off.foffset[4] +
                   n_state * n_state * n_exog; // hxu(n_state, n_state, n_exog)
  off.foffset[6] = off.foffset[5] +
                   n_ctrl * n_state * n_exog; // gxu(n_ctrl, n_state, n_exog)
  off.foffset[7] = off.foffset[6] +
                   n_state * n_exog * n_exog; // huu(n_state, n_exog, n_exog)
  off.foffset[8] =
      off.foffset[7] + n_ctrl * n_exog * n_exog; // guu(n_ctrl, n_exog, n_exog)
  off.foffset[9] = off.foffset[8] + n_state;     // hss(n_state)
  off.foffset[10] = off.foffset[9] + n_ctrl;     // gss(n_ctrl)
  off.foffset[11] = off.foffset[10] + n_var;     // steady_state(n_var)
  off.foffset[12] = off.foffset[11] + n_par;     // params(n_par)
  off.foffset[13] = off.foffset[12] + n_exog * n_exog; // Q(n_exog, n_exog)
  off.foffset[14] = off.foffset[13] + n_obs * n_obs;   // R(n_obs, n_obs)
  off.foffset[15] = off.foffset[14] + T * n_obs;       // obs(T, n_obs)
  off.foffset[16] = off.foffset[15] + nz;              // z0(nz)
  off.foffset[17] = off.foffset[16] + nz * nz;         // P0(nz, nz)
  return off;
}

arena_size sdsge_filter_unscented_input_arena_size(const i64 n_state,
                                                   const i64 n_ctrl,
                                                   const i64 n_exog,
                                                   const i64 n_obs, const i64 T,
                                                   const i64 n_par) {
  const arena_offset off = sdsge_filter_unscented_input_arena_offset(
      n_state, n_ctrl, n_exog, n_obs, T, n_par);
  return make_sizer(off.foffset[off.n_fbuf - 1], 0);
}

arena_offset sdsge_filter_unscented_output_arena_offset(const i64 n_state,
                                                        const i64 n_ctrl,
                                                        const i64 n_obs,
                                                        const i64 T) {
  arena_offset off = make_offset(14, 0);
  const i64 n_var = n_state + n_ctrl;
  const i64 nz = 2 * n_state;
  off.foffset[0] = T * n_var;                          // x_pred(T, n_var)
  off.foffset[1] = off.foffset[0] + T * n_var;         // x_filt(T, n_var)
  off.foffset[2] = off.foffset[1] + T * nz * nz;       // P_pred(T, nz, nz)
  off.foffset[3] = off.foffset[2] + T * nz * nz;       // P_filt(T, nz, nz)
  off.foffset[4] = off.foffset[3] + T * n_obs;         // y_pred(T, n_obs)
  off.foffset[5] = off.foffset[4] + T * n_obs;         // y_filt(T, n_obs)
  off.foffset[6] = off.foffset[5] + T * n_obs;         // innov(T, n_obs)
  off.foffset[7] = off.foffset[6] + T * n_obs;         // std_innov(T, n_obs)
  off.foffset[8] = off.foffset[7] + T * n_obs * n_obs; // S(T, n_obs, n_obs)
  off.foffset[9] = off.foffset[8] + 1;                 // loglik
  off.foffset[10] = off.foffset[9] + T * n_state;      // x1_pred(T, n_state)
  off.foffset[11] = off.foffset[10] + T * n_state;     // x2_pred(T, n_state)
  off.foffset[12] = off.foffset[11] + T * n_state;     // x1_filt(T, n_state)
  off.foffset[13] = off.foffset[12] + T * n_state;     // x2_filt(T, n_state)
  return off;
}

arena_size sdsge_filter_unscented_output_arena_size(const i64 n_state,
                                                    const i64 n_ctrl,
                                                    const i64 n_obs,
                                                    const i64 T) {
  const arena_offset off =
      sdsge_filter_unscented_output_arena_offset(n_state, n_ctrl, n_obs, T);
  return make_sizer(off.foffset[off.n_fbuf - 1], 0);
}

/* from tests.c */
arena_offset sdsge_mc_diag_sample_arena_offset(const i64 n, const i64 q,
                                               const arena_size work) {
  arena_offset off = make_offset(2, 1);
  off.foffset[0] = n * q;                         // data(n, q)
  off.foffset[1] = off.foffset[0] + work.n_float; // work
  off.ioffset[0] = work.n_int;                    // int work, often empty
  return off;
}

arena_size sdsge_mc_diag_sample_arena_size(const i64 n, const i64 q,
                                           const arena_size work) {
  const arena_offset off = sdsge_mc_diag_sample_arena_offset(n, q, work);
  return make_sizer(off.foffset[off.n_fbuf - 1], off.ioffset[off.n_ibuf - 1]);
}

arena_offset sdsge_mc_diag_output_arena_offset(void) {
  arena_offset off = make_offset(1, 1);
  off.foffset[0] = 1; // statistic
  off.ioffset[0] = 1; // status
  return off;
}

arena_size sdsge_mc_diag_output_arena_size(void) {
  const arena_offset off = sdsge_mc_diag_output_arena_offset();
  return make_sizer(off.foffset[off.n_fbuf - 1], off.ioffset[off.n_ibuf - 1]);
}

arena_offset sdsge_mc_diag_design_arena_offset(const i64 n, const i64 m,
                                               const arena_size work) {
  arena_offset off = make_offset(3, 1);
  off.foffset[0] = n;                             // y(n)
  off.foffset[1] = off.foffset[0] + n * m;        // X(n, m)
  off.foffset[2] = off.foffset[1] + work.n_float; // work
  off.ioffset[0] = work.n_int;                    // int work, often empty
  return off;
}

arena_size sdsge_mc_diag_design_arena_size(const i64 n, const i64 m,
                                           const arena_size work) {
  const arena_offset off = sdsge_mc_diag_design_arena_offset(n, m, work);
  return make_sizer(off.foffset[off.n_fbuf - 1], off.ioffset[off.n_ibuf - 1]);
}

arena_offset sdsge_mc_diag_augmented_arena_offset(const i64 n, const i64 k,
                                                  const arena_size work) {
  arena_offset off = make_offset(4, 1);
  off.foffset[0] = n;                             // eps(n)
  off.foffset[1] = off.foffset[0] + n * k;        // X(n, k)
  off.foffset[2] = off.foffset[1] + n * (k + 1);  // X_aug(n, k+1)
  off.foffset[3] = off.foffset[2] + work.n_float; // work
  off.ioffset[0] = work.n_int;                    // int work, often empty
  return off;
}

arena_size sdsge_mc_diag_augmented_arena_size(const i64 n, const i64 k,
                                              const arena_size work) {
  const arena_offset off = sdsge_mc_diag_augmented_arena_offset(n, k, work);
  return make_sizer(off.foffset[off.n_fbuf - 1], off.ioffset[off.n_ibuf - 1]);
}

/* from regression.c */
/* The one place a regression scratch width is named. No `default`, so a kind
 * added to the enum without a width here fails the build under -Wswitch. */
static arena_size
sdsge_mc_regression_scratch(const sdsge_mc_regression_kind kind, const i64 p,
                            const int intercept, const i64 n_alpha,
                            const i64 max_iter) {
  const i64 k = p - (intercept ? 1 : 0);
  switch (kind) {
  case SDSGE_MC_REGRESSION_OLS:
    return make_sizer(2 * p * p + 2 * p, 0); // L, G, g, work
  case SDSGE_MC_REGRESSION_RIDGE:
    return make_sizer(3 * p * p + 2 * p, 0); // L, G, G_unpen, g, col
  case SDSGE_MC_REGRESSION_RIDGE_GS:
    return make_sizer(3 * p * p + 3 * p, 0); // G_base, G, L, g, coef_work, col
  case SDSGE_MC_REGRESSION_LASSO:
    return make_sizer(2 * p * p + 2 * p, 0); // G_base, G, g, Gcoef
  case SDSGE_MC_REGRESSION_LASSO_GS:
    return make_sizer(2 * p * p + p            // gram
                          + max_iter + 1       // lam_path
                          + (max_iter + 1) * k // beta_path
                          + n_alpha * k        // beta_grid
                          + k * k + 8 * k,     // solver work
                      0);
  case SDSGE_MC_REGRESSION_ELASTIC_NET:
    return make_sizer(2 * p * p + 2 * p, 0); // G_base, G, g, Gcoef
  case SDSGE_MC_REGRESSION_ELASTIC_NET_GS:
    return make_sizer(2 * p * p + 3 * p    // G_base, G, g, Gcoef, beta
                          + n_alpha * k    // beta_grid
                          + 3 * k * k + k, // dof work
                      n_alpha);            // int work
  }
  return make_sizer(0, 0);
}

/* Only OLS reports a standard error. Decided here so no caller repeats it. */
static int sdsge_mc_regression_reports_se(const sdsge_mc_regression_kind kind) {
  return kind == SDSGE_MC_REGRESSION_OLS;
}

arena_offset sdsge_mc_regression_arena_offset(const i64 kind, const i64 n,
                                              const i64 p, const int intercept,
                                              const i64 n_alpha,
                                              const i64 max_iter) {
  arena_offset off = make_offset(3, 1);
  const arena_size scratch = sdsge_mc_regression_scratch(
      (sdsge_mc_regression_kind)kind, p, intercept, n_alpha, max_iter);
  off.foffset[0] = n * p;                            // X(n, p)
  off.foffset[1] = off.foffset[0] + n;               // y(n)
  off.foffset[2] = off.foffset[1] + scratch.n_float; // scratch
  off.ioffset[0] = scratch.n_int;                    // int scratch, often empty
  return off;
}

arena_size sdsge_mc_regression_arena_size(const i64 kind, const i64 n,
                                          const i64 p, const int intercept,
                                          const i64 n_alpha,
                                          const i64 max_iter) {
  const arena_offset off = sdsge_mc_regression_arena_offset(
      kind, n, p, intercept, n_alpha, max_iter);
  return make_sizer(off.foffset[off.n_fbuf - 1], off.ioffset[off.n_ibuf - 1]);
}

arena_offset sdsge_mc_regression_output_arena_offset(const i64 kind,
                                                     const i64 p) {
  arena_offset off = make_offset(4, 1);
  const int with_se =
      sdsge_mc_regression_reports_se((sdsge_mc_regression_kind)kind);
  off.foffset[0] = p;                                  // coef(p)
  off.foffset[1] = off.foffset[0] + 1;                 // ssr
  off.foffset[2] = off.foffset[1] + 1;                 // sst
  off.foffset[3] = off.foffset[2] + (with_se ? p : 0); // se(p), or nothing
  off.ioffset[0] = 1;                                  // status
  return off;
}

arena_size sdsge_mc_regression_output_arena_size(const i64 kind, const i64 p) {
  const arena_offset off = sdsge_mc_regression_output_arena_offset(kind, p);
  return make_sizer(off.foffset[off.n_fbuf - 1], off.ioffset[off.n_ibuf - 1]);
}

/* from transforms.c */
/* The one place a transform scratch width is named. No `default`, so a kind
 * added to the enum without a width here fails the build under -Wswitch. */
static i64 sdsge_mc_transform_scratch(const sdsge_mc_transform_kind kind,
                                      const i64 p, const i64 order) {
  switch (kind) {
  case SDSGE_MC_TRANSFORM_STANDARDIZE:
    return 2 * p; // mean(p), inv_std(p)
  case SDSGE_MC_TRANSFORM_LOG:
    return 0; // takes none
  case SDSGE_MC_TRANSFORM_LOG_DIFF:
    return p; // previous(p)
  case SDSGE_MC_TRANSFORM_DIFF:
    return order * p; // one previous level per order
  case SDSGE_MC_TRANSFORM_ROLLING_MEAN:
    return p; // sum(p)
  case SDSGE_MC_TRANSFORM_ROLLING_VAR:
  case SDSGE_MC_TRANSFORM_ROLLING_STD:
    return 2 * p; // mean(p), m2(p)
  }
  return 0;
}

arena_offset sdsge_mc_transform_arena_offset(const i64 kind, const i64 n,
                                             const i64 p, const i64 order,
                                             const i64 window) {
  (void)window; // no kind sizes its scratch against a window
  arena_offset off = make_offset(2, 0);
  off.foffset[0] = n * p; // input(n, p)
  off.foffset[1] =
      off.foffset[0] +
      sdsge_mc_transform_scratch((sdsge_mc_transform_kind)kind, p, order);
  return off;
}

arena_size sdsge_mc_transform_arena_size(const i64 kind, const i64 n,
                                         const i64 p, const i64 order,
                                         const i64 window) {
  const arena_offset off =
      sdsge_mc_transform_arena_offset(kind, n, p, order, window);
  return make_sizer(off.foffset[off.n_fbuf - 1], 0);
}

/* The one place a transform's output row count is named. No `default`, so a
 * kind added to the enum without a rule here fails the build under -Wswitch. */
i64 sdsge_mc_transform_output_rows(const i64 kind, const i64 n, const i64 order,
                                   const i64 window) {
  i64 rows = 0;

  switch ((sdsge_mc_transform_kind)kind) {
  case SDSGE_MC_TRANSFORM_STANDARDIZE:
  case SDSGE_MC_TRANSFORM_LOG:
    rows = n; // elementwise
    break;
  case SDSGE_MC_TRANSFORM_LOG_DIFF:
    rows = n - 1; // row i differences against row i - 1
    break;
  case SDSGE_MC_TRANSFORM_DIFF:
    rows = n - order; // one row consumed per level
    break;
  case SDSGE_MC_TRANSFORM_ROLLING_MEAN:
  case SDSGE_MC_TRANSFORM_ROLLING_VAR:
  case SDSGE_MC_TRANSFORM_ROLLING_STD:
    rows = n - window + 1; // one row per trailing window
    break;
  }

  return (rows > 0) ? rows : 0;
}
