#ifndef SDSGE_MC_LAYOUT_H
#define SDSGE_MC_LAYOUT_H

#include "../_common/sdsge_common.h"
#include "../kalman/kalman.h"

/* One step kind's arena description: where each buffer starts in its lane, and
 * how much of the lane it all takes. The totals ride alongside rather than
 * being summed out of the offsets, so an allocation reads a number instead of
 * walking for one. */
typedef struct {
  arena_size asize;
  arena_offset aoffset;
} sdsge_mc_layout;

/* from core_steps.h */
/* Source passthrough. ``input`` is the bound (n, p) source span in the input
 * arena, copied straight to the output so one field of a producer can be
 * retained on its own rather than the producer's whole output block. */
arena_size sdsge_passthrough_arena_size(i64 n, i64 p);
arena_offset sdsge_passthrough_arena_offset(i64 n, i64 p);
/* Raw model-data materialization. ``*_batched`` selects input[rep_idx] from
 * a leading replication axis; otherwise the same input span is copied each
 * time. Either input/output pair may be NULL when its element count is
 * zero. */
arena_size sdsge_raw_model_data_output_arena_size(i64 n_states, i64 n_shocks,
                                                  i64 n_observables);
arena_offset sdsge_raw_model_data_output_arena_offset(i64 n_states,
                                                      i64 n_shocks,
                                                      i64 n_observables);
/* ``input`` is [A(n,n), B(n,k), x0(n), shock(T,k), params(n_par)]. ``simout``
 * is [states(T,n), shocks(T,k), observables(T,m)], and ``m`` is zero when the
 * step was built without observables, which drops the trailing block. The shock
 * block is echoed from the input arena, which is where the runner stages a
 * bound path and where the native draw writes one, so a replication's own draw
 * is recoverable from its output alone. */
arena_size sdsge_simulate_order1_arena_size(i64 n, i64 k, i64 T, i64 n_par);
arena_offset sdsge_simulate_order1_arena_offset(i64 n, i64 k, i64 T, i64 n_par);
arena_size sdsge_simulate_order1_output_arena_size(i64 n, i64 k, i64 T, i64 m);
arena_offset sdsge_simulate_order1_output_arena_offset(i64 n, i64 k, i64 T,
                                                       i64 m);
/* ``input`` is [hx(nx,nx), gx(ny,nx), bu(nx+ny,n_exog), hxx(nx,nx,nx),
 * gxx(ny,nx,nx), hxu(nx,nx,n_exog), gxu(ny,nx,n_exog), huu(nx,n_exog,n_exog),
 * guu(ny,n_exog,n_exog), hss(nx), gss(ny), steady_state(nx+ny), x0(nx),
 * shock(T,n_exog), params(n_par), scratch]. ``simout`` matches the first-order
 * layout above, over ``nx+ny`` variables.
 *
 * ``bu`` spans every variable, not just the states: a control responds to an
 * innovation contemporaneously, which is why the first-order layout above it
 * carries the whole loading too. As there, the step splits the arena and calls
 * core's parameterized kernel. */
arena_size sdsge_simulate_order2_arena_size(i64 n_state, i64 n_var, i64 n_exog,
                                            i64 T, i64 n_par);
arena_offset sdsge_simulate_order2_arena_offset(i64 n_state, i64 n_var,
                                                i64 n_exog, i64 T, i64 n_par);
arena_size sdsge_simulate_order2_output_arena_size(i64 n_var, i64 n_exog, i64 T,
                                                   i64 m);
arena_offset sdsge_simulate_order2_output_arena_offset(i64 n_var, i64 n_exog,
                                                       i64 T, i64 m);
/* Linear filter input is [A(n,n), B(n,k), C(m,n), d(m), Q(k,k), R(m,m),
 * y(T,m), x0(n), P0(n,n)]. Output follows FilterRawResult field order:
 * [x_pred, x_filt, P_pred, P_filt, y_pred, y_filt, innov, std_innov, S,
 * eps_hat, loglik]. ``eps_hat`` is empty unless ``return_shocks``, so the
 * fields after it do not move when it is off. */
arena_size sdsge_filter_linear_input_arena_size(i64 n, i64 m, i64 k, i64 T);
arena_offset sdsge_filter_linear_input_arena_offset(i64 n, i64 m, i64 k, i64 T);
arena_size sdsge_filter_linear_output_arena_size(i64 n, i64 m, i64 k, i64 T,
                                                 int return_shocks);
arena_offset sdsge_filter_linear_output_arena_offset(i64 n, i64 m, i64 k, i64 T,
                                                     int return_shocks);
/* Extended filter input is [A(n,n), B(n,k), params(n_par), Q(k,k), R(m,m),
 * y(T,m), x0(n), P0(n,n)]. Output has the linear filter layout. */
arena_size sdsge_filter_extended_input_arena_size(i64 n, i64 m, i64 k, i64 T,
                                                  i64 n_par);
arena_offset sdsge_filter_extended_input_arena_offset(i64 n, i64 m, i64 k,
                                                      i64 T, i64 n_par);
arena_size sdsge_filter_extended_output_arena_size(i64 n, i64 m, i64 k, i64 T,
                                                   int return_shocks);
arena_offset sdsge_filter_extended_output_arena_offset(i64 n, i64 m, i64 k,
                                                       i64 T,
                                                       int return_shocks);
/* Unscented input is [hx(ns,ns), gx(nc,ns), bu(ns+nc,ne), hxx(ns,ns,ns),
 * gxx(nc,ns,ns), hxu(ns,ns,ne), gxu(nc,ns,ne), huu(ns,ne,ne), guu(nc,ne,ne),
 * hss(ns), gss(nc), steady_state(ns+nc), params(n_par), Q(ne,ne), R(no,no),
 * obs(T,no), z0(2*ns), P0(2*ns,2*ns)]. As in the second-order simulate, ``bu``
 * spans every variable rather than the states alone. Output follows
 * UnscentedFilterRawResult field order, which has no eps_hat and places loglik
 * ahead of the pruned first- and second-order state histories. */
arena_size sdsge_filter_unscented_input_arena_size(i64 n_state, i64 n_ctrl,
                                                   i64 n_exog, i64 n_obs, i64 T,
                                                   i64 n_par);
arena_offset sdsge_filter_unscented_input_arena_offset(i64 n_state, i64 n_ctrl,
                                                       i64 n_exog, i64 n_obs,
                                                       i64 T, i64 n_par);
arena_size sdsge_filter_unscented_output_arena_size(i64 n_state, i64 n_ctrl,
                                                    i64 n_obs, i64 T);
arena_offset sdsge_filter_unscented_output_arena_offset(i64 n_state, i64 n_ctrl,
                                                        i64 n_obs, i64 T);

/* from tests.h */
/* A diagnostic stages its data, then hands the `diag` kernels one work block
 * they size themselves, the way a filter hands kalman its scratch. `work` is a
 * parameter rather than something these layouts know, and each shape declares
 * an int buffer that stays empty unless the work asked for one.
 *
 * Three shapes cover every kind; a kind supplies its own dimensions and its own
 * work, so no layout here needs to know which diagnostic it is serving. */

/* [data(n,q), work]. The wald kinds over an (n,q) sample, and ljung_box and
 * jarque_bera over an (n,1) series whose work may be empty. */
arena_offset sdsge_mc_diag_sample_arena_offset(i64 n, i64 q, arena_size work);
arena_size sdsge_mc_diag_sample_arena_size(i64 n, i64 q, arena_size work);

/* Every diagnostic writes the same output: one statistic over an int lane of
 * one status. */
arena_offset sdsge_mc_diag_output_arena_offset(void);
arena_size sdsge_mc_diag_output_arena_size(void);

/* [y(n), X(n,m), work]. breusch_godfrey, chow, cusum and cusumsq, over the
 * design width each of them names differently. */
arena_offset sdsge_mc_diag_design_arena_offset(i64 n, i64 m, arena_size work);
arena_size sdsge_mc_diag_design_arena_size(i64 n, i64 m, arena_size work);

/* [eps(n), X(n,k), X_aug(n,k+1), work]. breusch_pagan alone, which builds the
 * augmented design in the arena rather than receiving it staged. */
arena_offset sdsge_mc_diag_augmented_arena_offset(i64 n, i64 k,
                                                  arena_size work);
arena_size sdsge_mc_diag_augmented_arena_size(i64 n, i64 k, arena_size work);

/* from regression.h */
/* Regression kinds, as the arena sees them. Data, not a status: the value
 * selects how much scratch a kind asks for and whether it reports a standard
 * error. */
typedef enum {
  SDSGE_MC_REGRESSION_OLS = 0,
  SDSGE_MC_REGRESSION_RIDGE,
  SDSGE_MC_REGRESSION_RIDGE_GS,
  SDSGE_MC_REGRESSION_LASSO,
  SDSGE_MC_REGRESSION_LASSO_GS,
  SDSGE_MC_REGRESSION_ELASTIC_NET,
  SDSGE_MC_REGRESSION_ELASTIC_NET_GS,
} sdsge_mc_regression_kind;

/* Every regression arena is [X(n,p), y(n), scratch] over an int lane the fit
 * takes whole. ``p`` is the design width after optional intercept
 * augmentation. The scratch stays one buffer because only the fit reads inside
 * it, and its width is named nowhere but `layout.c`. ``intercept``,
 * ``n_alpha`` and ``max_iter`` are read by the kinds that need them and
 * ignored by the rest. */
arena_offset sdsge_mc_regression_arena_offset(i64 kind, i64 n, i64 p,
                                              int intercept, i64 n_alpha,
                                              i64 max_iter);
arena_size sdsge_mc_regression_arena_size(i64 kind, i64 n, i64 p, int intercept,
                                          i64 n_alpha, i64 max_iter);

/* Output is [coef(p), ssr, sst, se(p)] over an int lane of [status]. Only OLS
 * reports a standard error, which the kind decides here rather than at each
 * caller, and ``se`` is empty otherwise so nothing ahead of it moves. */
arena_offset sdsge_mc_regression_output_arena_offset(i64 kind, i64 p);
arena_size sdsge_mc_regression_output_arena_size(i64 kind, i64 p);

/* from transforms.h */
/* Transform kinds, as the arena sees them. Data, not a status: the value only
 * selects how much scratch a kind asks for. */
typedef enum {
  SDSGE_MC_TRANSFORM_STANDARDIZE = 0,
  SDSGE_MC_TRANSFORM_LOG,
  SDSGE_MC_TRANSFORM_LOG_DIFF,
  SDSGE_MC_TRANSFORM_DIFF,
  SDSGE_MC_TRANSFORM_ROLLING_MEAN,
  SDSGE_MC_TRANSFORM_ROLLING_VAR,
  SDSGE_MC_TRANSFORM_ROLLING_STD,
} sdsge_mc_transform_kind;

/* Every transform arena is [input(n,p), scratch], and the output is written to
 * a lane of its own. One layout serves them all because only the scratch width
 * varies, and that width is named nowhere but `layout.c`.
 *
 * ``order`` is read by ``diff`` alone. A rolling window never reaches the
 * arena: it bounds a loop and scales a mean, while the running state stays p or
 * 2p wide whatever it is. The window does set the output shape,
 * out(n - window + 1, p), which the caller resolves from the field, not here.
 *
 * Scratch by kind: standardize 2p, log none, log_diff p, diff order*p,
 * rolling_mean p, rolling_var 2p, rolling_std 2p. */
arena_offset sdsge_mc_transform_arena_offset(i64 kind, i64 n, i64 p, i64 order);
arena_size sdsge_mc_transform_arena_size(i64 kind, i64 n, i64 p, i64 order);

#endif // !SDSGE_MC_LAYOUT_H
