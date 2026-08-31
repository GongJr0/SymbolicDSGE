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

/* Buffers each arena holds, in the order the offsets report them. One name for
 * the storage a caller reserves and the bound a walk runs to, so the two cannot
 * disagree. */
#define SDSGE_MC_DATAGEN_OUT_BUFFERS 3
#define SDSGE_MC_PASSTHROUGH_OUT_BUFFERS 1
#define SDSGE_MC_SIMULATE1_IN_BUFFERS 7
#define SDSGE_MC_SIMULATE2_IN_BUFFERS 16
#define SDSGE_MC_FILTER_LINEAR_IN_BUFFERS 9
#define SDSGE_MC_FILTER_EXTENDED_IN_BUFFERS 8
#define SDSGE_MC_FILTER_UNSCENTED_IN_BUFFERS 18
#define SDSGE_MC_FILTER_OUT_BUFFERS 11
#define SDSGE_MC_FILTER_UNSCENTED_OUT_BUFFERS 14
#define SDSGE_MC_REGRESSION_IN_BUFFERS 3
#define SDSGE_MC_REGRESSION_IN_INT_BUFFERS 1
#define SDSGE_MC_REGRESSION_OUT_BUFFERS 4
#define SDSGE_MC_REGRESSION_OUT_INT_BUFFERS 1
#define SDSGE_MC_TRANSFORM_IN_BUFFERS 2

/* Every count above fits an `arena_offset`. A lane that outgrows the descriptor
 * fails here rather than writing past the entries it was given. */
typedef char sdsge_mc_layout_capacity_check
    [(SDSGE_MC_DATAGEN_OUT_BUFFERS <= SDSGE_ARENA_MAX_BUFFERS &&
      SDSGE_MC_PASSTHROUGH_OUT_BUFFERS <= SDSGE_ARENA_MAX_BUFFERS &&
      SDSGE_MC_SIMULATE1_IN_BUFFERS <= SDSGE_ARENA_MAX_BUFFERS &&
      SDSGE_MC_SIMULATE2_IN_BUFFERS <= SDSGE_ARENA_MAX_BUFFERS &&
      SDSGE_MC_FILTER_LINEAR_IN_BUFFERS <= SDSGE_ARENA_MAX_BUFFERS &&
      SDSGE_MC_FILTER_EXTENDED_IN_BUFFERS <= SDSGE_ARENA_MAX_BUFFERS &&
      SDSGE_MC_FILTER_UNSCENTED_IN_BUFFERS <= SDSGE_ARENA_MAX_BUFFERS &&
      SDSGE_MC_FILTER_OUT_BUFFERS <= SDSGE_ARENA_MAX_BUFFERS &&
      SDSGE_MC_FILTER_UNSCENTED_OUT_BUFFERS <= SDSGE_ARENA_MAX_BUFFERS &&
      SDSGE_MC_REGRESSION_IN_BUFFERS <= SDSGE_ARENA_MAX_BUFFERS &&
      SDSGE_MC_REGRESSION_IN_INT_BUFFERS <= SDSGE_ARENA_MAX_BUFFERS &&
      SDSGE_MC_REGRESSION_OUT_BUFFERS <= SDSGE_ARENA_MAX_BUFFERS &&
      SDSGE_MC_REGRESSION_OUT_INT_BUFFERS <= SDSGE_ARENA_MAX_BUFFERS &&
      SDSGE_MC_TRANSFORM_IN_BUFFERS <= SDSGE_ARENA_MAX_BUFFERS)
         ? 1
         : -1];

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

/* from regression.h */
/* Every regression arena is [X(n,p), y(n), scratch]. ``p`` is the design width
 * after optional intercept augmentation. The scratch is one buffer here and
 * stays one: only the fit reads inside it, so nothing outside has to agree on
 * what it holds, and a `*_scratch_arena_size` states its width without a
 * layout. ``elastic_net_gs`` also takes an int lane, one buffer wide, which it
 * hands to the fit whole.
 *
 * Every kind writes the same output arena, [coef(p), ssr, sst, se(p)] over an
 * int lane of [status]. ``se`` is empty unless the kind reports one, so the
 * fields ahead of it do not move, and one layout serves all seven. */
arena_offset sdsge_mc_regression_output_arena_offset(i64 p, int with_se);
arena_size sdsge_mc_regression_output_arena_size(i64 p, int with_se);
arena_size sdsge_mc_ols_scratch_arena_size(i64 p);
arena_offset sdsge_mc_ols_work_arena_offset(i64 n, i64 p);
arena_size sdsge_mc_ols_work_arena_size(i64 n, i64 p);
arena_size sdsge_mc_ridge_scratch_arena_size(i64 p);
arena_offset sdsge_mc_ridge_work_arena_offset(i64 n, i64 p);
arena_size sdsge_mc_ridge_work_arena_size(i64 n, i64 p);
arena_size sdsge_mc_ridge_gs_scratch_arena_size(i64 p);
arena_offset sdsge_mc_ridge_gs_work_arena_offset(i64 n, i64 p);
arena_size sdsge_mc_ridge_gs_work_arena_size(i64 n, i64 p);
arena_size sdsge_mc_lasso_scratch_arena_size(i64 p);
arena_offset sdsge_mc_lasso_work_arena_offset(i64 n, i64 p);
arena_size sdsge_mc_lasso_work_arena_size(i64 n, i64 p);
arena_size sdsge_mc_lasso_gs_scratch_arena_size(i64 p, int intercept,
                                                i64 n_alpha, i64 max_iter);
arena_offset sdsge_mc_lasso_gs_work_arena_offset(i64 n, i64 p, int intercept,
                                                 i64 n_alpha, i64 max_iter);
arena_size sdsge_mc_lasso_gs_work_arena_size(i64 n, i64 p, int intercept,
                                             i64 n_alpha, i64 max_iter);
arena_size sdsge_mc_elastic_net_scratch_arena_size(i64 p);
arena_offset sdsge_mc_elastic_net_work_arena_offset(i64 n, i64 p);
arena_size sdsge_mc_elastic_net_work_arena_size(i64 n, i64 p);
arena_size sdsge_mc_elastic_net_gs_scratch_arena_size(i64 p, int intercept,
                                                      i64 n_alpha);
arena_offset sdsge_mc_elastic_net_gs_work_arena_offset(i64 n, i64 p,
                                                       int intercept,
                                                       i64 n_alpha);
arena_size sdsge_mc_elastic_net_gs_work_arena_size(i64 n, i64 p, int intercept,
                                                   i64 n_alpha);

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
