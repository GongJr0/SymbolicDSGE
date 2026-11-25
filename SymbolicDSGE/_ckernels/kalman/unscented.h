#ifndef SDSGE_KF_UNSCENTED_H
#define SDSGE_KF_UNSCENTED_H
#include "../_common/sdsge_common.h"
#include "kalman.h"

/* Unscented Kalman Filter */

typedef struct {
  meas_fn meas;
  const f64 *hx;
  const f64 *gx;
  const f64 *bu;
  const f64 *hxx;
  const f64 *gxx;
  const f64 *hxu;
  const f64 *gxu;
  const f64 *huu;
  const f64 *guu;
  const f64 *hss;
  const f64 *gss;
  const f64 *steady_state;
  const f64 *params;
  const f64 *Q;
  const f64 *R;
  const f64 *obs;
  const f64 *z0;
  const f64 *P0;
  i64 T;

  i64 n_state;
  i64 n_ctrl;
  i64 n_exog;
  i64 n_obs;
  i64 n_params;

  f64 alpha;
  f64 beta;
  f64 kappa;

  f64 jitter;
  int symmetrize;
  int store_history;
} ukf_inputs;

typedef struct {
  f64 *x1_pred; /* (T, n_state) */
  f64 *x2_pred; /* (T, n_state) */
  f64 *x1_filt; /* (T, n_state) */
  f64 *x2_filt; /* (T, n_state) */

  f64 *x_pred; /* (T, n_state + n_ctrl) */
  f64 *x_filt; /* (T, n_state + n_ctrl) */

  f64 *P_pred; /* (T, 2*n_state, 2*n_state) */
  f64 *P_filt; /* (T, 2*n_state, 2*n_state) */

  f64 *y_pred;    /* (T, n_obs) */
  f64 *y_filt;    /* (T, n_obs) */
  f64 *innov;     /* (T, n_obs) */
  f64 *std_innov; /* (T, n_obs) */
  f64 *S;         /* (T, n_obs, n_obs) */

  f64 *eps_hat; /* (T, n_exog) */
  f64 *loglik;
} ukf_outputs;

typedef struct {
  ukf_outputs *fwd_out;
  f64 *x1_smooth;  /* (T, n_state) */
  f64 *x2_smooth;  /* (T, n_state) */
  f64 *x_smooth;   /* (T, n_state + n_ctrl) */
  f64 *P_smooth;   /* (T, 2*n_state, 2*n_state) */
  f64 *eps_smooth; /* (T, n_exog) */
} ukf_smooth_outputs;

/* Run the unscented Kalman filter: second-order transition, nonlinear
 * measurements. Returns KF_OK, KF_ERR_P_FACTORIZATION, KF_ERR_Q_FACTORIZATION,
 * KF_ERR_S_FACTORIZATION, or KF_ERR_BAD_HYPERPARAMS. */
arena_size ukf_arena_size(const i64 n_state, const i64 n_ctrl, const i64 n_exog,
                          const i64 n_obs);
int ukf_forward(const ukf_inputs *in, f64 *SDSGE_RESTRICT arena,
                ukf_outputs *out);

/* Unscented RTS smoother on the augmented [z_{t-1}; eps_t] state. Returns the
 * forward pass's codes, or KF_ERR_P_FACTORIZATION if a filtered covariance
 * cannot be factored. */
arena_size ukf_smooth_arena_size(const i64 n_state, const i64 n_ctrl,
                                 const i64 n_exog, const i64 n_obs,
                                 const i64 T);
int ukf_smooth(const ukf_inputs *in, f64 *SDSGE_RESTRICT arena,
               ukf_smooth_outputs *out);

#endif // !SDSGE_KF_UNSCENTED_H
