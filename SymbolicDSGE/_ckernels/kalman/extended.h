#ifndef SDSGE_KF_EXTENDED_H
#define SDSGE_KF_EXTENDED_H
#include "../_common/sdsge_common.h"
#include "kalman.h"

/* Extended Kalman Filter */
typedef struct {
  meas_fn meas;
  meas_fn jac;
  const f64 *A, *B;
  const f64 *calib_params;
  const f64 *Q, *R, *steady_state, *y;
  const f64 *x0; /* (n,)   initial state mean */
  const f64 *P0; /* (n, n) initial state covariance (pre-symmetrized) */
  i64 T, n, m, k, n_par;
  f64 jitter;
  int symmetrize, joseph_cov, compute_y_filt, return_shocks, store_history;
} ekf_inputs;

typedef struct {
  f64 *x_pred, *x_filt;   /* (hT, n) */
  f64 *P_pred, *P_filt;   /* (hT, n, n) */
  f64 *y_pred, *y_filt;   /* (hT, m) */
  f64 *innov, *std_innov; /* (hT, m) */
  f64 *S;                 /* (hT, m, m) */
  f64 *eps_hat;           /* (shock_hT, k) */
  f64 *loglik;            /* scalar out */
} ekf_outputs;

typedef struct {
  ekf_outputs *fwd_out;
  f64 *x_smooth;   /* (T, n) */
  f64 *P_smooth;   /* (T, n, n) */
  f64 *eps_smooth; /* (T, k) */
} ekf_smooth_outputs;

/* Run the extended Kalman filter: linear transition, nonlinear measurement
 * via the meas/jac cfunc pointers (relinearized each step). Returns KF_OK,
 * KF_ERR_S_FACTORIZATION */
arena_size ekf_arena_size(const i64 n, const i64 m, const i64 k);
int ekf_forward(const ekf_inputs *in, f64 *SDSGE_RESTRICT arena,
                ekf_outputs *out);

#endif // !SDSGE_KF_EXTENDED_H
