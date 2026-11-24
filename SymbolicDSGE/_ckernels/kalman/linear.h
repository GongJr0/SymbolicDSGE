#ifndef SDSGE_KF_LINEAR_H
#define SDSGE_KF_LINEAR_H
#include "../_common/sdsge_common.h"

/* ---- Linear Kalman filter ---- */

/* Model + data inputs. All matrices C-contiguous, row-major, f64. */
typedef struct {
  i64 n;                   /* state dimension */
  i64 m;                   /* observation dimension */
  i64 k;                   /* shock dimension */
  i64 T;                   /* number of time steps */
  const f64 *A;            /* (n, n) state transition */
  const f64 *B;            /* (n, k) shock loading */
  const f64 *C;            /* (m, n) observation */
  const f64 *d;            /* (m,)   observation intercept */
  const f64 *Q;            /* (k, k) shock covariance */
  const f64 *R;            /* (m, m) measurement covariance */
  const f64 *steady_state; /* (n,) steady state of model variables */
  const f64 *y;            /* (T, m) observations */
  const f64 *x0;           /* (n,)   initial state mean */
  const f64 *P0;     /* (n, n) initial state covariance (pre-symmetrized) */
  int symmetrize;    /* symmetrize P/S each step (0/1) */
  int joseph_cov;    /* use Joseph form for P update (0/1) */
  f64 jitter;        /* diagonal jitter for the Cholesky */
  int return_shocks; /* compute eps_hat (0/1) */
  int store_history; /* fill the per-step history arrays (0/1) */
} kf_inputs;

/* Caller-allocated outputs. When store_history is 0 the history arrays may be
 * NULL / zero-length; eps_hat is written only when return_shocks &&
 * store_history. loglik is always written. */
typedef struct {
  f64 *x_pred, *x_filt;   /* (T, n) */
  f64 *P_pred, *P_filt;   /* (T, n, n) */
  f64 *y_pred, *y_filt;   /* (T, m) */
  f64 *innov, *std_innov; /* (T, m)  innovations v */
  f64 *S;                 /* (T, m, m) innovation covariances */
  f64 *eps_hat;           /* (T, k) or NULL */
  f64 *loglik;            /* scalar out */
} kf_outputs;

typedef struct {
  kf_outputs *fwd_out;
  f64 *x_smooth;   /* (T, n) */
  f64 *P_smooth;   /* (T, n, n) */
  f64 *eps_smooth; /* (T, k) */
} kf_smooth_outputs;

/* Run the linear Kalman filter. Returns KF_OK, KF_ERR_S_FACTORIZATION
 * (non-PD innovation covariance) */
arena_size kf_arena_size(const i64 n, const i64 m, const i64 k);
int kf_forward(const kf_inputs *in, f64 *SDSGE_RESTRICT arena, kf_outputs *out);

/* Linear Kalman smoother. Cannot fail unless the forward pass fails.
 * Returns KF_OK, KF_ERR_S_FACTORIZATION. */
int kf_smooth(const kf_inputs *in, f64 *SDSGE_RESTRICT arena,
              kf_smooth_outputs *out);

#endif /* SDSGE_KF_LINEAR_H */
