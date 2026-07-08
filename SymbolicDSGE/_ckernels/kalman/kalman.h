#ifndef SDSGE_KALMAN_H
#define SDSGE_KALMAN_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_linalg.h"

/* ERROR CODES */

#define KF_OK 0
#define KF_ERR_COMPLEX_MATRIX -1
#define KF_ERR_SHAPE_MISMATCH -2
#define KF_ERR_MATRIX_CONDITION -3
#define KF_ERR_SINGULAR_MATRIX -4
#define KF_ERR_ALLOC -5

/* Kalman hot-loop helpers ported from the numba `*_into` kernels in
 * SymbolicDSGE/kalman/filter.py. All matrices are C-contiguous, row-major, f64.
 * Buffers are caller-allocated; nothing here aliases (inputs and outputs are
 * always distinct). The dense linear-algebra primitives these build on
 * (sdsge_matmul, sdsge_chol, sdsge_*subst, sdsge_dot, ...) now live in
 * _common/sdsge_linalg so the regression / diagnostic kernels share them. */

/* -- Kalman-specific dense helpers (not general enough for sdsge_linalg) -- */

/* out(m) := A[row, :] - x(m), where A has m columns */
void kf_row_minus_vec(const f64 *A, i64 row, const f64 *x, f64 *out, i64 m);

/* Solve (L L^T) x = B[row, :] for the single row; writes x into out[row, :].
 * `n` is the Cholesky dimension (= column count of B and out); fbuf/bbuf are
 * scratch of length n. */
void kf_chol_solve_row(const f64 *L, const f64 *B, i64 row, f64 *fbuf,
                       f64 *bbuf, f64 *out, i64 n);

/* ---- Kalman-specific composition helpers ---- */

/* out(n,n) := A P_prev A^T + BQBT, via temp_nn(n,n) */
void kf_predict_cov(const f64 *A, const f64 *P_prev, const f64 *BQBT,
                    f64 *temp_nn, f64 *out, i64 n);

/* out(m,m) := C P_pred C^T + R, via temp_mn(m,n) */
void kf_measurement_cov(const f64 *C, const f64 *P_pred, const f64 *R,
                        f64 *temp_mn, f64 *out, i64 n, i64 m);

/* out(n,m) := P_pred(n,n) @ C(m,n)^T */
void kf_pc_t(const f64 *P_pred, const f64 *C, f64 *out, i64 n, i64 m);

/* out K(n,m): solve (L L^T) K[r,:] = PCt[r,:] per row; L is the m×m chol of S.
 * fbuf/bbuf are scratch of length m. */
void kf_gain_from_pc_t(const f64 *L, const f64 *PCt, f64 *fbuf, f64 *bbuf,
                       f64 *out, i64 n, i64 m);

/* out(n) := x_pred(n) + K(n,m) @ v(m) */
void kf_state_update(const f64 *x_pred, const f64 *K, const f64 *v, f64 *out,
                     i64 n, i64 m);

/* out(n,n) := I - A(n,n) */
void kf_identity_minus(const f64 *A, f64 *out, i64 n);

/* Joseph-form covariance update:
 *   out(n,n) := (I - K C) P_pred (I - K C)^T + K R K^T
 * KC, I_minus_KC, temp_nn are (n,n) scratch; temp_nm is (n,m) scratch. */
void kf_joseph_cov(const f64 *K, const f64 *C, const f64 *P_pred, const f64 *R,
                   f64 *KC, f64 *I_minus_KC, f64 *temp_nn, f64 *temp_nm,
                   f64 *out, i64 n, i64 m);

/* out(n,n) := sym(B(n,k) Q(k,k) B^T), via temp_nk(n,k) */
void kf_build_bqbt(const f64 *B, const f64 *Q, f64 *temp_nk, f64 *out, i64 n,
                   i64 k);

/* out(k,m) := Q(k,k) @ (B^T C^T)(k,m), via temp_km(k,m) */
void kf_build_shock_projection(const f64 *B, const f64 *C, const f64 *Q,
                               f64 *temp_km, f64 *out, i64 n, i64 k, i64 m);

/* ---- Linear Kalman filter hot loop ---- */

/* Model + data inputs. All matrices C-contiguous, row-major, f64. */
typedef struct {
  i64 n;             /* state dimension */
  i64 m;             /* observation dimension */
  i64 k;             /* shock dimension */
  i64 T;             /* number of time steps */
  const f64 *A;      /* (n, n) state transition */
  const f64 *B;      /* (n, k) shock loading */
  const f64 *C;      /* (m, n) observation */
  const f64 *d;      /* (m,)   observation intercept */
  const f64 *Q;      /* (k, k) shock covariance */
  const f64 *R;      /* (m, m) measurement covariance */
  const f64 *y;      /* (T, m) observations */
  const f64 *x0;     /* (n,)   initial state mean */
  const f64 *P0;     /* (n, n) initial state covariance (pre-symmetrized) */
  int symmetrize;    /* symmetrize P/S each step (0/1) */
  f64 jitter;        /* diagonal jitter for the Cholesky */
  int return_shocks; /* compute eps_hat (0/1) */
  int store_history; /* fill the per-step history arrays (0/1) */
} kf_inputs;

/* Caller-allocated outputs. When store_history is 0 the history arrays may be
 * NULL / zero-length; eps_hat is written only when return_shocks &&
 * store_history. loglik is always written. */
typedef struct {
  f64 *x_pred;    /* (T, n) */
  f64 *x_filt;    /* (T, n) */
  f64 *P_pred;    /* (T, n, n) */
  f64 *P_filt;    /* (T, n, n) */
  f64 *y_pred;    /* (T, m) */
  f64 *y_filt;    /* (T, m) */
  f64 *innov;     /* (T, m)  innovations v */
  f64 *std_innov; /* (T, m)  L^-1 v */
  f64 *S;         /* (T, m, m) innovation covariances */
  f64 *eps_hat;   /* (T, k) or NULL */
  f64 *loglik;    /* scalar out */
} kf_outputs;

/* Run the linear Kalman filter. Returns KF_OK, KF_ERR_MATRIX_CONDITION (non-PD
 * innovation covariance), or KF_ERR_ALLOC (scratch allocation failed). */
int kf_hot_loop(const kf_inputs *in, kf_outputs *out);

/* Unscented Kalman Filter */

typedef void (*meas_fn)(const f64 *SDSGE_RESTRICT x,
                        const f64 *SDSGE_RESTRICT params, f64 *out);

typedef struct {
  meas_fn meas;
  const f64 *hx;
  const f64 *gx;
  const f64 *bx;
  const f64 *hxx;
  const f64 *gxx;
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

  f64 *loglik;
} ukf_outputs;

i64 ukf_hot_loop(const ukf_inputs *in, ukf_outputs *out);

#endif /* SDSGE_KALMAN_H */
