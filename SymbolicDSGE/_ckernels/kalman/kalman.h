#ifndef SDSGE_KALMAN_H
#define SDSGE_KALMAN_H

#include "../_common/sdsge_common.h"

#define KF_OK 0
#define KF_ERR_COMPLEX_MATRIX -1
#define KF_ERR_SHAPE_MISMATCH -2
#define KF_ERR_MATRIX_CONDITION -3
#define KF_ERR_SINGULAR_MATRIX -4

/* Kalman hot-loop helpers ported from the numba `*_into` kernels in
 * SymbolicDSGE/kalman/filter.py. All matrices are C-contiguous, row-major, f64.
 * Buffers are caller-allocated; nothing here aliases (inputs and outputs are
 * always distinct). The dense linear-algebra primitives (kf_matmul, kf_chol_*,
 * kf_*subst, kf_dot, ...) are candidates to promote to _common/sdsge_linalg
 * once the regression port needs them. */

/* ---- Dense linear-algebra primitives ---- */

/* out(r,c) := 0 */
void kf_zero_mat(f64 *out, i64 r, i64 c);

/* P(n,n) := (P + P^T) / 2, in place */
void kf_sym_inplace(f64 *P, i64 n);

/* out(n,m) := A(n,p) @ B(p,m) */
void kf_matmul(const f64 *A, const f64 *B, f64 *out, i64 n, i64 p, i64 m);

/* out(n,m) := A(n,p) @ B(m,p)^T */
void kf_matmul_abt(const f64 *A, const f64 *B, f64 *out, i64 n, i64 p, i64 m);

/* out(n,m) := A(n,p) @ B(m,p)^T + C(n,m) */
void kf_matmul_abt_plus_c(const f64 *A, const f64 *B, const f64 *C, f64 *out,
                          i64 n, i64 p, i64 m);

/* out(n) := A(n,m) @ x(m) */
void kf_matvec(const f64 *A, const f64 *x, f64 *out, i64 n, i64 m);

/* out(n) := A(n,m) @ x(m) + b(n) */
void kf_matvec_plus_vec(const f64 *A, const f64 *x, const f64 *b, f64 *out,
                        i64 n, i64 m);

/* out(m) := A[row, :] - x(m), where A has m columns */
void kf_row_minus_vec(const f64 *A, i64 row, const f64 *x, f64 *out, i64 m);

/* dot product of a(n) and b(n) */
f64 kf_dot(const f64 *a, const f64 *b, i64 n);

/* 2 * sum(log(diag(L))) for lower-triangular L(n,n) */
f64 kf_logdet_from_chol(const f64 *L, i64 n);

/* Lower Cholesky L(n,n) of S(n,n) (+ jitter on the diagonal). Returns KF_OK,
 * or KF_ERR_MATRIX_CONDITION if S is not positive definite. */
int kf_chol_shifted(const f64 *S, f64 jitter, f64 *L, i64 n);

/* Solve L(n,n) x = b(n) for lower-triangular L; writes x into out(n). */
void kf_forward_subst(const f64 *L, const f64 *b, f64 *out, i64 n);

/* Solve L^T x = b using lower-triangular L(n,n); writes x into out(n). */
void kf_backward_subst_chol_t(const f64 *L, const f64 *b, f64 *out, i64 n);

/* Solve (L L^T) x = B[row, :] for the single row; writes x into out[row, :].
 * `n` is the Cholesky dimension (= column count of B and out); fbuf/bbuf are
 * scratch of length n. */
void kf_chol_solve_row(const f64 *L, const f64 *B, i64 row, f64 *fbuf, f64 *bbuf,
                       f64 *out, i64 n);

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
void kf_build_bqbt(const f64 *B, const f64 *Q, f64 *temp_nk, f64 *out,
                   i64 n, i64 k);

/* out(k,m) := Q(k,k) @ (B^T C^T)(k,m), via temp_km(k,m) */
void kf_build_shock_projection(const f64 *B, const f64 *C, const f64 *Q,
                               f64 *temp_km, f64 *out, i64 n, i64 k, i64 m);

#endif /* SDSGE_KALMAN_H */
