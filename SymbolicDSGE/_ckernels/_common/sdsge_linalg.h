#ifndef SDSGE_LINALG_H
#define SDSGE_LINALG_H

#include "sdsge_common.h"

/* Dense linear-algebra primitives shared across the native subsystems. These
 * were first written for the kalman hot loop (as the `kf_*` kernels) and are now
 * promoted here so the regression / diagnostic kernels reuse them. Everything is
 * plain C on C-contiguous, row-major f64 buffers; no CPython, no NumPy, no BLAS
 * (matrices are small and we keep bit-parity with the numba reference). Buffers
 * are caller-allocated and never alias unless a function documents otherwise. */

/* out(r,c) := 0 */
void sdsge_zero_mat(f64 *out, i64 r, i64 c);

/* P(n,n) := (P + P^T) / 2, in place */
void sdsge_sym_inplace(f64 *P, i64 n);

/* out(n,m) := A(n,p) @ B(p,m) */
void sdsge_matmul(const f64 *A, const f64 *B, f64 *out, i64 n, i64 p, i64 m);

/* out(n,m) := A(n,p) @ B(m,p)^T */
void sdsge_matmul_abt(const f64 *A, const f64 *B, f64 *out, i64 n, i64 p, i64 m);

/* out(n,m) := A(n,p) @ B(m,p)^T + C(n,m) */
void sdsge_matmul_abt_plus_c(const f64 *A, const f64 *B, const f64 *C, f64 *out,
                             i64 n, i64 p, i64 m);

/* out(n) := A(n,m) @ x(m) */
void sdsge_matvec(const f64 *A, const f64 *x, f64 *out, i64 n, i64 m);

/* out(n) := A(n,m) @ x(m) + b(n) */
void sdsge_matvec_plus_vec(const f64 *A, const f64 *x, const f64 *b, f64 *out,
                           i64 n, i64 m);

/* dot product of a(n) and b(n) */
f64 sdsge_dot(const f64 *a, const f64 *b, i64 n);

/* 2 * sum(log(diag(L))) for lower-triangular L(n,n) */
f64 sdsge_logdet_from_chol(const f64 *L, i64 n);

/* Lower Cholesky L(n,n) of S(n,n) (+ jitter on the diagonal). Returns SDSGE_OK,
 * or SDSGE_NOT_PD if S is not positive definite. Pass jitter == 0 for a plain
 * factorization. */
int sdsge_chol(const f64 *S, f64 jitter, f64 *L, i64 n);

/* Solve L(n,n) x = b(n) for lower-triangular L; writes x into out(n). out may
 * alias b. */
void sdsge_forward_subst(const f64 *L, const f64 *b, f64 *out, i64 n);

/* Solve L^T x = b using lower-triangular L(n,n); writes x into out(n). out may
 * alias b. */
void sdsge_backward_subst_chol_t(const f64 *L, const f64 *b, f64 *out, i64 n);

/* ---- Gram-matrix / normal-equation helpers (regression, diagnostics) ---- */

/* G(p,p) := X(n,p)^T X(n,p). Lower triangle accumulated row-by-row (matching the
 * numba xtx_xty summation order), then mirrored to full symmetric. */
void sdsge_gram(const f64 *X, f64 *G, i64 n, i64 p);

/* g(p) := X(n,p)^T y(n) */
void sdsge_gram_rhs(const f64 *X, const f64 *y, f64 *g, i64 n, i64 p);

/* Solve the SPD system G(p,p) coef = g(p) via Cholesky. coef(p) is the output
 * (may alias g); scratch_L(p,p) holds the factor. Returns SDSGE_OK or
 * SDSGE_NOT_PD (G not positive definite -- caller falls back to lstsq). */
int sdsge_chol_solve(const f64 *G, const f64 *g, f64 *coef, f64 *scratch_L,
                     i64 p);

/* SPD inverse: Pinv(p,p) := G(p,p)^-1 via Cholesky. scratch_L(p,p) holds the
 * factor. Returns SDSGE_OK or SDSGE_NOT_PD. */
int sdsge_chol_inv(const f64 *G, f64 *Pinv, f64 *scratch_L, i64 p);

#endif /* SDSGE_LINALG_H */
