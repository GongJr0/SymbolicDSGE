#ifndef SDSGE_LINALG_H
#define SDSGE_LINALG_H

#include "sdsge_common.h"

/* Dense linear-algebra primitives shared across the native subsystems. These
 * were first written for the kalman hot loop (as the `kf_*` kernels) and are
 * now promoted here so the regression / diagnostic kernels reuse them.
 * Everything is plain C on C-contiguous, row-major f64 buffers; no CPython, no
 * NumPy, no BLAS (matrices are small and we keep bit-parity with the numba
 * reference). Buffers are caller-allocated and never alias unless a function
 * documents otherwise. */

/* out(r,c) := 0 */
void sdsge_zero_mat(f64 *out, i64 r, i64 c);

/* P(n,n) := (P + P^T) / 2, in place */
void sdsge_sym_inplace(f64 *P, i64 n);

/* out(n,m) := A(n,p) @ B(p,m) */
void sdsge_matmul(const f64 *A, const f64 *B, f64 *out, i64 n, i64 p, i64 m);

/* out(p,m) := A(n,p)^T @ B(n,m). The transpose is folded into the indexing (no
 * materialized A^T); contraction is over the row axis, accumulated row-by-row
 * like sdsge_gram (of which this is the asymmetric generalization: matmul_atb
 * of X with itself equals the full gram of X). `out` must not alias A or B, but
 * A and B may overlap each other -- both are read-only, so e.g. lagged views of
 * one buffer (A = M, B = M + j*cols) are fine. */
void sdsge_matmul_atb(const f64 *A, const f64 *B, f64 *out, i64 n, i64 p,
                      i64 m);

/* out(n,m) := A(n,p) @ B(m,p)^T */
void sdsge_matmul_abt(const f64 *A, const f64 *B, f64 *out, i64 n, i64 p,
                      i64 m);

/* out(n,m) := A(n,p) @ B(m,p)^T + C(n,m) */
void sdsge_matmul_abt_plus_c(const f64 *A, const f64 *B, const f64 *C, f64 *out,
                             i64 n, i64 p, i64 m);

/* out(n) := A(n,m) @ x(m) */
void sdsge_matvec(const f64 *A, const f64 *x, f64 *out, i64 n, i64 m);

/* out(n) := A(n,m) @ x(m) + b(n) */
void sdsge_matvec_plus_vec(const f64 *A, const f64 *x, const f64 *b, f64 *out,
                           i64 n, i64 m);

/* out(n) := a(n) - b(n).  out may alias a or b. */
void sdsge_vsub(const f64 *a, const f64 *b, f64 *out, i64 n);

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

/* G(p,p) := X(n,p)^T X(n,p). Lower triangle accumulated row-by-row (matching
 * the numba xtx_xty summation order), then mirrored to full symmetric. */
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

/* ---- small statistical reductions (diagnostics) ---- */

/* Sort n doubles ascending, in place (libc qsort). */
void sdsge_sort_f64(f64 *arr, i64 n);

/* Median of n doubles; sorts arr IN PLACE (destroys order). Even n averages the
 * two central order statistics (matches numpy.median). Returns NaN for n <= 0.
 */
f64 sdsge_median_f64(f64 *arr, i64 n);

/* LU Types and Functions */
typedef struct {
  f64 *lu;
  i64 *piv;
  i64 n;
  i64 err;
} f64_lu;

i64 sdsge_lu_factor_inplace(f64 *SDSGE_RESTRICT A, i64 *SDSGE_RESTRICT pivot,
                            const i64 n);

f64_lu sdsge_lu_factor(const f64 *SDSGE_RESTRICT A, const i64 n);

void sdsge_lu_free(f64_lu *lu);

void sdsge_lu_solve(const f64 *SDSGE_RESTRICT LU, const i64 *SDSGE_RESTRICT piv,
                    const f64 *SDSGE_RESTRICT B, f64 *SDSGE_RESTRICT X,
                    const i64 n, const i64 m);

/* Solve A(n,n) X = B(n,m) via LU with partial pivoting (X may not alias B).
 * Returns SDSGE_LU_SUCCESS, SDSGE_LU_SINGULAR, or SDSGE_LU_ALLOC_FAIL. */
i64 sdsge_solve(const f64 *SDSGE_RESTRICT A, const f64 *SDSGE_RESTRICT B,
                const i64 n, const i64 m, f64 *SDSGE_RESTRICT X);

/* Ainv(n,n) := A(n,n)^-1 via LU. Same return codes as sdsge_solve. */
i64 sdsge_inv(const f64 *SDSGE_RESTRICT A, const i64 n,
              f64 *SDSGE_RESTRICT Ainv);

/* ERROR CODES */
#define SDSGE_LU_SUCCESS 0
#define SDSGE_LU_ALLOC_FAIL -1
#define SDSGE_LU_SINGULAR -2

#endif /* SDSGE_LINALG_H */
