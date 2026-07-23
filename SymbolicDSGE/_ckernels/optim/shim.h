#ifndef SDSGE_OPTIM_SHIM_H
#define SDSGE_OPTIM_SHIM_H

#include "../_common/sdsge_common.h"

/* Low-level linear-algebra primitives the vendored L-BFGS-B kernel calls
 * (vector copy / axpy / scal / dot / 2-norm, Cholesky factor, triangular solve).
 * Self-contained: hand-rolled loops, no external library. The kernel calls these
 * directly, passing operands by pointer (its calling convention); the operands
 * are always small, where a plain loop beats a linked library's per-call
 * overhead. Correct for arbitrary strides / uplo / trans / diag / lda. */

void sdsge_shim_dcopy(i64 *n, f64 *x, i64 *incx, f64 *y, i64 *incy);
void sdsge_shim_daxpy(i64 *n, f64 *a, f64 *x, i64 *incx, f64 *y, i64 *incy);
void sdsge_shim_dscal(i64 *n, f64 *a, f64 *x, i64 *incx);
f64 sdsge_shim_ddot(i64 *n, f64 *x, i64 *incx, f64 *y, i64 *incy);
f64 sdsge_shim_dnrm2(i64 *n, f64 *x, i64 *incx);
void sdsge_shim_dpotrf(char *uplo, i64 *n, f64 *a, i64 *lda, i64 *info);
void sdsge_shim_dtrtrs(char *uplo, char *trans, char *diag, i64 *n, i64 *nrhs,
                       f64 *a, i64 *lda, f64 *b, i64 *ldb, i64 *info);

#endif /* SDSGE_OPTIM_SHIM_H */
