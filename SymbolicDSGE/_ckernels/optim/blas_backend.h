#ifndef SDSGE_OPTIM_BLAS_BACKEND_H
#define SDSGE_OPTIM_BLAS_BACKEND_H

#include "../_common/sdsge_common.h"

/* The seven BLAS/LAPACK routines lbfgsb.c calls, as the Fortran ABI (pointer
 * args, char flags) exposed by scipy's cython_blas / cython_lapack. The capsule
 * backend assigns these verbatim from __pyx_capi__ pointers (the zgges pattern);
 * the shim backend provides self-contained equivalents. One transpiled setulb
 * body calls through this vtable, so both backends share it and the bench is
 * apples-to-apples.
 *
 * `blas_int` matches scipy's CBLAS_INT (LP64 -> 32-bit int; not ILP64 here). */
typedef int blas_int;

typedef void (*blas_dcopy_t)(blas_int *n, double *x, blas_int *incx, double *y,
                             blas_int *incy);
typedef void (*blas_daxpy_t)(blas_int *n, double *a, double *x, blas_int *incx,
                             double *y, blas_int *incy);
typedef void (*blas_dscal_t)(blas_int *n, double *a, double *x, blas_int *incx);
typedef double (*blas_ddot_t)(blas_int *n, double *x, blas_int *incx, double *y,
                              blas_int *incy);
typedef double (*blas_dnrm2_t)(blas_int *n, double *x, blas_int *incx);
typedef void (*lapack_dpotrf_t)(char *uplo, blas_int *n, double *a,
                                blas_int *lda, blas_int *info);
typedef void (*lapack_dtrtrs_t)(char *uplo, char *trans, char *diag, blas_int *n,
                                blas_int *nrhs, double *a, blas_int *lda,
                                double *b, blas_int *ldb, blas_int *info);

typedef struct {
  blas_dcopy_t dcopy;
  blas_daxpy_t daxpy;
  blas_dscal_t dscal;
  blas_ddot_t ddot;
  blas_dnrm2_t dnrm2;
  lapack_dpotrf_t dpotrf;
  lapack_dtrtrs_t dtrtrs;
} sdsge_blas_ops;

/* Fill `ops` with the self-contained shim implementations (no external BLAS). */
void sdsge_blas_ops_shim(sdsge_blas_ops *ops);

/* Fill `ops` from raw capsule pointers (cython_blas / cython_lapack), casting
 * each to its typed slot. The capsule backend; mirrors the zgges pattern. */
void sdsge_blas_ops_from_ptrs(sdsge_blas_ops *ops, void *dcopy, void *daxpy,
                              void *dscal, void *ddot, void *dnrm2, void *dpotrf,
                              void *dtrtrs);

#endif /* SDSGE_OPTIM_BLAS_BACKEND_H */
