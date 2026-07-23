#include "blas_backend.h"

#include <math.h>

/* Backend the vendored lbfgsb.c dispatches through (see lbfgsb.h). */
const sdsge_blas_ops *sdsge_optim_blas = NULL;

void sdsge_optim_set_blas(const sdsge_blas_ops *ops) { sdsge_optim_blas = ops; }

/* ------------------------------------------------------------------ */
/* Self-contained shims: the Fortran BLAS/LAPACK ABI, no external BLAS. */
/* ------------------------------------------------------------------ */

/* BLAS index of the first element for a strided vector (BLAS convention:
 * negative increments run the vector backwards). */
static inline blas_int shim_ix0(blas_int n, blas_int inc) {
  return inc > 0 ? 0 : (n - 1) * (-inc);
}

static void shim_dcopy(blas_int *n, double *x, blas_int *incx, double *y,
                       blas_int *incy) {
  blas_int N = *n, ix = shim_ix0(N, *incx), iy = shim_ix0(N, *incy);
  for (blas_int i = 0; i < N; ++i, ix += *incx, iy += *incy) {
    y[iy] = x[ix];
  }
}

static void shim_daxpy(blas_int *n, double *a, double *x, blas_int *incx,
                       double *y, blas_int *incy) {
  const double alpha = *a;
  blas_int N = *n, ix = shim_ix0(N, *incx), iy = shim_ix0(N, *incy);
  for (blas_int i = 0; i < N; ++i, ix += *incx, iy += *incy) {
    y[iy] += alpha * x[ix];
  }
}

static void shim_dscal(blas_int *n, double *a, double *x, blas_int *incx) {
  const double alpha = *a;
  blas_int N = *n, ix = shim_ix0(N, *incx);
  for (blas_int i = 0; i < N; ++i, ix += *incx) {
    x[ix] *= alpha;
  }
}

static double shim_ddot(blas_int *n, double *x, blas_int *incx, double *y,
                        blas_int *incy) {
  blas_int N = *n, ix = shim_ix0(N, *incx), iy = shim_ix0(N, *incy);
  double s = 0.0;
  for (blas_int i = 0; i < N; ++i, ix += *incx, iy += *incy) {
    s += x[ix] * y[iy];
  }
  return s;
}

/* Scaled 2-norm, matching LAPACK dnrm2's overflow-safe reduction. */
static double shim_dnrm2(blas_int *n, double *x, blas_int *incx) {
  blas_int N = *n, ix = shim_ix0(N, *incx);
  if (N < 1 || *incx == 0) {
    return 0.0;
  }
  double scale = 0.0, ssq = 1.0;
  for (blas_int i = 0; i < N; ++i, ix += *incx) {
    double xi = x[ix];
    if (xi != 0.0) {
      double ax = fabs(xi);
      if (scale < ax) {
        double r = scale / ax;
        ssq = 1.0 + ssq * r * r;
        scale = ax;
      } else {
        double r = ax / scale;
        ssq += r * r;
      }
    }
  }
  return scale * sqrt(ssq);
}

/* Cholesky of an SPD matrix, in place, column-major, leading dim lda. uplo 'U':
 * A = U^T U (upper stored); 'L': A = L L^T (lower stored). info = 0 on success,
 * k>0 if the order-k leading minor is not positive definite. */
static void shim_dpotrf(char *uplo, blas_int *np, double *a, blas_int *ldap,
                        blas_int *info) {
  const blas_int n = *np, lda = *ldap;
  const int upper = (*uplo == 'U' || *uplo == 'u');
  *info = 0;
  if (upper) {
    for (blas_int j = 0; j < n; ++j) {
      double d = a[j + j * lda];
      for (blas_int k = 0; k < j; ++k) {
        d -= a[k + j * lda] * a[k + j * lda];
      }
      if (!(d > 0.0)) {
        *info = j + 1;
        return;
      }
      d = sqrt(d);
      a[j + j * lda] = d;
      for (blas_int i = j + 1; i < n; ++i) {
        double s = a[j + i * lda];
        for (blas_int k = 0; k < j; ++k) {
          s -= a[k + j * lda] * a[k + i * lda];
        }
        a[j + i * lda] = s / d;
      }
    }
  } else {
    for (blas_int j = 0; j < n; ++j) {
      double d = a[j + j * lda];
      for (blas_int k = 0; k < j; ++k) {
        d -= a[j + k * lda] * a[j + k * lda];
      }
      if (!(d > 0.0)) {
        *info = j + 1;
        return;
      }
      d = sqrt(d);
      a[j + j * lda] = d;
      for (blas_int i = j + 1; i < n; ++i) {
        double s = a[i + j * lda];
        for (blas_int k = 0; k < j; ++k) {
          s -= a[i + k * lda] * a[j + k * lda];
        }
        a[i + j * lda] = s / d;
      }
    }
  }
}

/* Solve op(A) X = B for triangular A, in place on B (nrhs columns), column-major.
 * uplo U/L, trans N/T (op = A or A^T), diag N/U (unit diagonal). */
static void shim_dtrtrs(char *uplo, char *trans, char *diag, blas_int *np,
                        blas_int *nrhsp, double *a, blas_int *ldap, double *b,
                        blas_int *ldbp, blas_int *info) {
  const blas_int n = *np, nrhs = *nrhsp, lda = *ldap, ldb = *ldbp;
  const int upper = (*uplo == 'U' || *uplo == 'u');
  const int notrans = (*trans == 'N' || *trans == 'n');
  const int unit = (*diag == 'U' || *diag == 'u');
  *info = 0;
  if (!unit) {
    for (blas_int i = 0; i < n; ++i) {
      if (a[i + i * lda] == 0.0) {
        *info = i + 1;
        return;
      }
    }
  }
  for (blas_int c = 0; c < nrhs; ++c) {
    double *x = &b[(size_t)c * ldb];
    if (notrans && upper) { /* U x = b : back substitution */
      for (blas_int i = n - 1; i >= 0; --i) {
        double s = x[i];
        for (blas_int k = i + 1; k < n; ++k) {
          s -= a[i + k * lda] * x[k];
        }
        x[i] = unit ? s : s / a[i + i * lda];
      }
    } else if (notrans && !upper) { /* L x = b : forward substitution */
      for (blas_int i = 0; i < n; ++i) {
        double s = x[i];
        for (blas_int k = 0; k < i; ++k) {
          s -= a[i + k * lda] * x[k];
        }
        x[i] = unit ? s : s / a[i + i * lda];
      }
    } else if (!notrans && upper) { /* U^T x = b : U^T lower, forward */
      for (blas_int i = 0; i < n; ++i) {
        double s = x[i];
        for (blas_int k = 0; k < i; ++k) {
          s -= a[k + i * lda] * x[k];
        }
        x[i] = unit ? s : s / a[i + i * lda];
      }
    } else { /* L^T x = b : L^T upper, back */
      for (blas_int i = n - 1; i >= 0; --i) {
        double s = x[i];
        for (blas_int k = i + 1; k < n; ++k) {
          s -= a[k + i * lda] * x[k];
        }
        x[i] = unit ? s : s / a[i + i * lda];
      }
    }
  }
}

void sdsge_blas_ops_shim(sdsge_blas_ops *ops) {
  ops->dcopy = shim_dcopy;
  ops->daxpy = shim_daxpy;
  ops->dscal = shim_dscal;
  ops->ddot = shim_ddot;
  ops->dnrm2 = shim_dnrm2;
  ops->dpotrf = shim_dpotrf;
  ops->dtrtrs = shim_dtrtrs;
}

void sdsge_blas_ops_from_ptrs(sdsge_blas_ops *ops, void *dcopy, void *daxpy,
                              void *dscal, void *ddot, void *dnrm2, void *dpotrf,
                              void *dtrtrs) {
  ops->dcopy = (blas_dcopy_t)dcopy;
  ops->daxpy = (blas_daxpy_t)daxpy;
  ops->dscal = (blas_dscal_t)dscal;
  ops->ddot = (blas_ddot_t)ddot;
  ops->dnrm2 = (blas_dnrm2_t)dnrm2;
  ops->dpotrf = (lapack_dpotrf_t)dpotrf;
  ops->dtrtrs = (lapack_dtrtrs_t)dtrtrs;
}
