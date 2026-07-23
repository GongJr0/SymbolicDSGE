#include "shim.h"

#include <math.h>

/* Linear-algebra primitives for the vendored L-BFGS-B kernel (see shim.h).
 * Hand-rolled, no external library. The kernel passes operands by pointer. */

/* Index of the first element for a strided vector (a negative increment runs the
 * vector backwards). */
static inline i64 shim_ix0(i64 n, i64 inc) {
  return inc > 0 ? 0 : (n - 1) * (-inc);
}

void sdsge_shim_dcopy(i64 *n, f64 *x, i64 *incx, f64 *y, i64 *incy) {
  i64 N = *n, ix = shim_ix0(N, *incx), iy = shim_ix0(N, *incy);
  for (i64 i = 0; i < N; ++i, ix += *incx, iy += *incy) {
    y[iy] = x[ix];
  }
}

void sdsge_shim_daxpy(i64 *n, f64 *a, f64 *x, i64 *incx, f64 *y, i64 *incy) {
  const f64 alpha = *a;
  i64 N = *n, ix = shim_ix0(N, *incx), iy = shim_ix0(N, *incy);
  for (i64 i = 0; i < N; ++i, ix += *incx, iy += *incy) {
    y[iy] += alpha * x[ix];
  }
}

void sdsge_shim_dscal(i64 *n, f64 *a, f64 *x, i64 *incx) {
  const f64 alpha = *a;
  i64 N = *n, ix = shim_ix0(N, *incx);
  for (i64 i = 0; i < N; ++i, ix += *incx) {
    x[ix] *= alpha;
  }
}

f64 sdsge_shim_ddot(i64 *n, f64 *x, i64 *incx, f64 *y, i64 *incy) {
  i64 N = *n, ix = shim_ix0(N, *incx), iy = shim_ix0(N, *incy);
  f64 s = 0.0;
  for (i64 i = 0; i < N; ++i, ix += *incx, iy += *incy) {
    s += x[ix] * y[iy];
  }
  return s;
}

/* Scaled 2-norm (overflow-safe reduction). */
f64 sdsge_shim_dnrm2(i64 *n, f64 *x, i64 *incx) {
  i64 N = *n, ix = shim_ix0(N, *incx);
  if (N < 1 || *incx == 0) {
    return 0.0;
  }
  f64 scale = 0.0, ssq = 1.0;
  for (i64 i = 0; i < N; ++i, ix += *incx) {
    f64 xi = x[ix];
    if (xi != 0.0) {
      f64 ax = fabs(xi);
      if (scale < ax) {
        f64 r = scale / ax;
        ssq = 1.0 + ssq * r * r;
        scale = ax;
      } else {
        f64 r = ax / scale;
        ssq += r * r;
      }
    }
  }
  return scale * sqrt(ssq);
}

/* Cholesky of an SPD matrix, in place, column-major, leading dim lda. uplo 'U':
 * A = U^T U (upper stored); 'L': A = L L^T (lower stored). info = 0 on success,
 * k>0 if the order-k leading minor is not positive definite. */
void sdsge_shim_dpotrf(char *uplo, i64 *np, f64 *a, i64 *ldap, i64 *info) {
  const i64 n = *np, lda = *ldap;
  const int upper = (*uplo == 'U' || *uplo == 'u');
  *info = 0;
  if (upper) {
    for (i64 j = 0; j < n; ++j) {
      f64 d = a[j + j * lda];
      for (i64 k = 0; k < j; ++k) {
        d -= a[k + j * lda] * a[k + j * lda];
      }
      if (!(d > 0.0)) {
        *info = j + 1;
        return;
      }
      d = sqrt(d);
      a[j + j * lda] = d;
      for (i64 i = j + 1; i < n; ++i) {
        f64 s = a[j + i * lda];
        for (i64 k = 0; k < j; ++k) {
          s -= a[k + j * lda] * a[k + i * lda];
        }
        a[j + i * lda] = s / d;
      }
    }
  } else {
    for (i64 j = 0; j < n; ++j) {
      f64 d = a[j + j * lda];
      for (i64 k = 0; k < j; ++k) {
        d -= a[j + k * lda] * a[j + k * lda];
      }
      if (!(d > 0.0)) {
        *info = j + 1;
        return;
      }
      d = sqrt(d);
      a[j + j * lda] = d;
      for (i64 i = j + 1; i < n; ++i) {
        f64 s = a[i + j * lda];
        for (i64 k = 0; k < j; ++k) {
          s -= a[i + k * lda] * a[j + k * lda];
        }
        a[i + j * lda] = s / d;
      }
    }
  }
}

/* Solve op(A) X = B for triangular A, in place on B (nrhs columns), column-major.
 * uplo U/L, trans N/T (op = A or A^T), diag N/U (unit diagonal). */
void sdsge_shim_dtrtrs(char *uplo, char *trans, char *diag, i64 *np, i64 *nrhsp,
                       f64 *a, i64 *ldap, f64 *b, i64 *ldbp, i64 *info) {
  const i64 n = *np, nrhs = *nrhsp, lda = *ldap, ldb = *ldbp;
  const int upper = (*uplo == 'U' || *uplo == 'u');
  const int notrans = (*trans == 'N' || *trans == 'n');
  const int unit = (*diag == 'U' || *diag == 'u');
  *info = 0;
  if (!unit) {
    for (i64 i = 0; i < n; ++i) {
      if (a[i + i * lda] == 0.0) {
        *info = i + 1;
        return;
      }
    }
  }
  for (i64 c = 0; c < nrhs; ++c) {
    f64 *x = &b[c * ldb];
    if (notrans && upper) { /* U x = b : back substitution */
      for (i64 i = n - 1; i >= 0; --i) {
        f64 s = x[i];
        for (i64 k = i + 1; k < n; ++k) {
          s -= a[i + k * lda] * x[k];
        }
        x[i] = unit ? s : s / a[i + i * lda];
      }
    } else if (notrans && !upper) { /* L x = b : forward substitution */
      for (i64 i = 0; i < n; ++i) {
        f64 s = x[i];
        for (i64 k = 0; k < i; ++k) {
          s -= a[i + k * lda] * x[k];
        }
        x[i] = unit ? s : s / a[i + i * lda];
      }
    } else if (!notrans && upper) { /* U^T x = b : U^T lower, forward */
      for (i64 i = 0; i < n; ++i) {
        f64 s = x[i];
        for (i64 k = 0; k < i; ++k) {
          s -= a[k + i * lda] * x[k];
        }
        x[i] = unit ? s : s / a[i + i * lda];
      }
    } else { /* L^T x = b : L^T upper, back */
      for (i64 i = n - 1; i >= 0; --i) {
        f64 s = x[i];
        for (i64 k = i + 1; k < n; ++k) {
          s -= a[k + i * lda] * x[k];
        }
        x[i] = unit ? s : s / a[i + i * lda];
      }
    }
  }
}
