#include "sdsge_complex.h"
#include <stdlib.h>
#include <string.h>

/* Scalar c128 arithmetic is now defined `static inline` in sdsge_complex.h. */

/* Linalg (LU, Matmul, Solve, Inverse) */

void c128_matmul(const c128 *SDSGE_RESTRICT A, const c128 *SDSGE_RESTRICT B,
                 const i64 m, const i64 n, const i64 p,
                 c128 *SDSGE_RESTRICT out) {
  /* GEMM row-major */

  for (i64 i = 0; i < m; ++i) {
    for (i64 j = 0; j < p; ++j) {
      c128 sum = c128_make(0.0, 0.0);

      for (i64 k = 0; k < n; ++k) {
        sum = c128_add(sum, c128_mul(A[i * n + k], B[k * p + j]));
      }
      out[i * p + j] = sum;
    }
  }
}

i64 c128_lu_factor_inplace(c128 *SDSGE_RESTRICT A, i64 *SDSGE_RESTRICT pivot,
                           const i64 n) {
  if (!A || !pivot || n <= 0) {
    return SDSGE_LU_ALLOC_FAIL;
  }

  for (i64 k = 0; k < n; ++k) {
    i64 piv = k;
    f64 best = c128_abs2(A[k * n + k]);

    for (i64 i = k + 1; i < n; ++i) {
      f64 v = c128_abs2(A[i * n + k]);
      if (v > best) {
        best = v;
        piv = i;
      }
    }
    pivot[k] = piv;

    if (best == 0.0) {
      return SDSGE_LU_SINGULAR;
    }

    if (piv != k) {
      for (i64 j = 0; j < n; ++j) {
        c128 tmp = A[k * n + j];
        A[k * n + j] = A[piv * n + j];
        A[piv * n + j] = tmp;
      }
    }

    c128 Akk = A[k * n + k];
    for (i64 i = k + 1; i < n; ++i) {
      A[i * n + k] = c128_div(A[i * n + k], Akk);

      c128 Lik = A[i * n + k];
      for (i64 j = k + 1; j < n; ++j) {
        A[i * n + j] = c128_sub(A[i * n + j], c128_mul(Lik, A[k * n + j]));
      }
    }
  }
  return SDSGE_LU_SUCCESS;
}

c128_lu c128_lu_factor(const c128 *A, const i64 n) {
  c128_lu result;
  result.lu = NULL;
  result.piv = NULL;
  result.n = n;
  result.err = SDSGE_LU_SUCCESS;

  if (!A || n <= 0) {
    result.err = SDSGE_LU_ALLOC_FAIL;
    return result;
  }

  result.lu = (c128 *)malloc(sizeof(c128) * n * n);
  result.piv = (i64 *)malloc(sizeof(i64) * n);
  if (!result.lu || !result.piv) {
    result.err = SDSGE_LU_ALLOC_FAIL;
    free(result.lu);
    free(result.piv);
    result.lu = NULL;
    result.piv = NULL;
    return result;
  }

  memcpy(result.lu, A, sizeof(c128) * n * n);
  result.err = c128_lu_factor_inplace(result.lu, result.piv, n);
  return result;
}

void c128_lu_free(c128_lu *lu) {
  if (!lu)
    return;
  free(lu->lu);
  free(lu->piv);
  lu->lu = NULL;
  lu->piv = NULL;
  lu->n = 0;
  lu->err = 0;
}

void c128_lu_solve(const c128 *SDSGE_RESTRICT LU, const i64 *SDSGE_RESTRICT piv,
                   const c128 *SDSGE_RESTRICT B, c128 *SDSGE_RESTRICT X,
                   const i64 n, const i64 m) {
  /* X := B, then replay the factorization's row swaps (k <-> piv[k], forward).
   */
  memcpy(X, B, sizeof(c128) * n * m);
  for (i64 k = 0; k < n; ++k) {
    i64 pk = piv[k];
    if (pk != k) {
      for (i64 j = 0; j < m; ++j) {
        c128 tmp = X[k * m + j];
        X[k * m + j] = X[pk * m + j];
        X[pk * m + j] = tmp;
      }
    }
  }

  /* Forward substitution: L Y = PB, with L unit lower-triangular. */
  for (i64 i = 0; i < n; ++i) {
    for (i64 k = 0; k < i; ++k) {
      c128 Lik = LU[i * n + k];
      for (i64 j = 0; j < m; ++j) {
        X[i * m + j] = c128_sub(X[i * m + j], c128_mul(Lik, X[k * m + j]));
      }
    }
  }

  /* Back substitution: U X = Y. */
  for (i64 i = n - 1; i >= 0; --i) {
    for (i64 k = i + 1; k < n; ++k) {
      c128 Uik = LU[i * n + k];
      for (i64 j = 0; j < m; ++j) {
        X[i * m + j] = c128_sub(X[i * m + j], c128_mul(Uik, X[k * m + j]));
      }
    }
    c128 Uii = LU[i * n + i];
    for (i64 j = 0; j < m; ++j) {
      X[i * m + j] = c128_div(X[i * m + j], Uii);
    }
  }
}
