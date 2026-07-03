#include "sdsge_linalg.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void sdsge_zero_mat(f64 *SDSGE_RESTRICT out, i64 r, i64 c) {
  const i64 total = r * c;
  for (i64 i = 0; i < total; ++i)
    out[i] = 0.0;
}

void sdsge_sym_inplace(f64 *SDSGE_RESTRICT P, i64 n) {
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = i + 1; j < n; ++j) {
      f64 avg = 0.5 * (P[i * n + j] + P[j * n + i]);
      P[i * n + j] = avg;
      P[j * n + i] = avg;
    }
  }
}

void sdsge_matmul(const f64 *SDSGE_RESTRICT A, const f64 *SDSGE_RESTRICT B,
                  f64 *SDSGE_RESTRICT out, i64 n, i64 p, i64 m) {
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < m; ++j) {
      f64 s = 0.0;
      for (i64 k = 0; k < p; ++k)
        s += A[i * p + k] * B[k * m + j];
      out[i * m + j] = s;
    }
  }
}

void sdsge_matmul_atb(const f64 *SDSGE_RESTRICT A, const f64 *SDSGE_RESTRICT B,
                      f64 *SDSGE_RESTRICT out, i64 n, i64 p, i64 m) {
  /* Row-contraction accumulated row-by-row (matches sdsge_gram's order, and is
   * cache-optimal: A row and B row stream contiguously, out stays hot). */
  sdsge_zero_mat(out, p, m);
  for (i64 i = 0; i < n; ++i) {
    const f64 *Ai = A + i * p;
    const f64 *Bi = B + i * m;
    for (i64 k = 0; k < p; ++k) {
      f64 aik = Ai[k];
      for (i64 l = 0; l < m; ++l)
        out[k * m + l] += aik * Bi[l];
    }
  }
}

void sdsge_matmul_abt(const f64 *SDSGE_RESTRICT A, const f64 *SDSGE_RESTRICT B,
                      f64 *SDSGE_RESTRICT out, i64 n, i64 p, i64 m) {
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < m; ++j) {
      f64 s = 0.0;
      for (i64 k = 0; k < p; ++k)
        s += A[i * p + k] * B[j * p + k];
      out[i * m + j] = s;
    }
  }
}

void sdsge_matmul_abt_plus_c(const f64 *SDSGE_RESTRICT A,
                             const f64 *SDSGE_RESTRICT B,
                             const f64 *SDSGE_RESTRICT C,
                             f64 *SDSGE_RESTRICT out, i64 n, i64 p, i64 m) {
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < m; ++j) {
      f64 s = 0.0;
      for (i64 k = 0; k < p; ++k)
        s += A[i * p + k] * B[j * p + k];
      out[i * m + j] = s + C[i * m + j];
    }
  }
}

void sdsge_matvec(const f64 *SDSGE_RESTRICT A, const f64 *SDSGE_RESTRICT x,
                  f64 *SDSGE_RESTRICT out, i64 n, i64 m) {
  for (i64 i = 0; i < n; ++i) {
    f64 s = 0.0;
    for (i64 j = 0; j < m; ++j)
      s += A[i * m + j] * x[j];
    out[i] = s;
  }
}

void sdsge_matvec_plus_vec(const f64 *SDSGE_RESTRICT A,
                           const f64 *SDSGE_RESTRICT x,
                           const f64 *SDSGE_RESTRICT b, f64 *SDSGE_RESTRICT out,
                           i64 n, i64 m) {
  for (i64 i = 0; i < n; ++i) {
    f64 s = b[i];
    for (i64 j = 0; j < m; ++j)
      s += A[i * m + j] * x[j];
    out[i] = s;
  }
}

void sdsge_vsub(const f64 *a, const f64 *b, f64 *out, i64 n) {
  for (i64 i = 0; i < n; ++i)
    out[i] = a[i] - b[i];
}

f64 sdsge_dot(const f64 *SDSGE_RESTRICT a, const f64 *SDSGE_RESTRICT b, i64 n) {
  f64 s = 0.0;
  for (i64 i = 0; i < n; ++i)
    s += a[i] * b[i];
  return s;
}

f64 sdsge_logdet_from_chol(const f64 *SDSGE_RESTRICT L, i64 n) {
  f64 s = 0.0;
  for (i64 i = 0; i < n; ++i)
    s += log(L[i * n + i]);
  return 2.0 * s;
}

int sdsge_chol(const f64 *SDSGE_RESTRICT S, f64 jitter, f64 *SDSGE_RESTRICT L,
               i64 n) {
  sdsge_zero_mat(L, n, n);
  /* Scale reference for a relative positive-definiteness threshold: the
   * largest (jittered) diagonal entry. A genuinely rank-deficient matrix has
   * an exact-arithmetic pivot of zero, but in floating point that pivot rounds
   * to a tiny value of either sign that depends on the compiler/BLAS build.
   * Testing the pivot against 0.0 therefore detects rank deficiency
   * nondeterministically across builds. Comparing against scale * n * eps
   * makes the decision deterministic while never rejecting a pivot of a truly
   * positive-definite matrix (those sit far above this floor). */
  f64 scale = 0.0;
  for (i64 i = 0; i < n; ++i) {
    f64 d = S[i * n + i];
    if (jitter > 0.0)
      d += jitter;
    if (d > scale)
      scale = d;
  }
  const f64 pivot_tol = scale * (f64)n * DBL_EPSILON;
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j <= i; ++j) {
      f64 s = S[i * n + j];
      if (i == j && jitter > 0.0)
        s += jitter;
      for (i64 k = 0; k < j; ++k)
        s -= L[i * n + k] * L[j * n + k];
      if (i == j) {
        if (s <= pivot_tol)
          return SDSGE_NOT_PD;
        L[i * n + j] = sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  return SDSGE_OK;
}

/* out may alias b: out[i] is written only after reading b[i], and the loop
 * reads out[j] for j < i, which were already written this call. */
void sdsge_forward_subst(const f64 *SDSGE_RESTRICT L, const f64 *b, f64 *out,
                         i64 n) {
  for (i64 i = 0; i < n; ++i) {
    f64 s = 0.0;
    for (i64 j = 0; j < i; ++j)
      s += L[i * n + j] * out[j];
    out[i] = (b[i] - s) / L[i * n + i];
  }
}

/* out may alias b: descending i reads out[j] for j > i (already written) and
 * b[i] at the index about to be written. */
void sdsge_backward_subst_chol_t(const f64 *SDSGE_RESTRICT L, const f64 *b,
                                 f64 *out, i64 n) {
  for (i64 i = n - 1; i >= 0; --i) {
    f64 s = 0.0;
    for (i64 j = i + 1; j < n; ++j)
      s += L[j * n + i] * out[j];
    out[i] = (b[i] - s) / L[i * n + i];
  }
}

void sdsge_gram(const f64 *SDSGE_RESTRICT X, f64 *SDSGE_RESTRICT G, i64 n,
                i64 p) {
  sdsge_zero_mat(G, p, p);
  /* Lower triangle, accumulated row-by-row to match the numba xtx_xty order. */
  for (i64 r = 0; r < n; ++r) {
    const f64 *Xr = X + r * p;
    for (i64 a = 0; a < p; ++a) {
      f64 xra = Xr[a];
      for (i64 b = 0; b <= a; ++b)
        G[a * p + b] += xra * Xr[b];
    }
  }
  /* Mirror to full symmetric. */
  for (i64 a = 0; a < p; ++a)
    for (i64 b = 0; b < a; ++b)
      G[b * p + a] = G[a * p + b];
}

void sdsge_gram_rhs(const f64 *SDSGE_RESTRICT X, const f64 *SDSGE_RESTRICT y,
                    f64 *SDSGE_RESTRICT g, i64 n, i64 p) {
  for (i64 a = 0; a < p; ++a)
    g[a] = 0.0;
  for (i64 r = 0; r < n; ++r) {
    const f64 *Xr = X + r * p;
    f64 yr = y[r];
    for (i64 a = 0; a < p; ++a)
      g[a] += Xr[a] * yr;
  }
}

int sdsge_chol_solve(const f64 *SDSGE_RESTRICT G, const f64 *SDSGE_RESTRICT g,
                     f64 *SDSGE_RESTRICT coef, f64 *SDSGE_RESTRICT scratch_L,
                     i64 p) {
  int status = sdsge_chol(G, 0.0, scratch_L, p);
  if (status != SDSGE_OK)
    return status;
  /* coef holds z := L^-1 g, then is overwritten with L^-T z in place. */
  sdsge_forward_subst(scratch_L, g, coef, p);
  sdsge_backward_subst_chol_t(scratch_L, coef, coef, p);
  return SDSGE_OK;
}

int sdsge_chol_inv(const f64 *SDSGE_RESTRICT G, f64 *SDSGE_RESTRICT Pinv,
                   f64 *SDSGE_RESTRICT scratch_L, i64 p) {
  int status = sdsge_chol(G, 0.0, scratch_L, p);
  if (status != SDSGE_OK)
    return status;
  /* Solve (L L^T) x = e_j for each unit column j; column j of Pinv stores the
   * forward-substitution result z, then is overwritten in place with x. */
  for (i64 j = 0; j < p; ++j) {
    for (i64 i = 0; i < p; ++i) {
      f64 s = 0.0;
      for (i64 t = 0; t < i; ++t)
        s += scratch_L[i * p + t] * Pinv[t * p + j];
      f64 rhs = (i == j) ? 1.0 : 0.0;
      Pinv[i * p + j] = (rhs - s) / scratch_L[i * p + i];
    }
    for (i64 i = p - 1; i >= 0; --i) {
      f64 s = 0.0;
      for (i64 t = i + 1; t < p; ++t)
        s += scratch_L[t * p + i] * Pinv[t * p + j];
      Pinv[i * p + j] = (Pinv[i * p + j] - s) / scratch_L[i * p + i];
    }
  }
  return SDSGE_OK;
}

static int sdsge_cmp_f64(const void *a, const void *b) {
  f64 fa = *(const f64 *)a;
  f64 fb = *(const f64 *)b;
  return (fa > fb) - (fa < fb);
}

void sdsge_sort_f64(f64 *SDSGE_RESTRICT arr, i64 n) {
  if (n > 1)
    qsort(arr, (size_t)n, sizeof(f64), sdsge_cmp_f64);
}

f64 sdsge_median_f64(f64 *SDSGE_RESTRICT arr, i64 n) {
  if (n <= 0)
    return NAN;
  sdsge_sort_f64(arr, n);
  if (n % 2 == 0)
    return 0.5 * (arr[n / 2 - 1] + arr[n / 2]);
  return arr[n / 2];
}

i64 sdsge_lu_factor_inplace(f64 *SDSGE_RESTRICT A, i64 *SDSGE_RESTRICT pivot,
                            const i64 n) {
  if (!A || !pivot || n <= 0) {
    return SDSGE_LU_ALLOC_FAIL;
  }

  for (i64 k = 0; k < n; ++k) {
    i64 piv = k;
    f64 best = fabs(A[k * n + k]);

    for (i64 i = k + 1; i < n; ++i) {
      f64 v = fabs(A[i * n + k]);
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
        f64 tmp = A[k * n + j];
        A[k * n + j] = A[piv * n + j];
        A[piv * n + j] = tmp;
      }
    }

    f64 inv_Akk = 1.0 / A[k * n + k];

    for (i64 i = k + 1; i < n; ++i) {
      A[i * n + k] *= inv_Akk;

      f64 Lik = A[i * n + k];
      for (i64 j = k + 1; j < n; ++j) {
        A[i * n + j] -= Lik * A[k * n + j];
      }
    }
  }
  return SDSGE_LU_SUCCESS;
}

f64_lu sdsge_lu_factor(const f64 *SDSGE_RESTRICT A, const i64 n) {
  f64_lu result;
  result.lu = NULL;
  result.piv = NULL;
  result.n = n;
  result.err = SDSGE_LU_SUCCESS;

  if (!A || n <= 0) {
    result.err = SDSGE_LU_ALLOC_FAIL;
    return result;
  }

  result.lu = (f64 *)malloc(sizeof(f64) * n * n);
  result.piv = (i64 *)malloc(sizeof(i64) * n);
  if (!result.lu || !result.piv) {
    result.err = SDSGE_LU_ALLOC_FAIL;
    free(result.lu);
    free(result.piv);
    result.lu = NULL;
    result.piv = NULL;
    return result;
  }

  memcpy(result.lu, A, sizeof(f64) * n * n);
  result.err = sdsge_lu_factor_inplace(result.lu, result.piv, n);
  return result;
}

void sdsge_lu_free(f64_lu *lu) {
  if (!lu) {
    return;
  }
  free(lu->lu);
  free(lu->piv);
  lu->lu = NULL;
  lu->piv = NULL;
  lu->n = 0;
  lu->err = 0;
}

void sdsge_lu_solve(const f64 *SDSGE_RESTRICT LU, const i64 *SDSGE_RESTRICT piv,
                    const f64 *SDSGE_RESTRICT B, f64 *SDSGE_RESTRICT X,
                    const i64 n, const i64 m) {
  memcpy(X, B, sizeof(f64) * n * m);
  for (i64 k = 0; k < n; ++k) {
    i64 pk = piv[k];
    if (pk != k) {
      for (i64 j = 0; j < m; ++j) {
        f64 tmp = X[k * m + j];
        X[k * m + j] = X[pk * m + j];
        X[pk * m + j] = tmp;
      }
    }
  }

  for (i64 i = 0; i < n; ++i) {
    for (i64 k = 0; k < i; ++k) {
      f64 Lik = LU[i * n + k];
      for (i64 j = 0; j < m; ++j) {
        X[i * m + j] -= Lik * X[k * m + j];
      }
    }
  }

  for (i64 i = n - 1; i >= 0; --i) {
    for (i64 k = i + 1; k < n; ++k) {
      f64 Uik = LU[i * n + k];
      for (i64 j = 0; j < m; ++j) {
        X[i * m + j] -= Uik * X[k * m + j];
      }
    }
    f64 inv_Uii = 1.0 / LU[i * n + i];
    for (i64 j = 0; j < m; ++j) {
      X[i * m + j] *= inv_Uii;
    }
  }
}

i64 sdsge_solve(const f64 *SDSGE_RESTRICT A, const f64 *SDSGE_RESTRICT B,
                const i64 n, const i64 m, f64 *SDSGE_RESTRICT X) {
  f64_lu lu = sdsge_lu_factor(A, n);
  if (lu.err != SDSGE_LU_SUCCESS) {
    i64 err = lu.err;
    sdsge_lu_free(&lu);
    return err;
  }
  sdsge_lu_solve(lu.lu, lu.piv, B, X, n, m);
  sdsge_lu_free(&lu);
  return SDSGE_LU_SUCCESS;
}

i64 sdsge_inv(const f64 *SDSGE_RESTRICT A, const i64 n,
              f64 *SDSGE_RESTRICT Ainv) {
  f64 *I = (f64 *)malloc(sizeof(f64) * n * n);
  if (!I) {
    return SDSGE_LU_ALLOC_FAIL;
  }

  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < n; ++j) {
      I[i * n + j] = (i == j) ? 1.0 : 0.0;
    }
  }

  i64 err = sdsge_solve(A, I, n, n, Ainv);
  free(I);
  return err;
}
