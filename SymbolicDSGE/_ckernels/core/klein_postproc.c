#include "klein_postproc.h"
#include <string.h>

/* ||A||_1 of a row-major n by n matrix: the largest absolute column sum. The
 * column walk strides a row at a time, which costs nothing at these sizes: the
 * matrix is resident after the first pass, and the two calls sit beside three
 * n^3 products. Accumulating row-major would buy a scratch parameter and the
 * liveness of the caller with it. */
static inline f64 c128_norm1(const c128 *SDSGE_RESTRICT A, const i64 n) {
  f64 best = 0.0;
  for (i64 j = 0; j < n; ++j) {
    f64 sum = 0.0;
    for (i64 i = 0; i < n; ++i) {
      sum += c128_abs(A[i * n + j]);
    }
    if (sum > best) {
      best = sum;
    }
  }
  return best;
}

arena_size klein_postproc_arena_size(const i64 n_s, const i64 n_cs) {
  const i64 sq = n_s * n_s;
  return make_sizer(2 * (7 * sq         /* z11, s11, t11, z11i, dyn, eye, tmp */
                         + n_cs * n_s), /* z21 */
                    n_s /* LU pivot */);
}

i64 klein_postproc(const c128 *SDSGE_RESTRICT s, const c128 *SDSGE_RESTRICT t,
                   const c128 *SDSGE_RESTRICT z, const i64 n_s, const i64 n_cs,
                   c128 *SDSGE_RESTRICT f, c128 *SDSGE_RESTRICT p,
                   i64 *SDSGE_RESTRICT stab, c128 *SDSGE_RESTRICT eig,
                   f64 *SDSGE_RESTRICT arena, i64 *SDSGE_RESTRICT iarena) {
  i64 N = n_s + n_cs;

  /* Stamp the stab sentinel up front: the verdict below is the last thing this
   * function writes, so every early return leaves a non-zero (undetermined)
   * stab rather than a stale value, and a caller that reads stab without
   * checking the return code still sees a violation. */
  *stab = SDSGE_KLEIN_STAB_UNSET;

  const i64 sq = n_s * n_s;
  c128 *SDSGE_RESTRICT z11 = (c128 *)arena;
  c128 *SDSGE_RESTRICT z21 = z11 + sq;
  c128 *SDSGE_RESTRICT s11 = z21 + (n_cs * n_s);
  c128 *SDSGE_RESTRICT t11 = s11 + sq;
  c128 *SDSGE_RESTRICT z11i = t11 + sq;
  c128 *SDSGE_RESTRICT dyn = z11i + sq;
  c128 *SDSGE_RESTRICT eye = dyn + sq;
  c128 *SDSGE_RESTRICT tmp = eye + sq;

  /* Fill z11, s11, t11 */
  for (i64 i = 0; i < n_s; ++i) {
    for (i64 j = 0; j < n_s; ++j) {
      z11[i * n_s + j] = z[i * N + j];
      s11[i * n_s + j] = s[i * N + j];
      t11[i * n_s + j] = t[i * N + j];
    }
  }

  /* Fill z21 */
  for (i64 i = 0; i < n_cs; ++i) {
    for (i64 j = 0; j < n_s; ++j) {
      z21[i * n_s + j] = z[(n_s + i) * N + j];
    }
  }

  /* eig[i] = t[i,i] / s[i,i] */
  for (i64 i = 0; i < N; ++i) {
    if (c128_abs(s[i * N + i]) > 1e-12) {
      eig[i] = c128_div(t[i * N + i], s[i * N + i]);
    } else {
      eig[i] = c128_make(INFINITY, 0.0);
    }
  }

  if (c128_abs(t[(n_s - 1) * N + (n_s - 1)]) >
      c128_abs(s[(n_s - 1) * N + (n_s - 1)])) {
    return SDSGE_KLEIN_NO_STABLE_SOLUTION; /* Too Few stable eigenvalues */
  }

  /* An s11 diagonal entry at the eig loop's bound is the singular s11 the dyn
   * solve would otherwise factor. Written as that test negated, so the two
   * cannot drift apart and a NAN cannot pass either. */
  for (i64 i = 0; i < n_s; ++i) {
    if (!(c128_abs(s[i * N + i]) > 1e-12)) {
      return SDSGE_KLEIN_POSTPROC_INFINITE_ROOT;
    }
  }

  /* z11i = z11^-1. z11 is a factor of p below, so the LU takes a copy, held in
   * tmp until the product needs that buffer. */
  memcpy(tmp, z11, sq * sizeof(c128));

  for (i64 i = 0; i < n_s; ++i) {
    for (i64 j = 0; j < n_s; ++j) {
      eye[i * n_s + j] = c128_make(i == j ? 1.0 : 0.0, 0.0);
    }
  }
  if (c128_lu_factor_inplace(tmp, iarena, n_s) != SDSGE_LU_SUCCESS) {
    return SDSGE_KLEIN_POSTPROC_RANK_FAIL;
  }
  c128_lu_solve(tmp, iarena, eye, z11i, n_s, n_s);

  /* A z11 the factorization accepted but whose inverse carries no digits. */
  const f64 rcond = 1.0 / (c128_norm1(z11, n_s) * c128_norm1(z11i, n_s));
  if (!(rcond > 1e-9)) {
    return SDSGE_KLEIN_POSTPROC_RANK_FAIL;
  }

  /* dyn = solve(s11, t11). s11 is dead after this, so it factors in place.
   * An upper triangular block pivots on its own diagonal, so the gate above has
   * already rejected what this arm could catch. */
  if (c128_lu_factor_inplace(s11, iarena, n_s) != SDSGE_LU_SUCCESS) {
    return SDSGE_KLEIN_POSTPROC_INFINITE_ROOT;
  }
  c128_lu_solve(s11, iarena, t11, dyn, n_s, n_s);

  c128_matmul(z21, z11i, n_cs, n_s, n_s, f);
  c128_matmul(z11, dyn, n_s, n_s, n_s, tmp);
  c128_matmul(tmp, z11i, n_s, n_s, n_s, p);

  *stab = 0;
  if (n_s < N) {
    if (c128_abs(t[n_s * N + n_s]) < c128_abs(s[n_s * N + n_s])) {
      /* Too Many stable eigenvalues.
       * The solution remains valid, having cleared the singularity and rcond
       * tests. It is a BK failure that the user can choose to ignore without
       * hindering any following library interaction. */
      *stab = 1;
    }
  }
  return SDSGE_KLEIN_POSTPROC_SUCCESS;
}
