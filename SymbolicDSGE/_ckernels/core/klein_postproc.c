#include "klein_postproc.h"

arena_size klein_postproc_arena_size(const i64 n_s, const i64 n_cs) {
  const i64 sq = n_s * n_s;
  return make_sizer(2 * (6 * sq        /* z11, s11, t11, z11i, dyn, tmp */
                         + n_cs * n_s  /* z21 */
                         + sq          /* LU factor copy */
                         + sq),        /* identity RHS for the inverse */
                    n_s /* LU pivot */);
}

i64 klein_postproc(const c128 *SDSGE_RESTRICT s, const c128 *SDSGE_RESTRICT t,
                   const c128 *SDSGE_RESTRICT z, const i64 n_s, const i64 n_cs,
                   c128 *SDSGE_RESTRICT f, c128 *SDSGE_RESTRICT p,
                   i64 *SDSGE_RESTRICT stab, c128 *SDSGE_RESTRICT eig,
                   f64 *SDSGE_RESTRICT arena, i64 *SDSGE_RESTRICT iarena) {
  i64 N = n_s + n_cs;

  /* Stamp the stab sentinel up front: every early-return failure below leaves a
   * non-zero (undetermined) stab, never a stale value, so a caller that reads
   * stab without checking the return code still sees a violation. Overwritten
   * with the real 0 / -1 / +1 on the successful path. */
  *stab = SDSGE_KLEIN_STAB_UNSET;

  /* A model with no states has no Klein solution. Fail fast: the state/inv/solve
   * routines all assume n_s >= 1. */
  if (n_s <= 0) {
    return SDSGE_KLEIN_POSTPROC_INVALID;
  }

  const i64 sq = n_s * n_s;
  c128 *cp = (c128 *)arena;
  c128 *SDSGE_RESTRICT z11 = cp;
  cp += sq;
  c128 *SDSGE_RESTRICT z21 = cp;
  cp += n_cs * n_s;
  c128 *SDSGE_RESTRICT s11 = cp;
  cp += sq;
  c128 *SDSGE_RESTRICT t11 = cp;
  cp += sq;
  c128 *SDSGE_RESTRICT z11i = cp;
  cp += sq;
  c128 *SDSGE_RESTRICT dyn = cp;
  cp += sq;
  c128 *SDSGE_RESTRICT tmp = cp;
  cp += sq;
  c128 *SDSGE_RESTRICT lu = cp;
  cp += sq;
  c128 *SDSGE_RESTRICT eye = cp;

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

  /* z11i = z11^-1, factored out of place because z11 is still needed below. A
   * singular z11 is a Blanchard-Kahn failure. */
  for (i64 i = 0; i < sq; ++i) {
    lu[i] = z11[i];
  }
  for (i64 i = 0; i < n_s; ++i) {
    for (i64 j = 0; j < n_s; ++j) {
      eye[i * n_s + j] = c128_make(i == j ? 1.0 : 0.0, 0.0);
    }
  }
  if (c128_lu_factor_inplace(lu, iarena, n_s) != SDSGE_LU_SUCCESS) {
    return SDSGE_KLEIN_POSTPROC_SINGULAR;
  }
  c128_lu_solve(lu, iarena, eye, z11i, n_s, n_s);

  *stab = 0;
  if (c128_abs(t[(n_s - 1) * N + (n_s - 1)]) >
      c128_abs(s[(n_s - 1) * N + (n_s - 1)])) {
    *stab = -1; /* Too Few stable eigenvalues */
  }

  if (n_s < N) {
    if (c128_abs(t[n_s * N + n_s]) < c128_abs(s[n_s * N + n_s])) {
      *stab = 1; /* Too Many stable eigenvalues */
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

  /* dyn = solve(s11, t11). s11 is dead after this, so it factors in place.
   * Singular s11 is again a Blanchard-Kahn failure. */
  if (c128_lu_factor_inplace(s11, iarena, n_s) != SDSGE_LU_SUCCESS) {
    return SDSGE_KLEIN_POSTPROC_SINGULAR;
  }
  c128_lu_solve(s11, iarena, t11, dyn, n_s, n_s);

  c128_matmul(z21, z11i, n_cs, n_s, n_s, f);
  c128_matmul(z11, dyn, n_s, n_s, n_s, tmp);
  c128_matmul(tmp, z11i, n_s, n_s, n_s, p);

  return SDSGE_KLEIN_POSTPROC_SUCCESS;
}
