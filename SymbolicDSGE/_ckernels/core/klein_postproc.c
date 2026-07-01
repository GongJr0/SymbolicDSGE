#include "klein_postproc.h"
#include <stdlib.h>

i64 klein_postproc(const c128 *SDSGE_RESTRICT s, const c128 *SDSGE_RESTRICT t,
                   const c128 *SDSGE_RESTRICT z, const i64 n_s, const i64 n_cs,
                   c128 *SDSGE_RESTRICT f, c128 *SDSGE_RESTRICT p,
                   i64 *SDSGE_RESTRICT stab, c128 *SDSGE_RESTRICT eig) {
  i64 N = n_s + n_cs;

  /* A model with no states has no Klein solution. Fail fast before any
   * allocation -- the state/inv/solve routines all assume n_s >= 1. */
  if (n_s <= 0) {
    return SDSGE_KLEIN_POSTPROC_INVALID;
  }

  /* Allocate z11, z21, s11, t11 */
  c128 *SDSGE_RESTRICT z11 = (c128 *)malloc(sizeof(c128) * n_s * n_s);
  c128 *SDSGE_RESTRICT z21 = (c128 *)malloc(sizeof(c128) * n_cs * n_s);
  c128 *SDSGE_RESTRICT s11 = (c128 *)malloc(sizeof(c128) * n_s * n_s);
  c128 *SDSGE_RESTRICT t11 = (c128 *)malloc(sizeof(c128) * n_s * n_s);
  c128 *SDSGE_RESTRICT z11i = (c128 *)malloc(sizeof(c128) * n_s * n_s);

  c128 *SDSGE_RESTRICT dyn = (c128 *)malloc(sizeof(c128) * n_s * n_s);
  c128 *SDSGE_RESTRICT tmp = (c128 *)malloc(sizeof(c128) * n_s * n_s);

  if (!z11 || !z21 || !s11 || !t11 || !z11i || !dyn || !tmp) {
    free(z11);
    free(z21);
    free(s11);
    free(t11);
    free(z11i);
    free(dyn);
    free(tmp);
    return SDSGE_KLEIN_POSTPROC_ALLOC_FAIL;
  }

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

  /* Invert z11. c128_inv returns SDSGE_LU_SINGULAR when z11 is singular (a
   * Blanchard-Kahn failure), or SDSGE_LU_ALLOC_FAIL on OOM. */
  i64 err_z = c128_inv(z11, n_s, z11i);
  if (err_z != SDSGE_LU_SUCCESS) {
    free(z11);
    free(z21);
    free(s11);
    free(t11);
    free(z11i);
    free(dyn);
    free(tmp);
    return (err_z == SDSGE_LU_SINGULAR) ? SDSGE_KLEIN_POSTPROC_SINGULAR
                                        : SDSGE_KLEIN_POSTPROC_ALLOC_FAIL;
  }

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

  /* dyn = solve(s11, t11). Singular s11 is again a Blanchard-Kahn failure. */
  i64 dyn_err = c128_solve(s11, t11, n_s, n_s, dyn);
  if (dyn_err != SDSGE_LU_SUCCESS) {
    free(z11);
    free(z21);
    free(s11);
    free(t11);
    free(z11i);
    free(dyn);
    free(tmp);
    return (dyn_err == SDSGE_LU_SINGULAR) ? SDSGE_KLEIN_POSTPROC_SINGULAR
                                          : SDSGE_KLEIN_POSTPROC_ALLOC_FAIL;
  }

  c128_matmul(z21, z11i, n_cs, n_s, n_s, f);
  c128_matmul(z11, dyn, n_s, n_s, n_s, tmp);
  c128_matmul(tmp, z11i, n_s, n_s, n_s, p);

  free(z11);
  free(z21);
  free(s11);
  free(t11);
  free(z11i);
  free(dyn);
  free(tmp);

  return SDSGE_KLEIN_POSTPROC_SUCCESS;
}
