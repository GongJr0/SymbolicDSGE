#include "klein_qz.h"

/* Klein 'ouc' selection: select |alpha/beta| > 1, i.e. the generalized
 * eigenvalue lies outside the unit circle. Division-safe magnitude compare
 * (|alpha|^2 > |beta|^2); beta == 0 (infinite eigenvalue) selects true, as it
 * must. Fortran LOGICAL return: nonzero (true) / zero (false). Ported verbatim
 * from `_klein_ouc` in _core.pyx. */
static int klein_ouc(const c128 *alpha, const c128 *beta) {
  const f64 aa = alpha->re * alpha->re + alpha->im * alpha->im;
  const f64 bb = beta->re * beta->re + beta->im * beta->im;
  return aa > bb;
}

arena_size klein_qz_arena_size(const i64 n) {
  return make_sizer(2 * n + 2 * n                       /* alpha, beta */
                        + 8 * n                         /* rwork */
                        + 2 * KLEIN_QZ_LWORK_PER_N * n, /* work */
                    (n + 1) / 2 /* bwork, n Fortran LOGICALs */);
}

i64 klein_qz(klein_zgges_fn zgges, i64 n, c128 *SDSGE_RESTRICT s,
             c128 *SDSGE_RESTRICT t, c128 *SDSGE_RESTRICT z,
             i64 *SDSGE_RESTRICT sdim, f64 *SDSGE_RESTRICT arena,
             i64 *SDSGE_RESTRICT iarena) {
  if (n == 0) {
    *sdim = 0;
    return KLEIN_QZ_OK;
  }

  const int n32 = (int)n;

  /* jobvsl = "N": VSL (left Schur vectors) is not needed downstream; skipping
   * it leaves S/T/Z identical to the jobvsl="V" path. jobvsr = "V" computes Z;
   * sort = "S" applies the ouc ordering via the selctg callback. */
  const char jobvsl = 'N';
  const char jobvsr = 'V';
  const char sort = 'S';
  int info = 0;
  int ldvsl = 1;
  c128 vsl_dummy = c128_make(0.0, 0.0); /* not referenced when jobvsl = 'N' */

  c128 *cp = (c128 *)arena;
  c128 *alpha = cp;
  cp += n;
  c128 *beta = cp;
  cp += n;
  f64 *rwork = (f64 *)cp;
  c128 *work = (c128 *)(rwork + 8 * n);
  int *bwork = (int *)iarena;
  int sdim32 = 0;
  const int lwork = (int)(KLEIN_QZ_LWORK_PER_N * n);

  zgges(&jobvsl, &jobvsr, &sort, &klein_ouc, &n32, s, &n32, t, &n32, &sdim32,
        alpha, beta, &vsl_dummy, &ldvsl, z, &n32, work, &lwork, rwork, bwork,
        &info);
  *sdim = (i64)sdim32;
  return (info != 0) ? KLEIN_QZ_LAPACK_FAIL : KLEIN_QZ_OK;
}
