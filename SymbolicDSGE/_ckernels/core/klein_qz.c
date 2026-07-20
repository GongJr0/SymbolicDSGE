#include "klein_qz.h"

#include <stdlib.h>

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

i64 klein_qz(klein_zgges_fn zgges, i64 n, c128 *SDSGE_RESTRICT s,
             c128 *SDSGE_RESTRICT t, c128 *SDSGE_RESTRICT z) {
  if (n == 0) {
    return KLEIN_QZ_OK;
  }

  const int n32 = (int)n;

  /* jobvsl = "N": VSL (left Schur vectors) is not needed downstream; skipping
   * it leaves S/T/Z identical to the jobvsl="V" path. jobvsr = "V" computes Z;
   * sort = "S" applies the ouc ordering via the selctg callback. */
  const char jobvsl = 'N';
  const char jobvsr = 'V';
  const char sort = 'S';
  int sdim = 0;
  int info = 0;
  int ldvsl = 1;
  c128 vsl_dummy = c128_make(0.0, 0.0); /* not referenced when jobvsl = 'N' */

  c128 *alpha = (c128 *)malloc((size_t)n * sizeof(c128));
  c128 *beta = (c128 *)malloc((size_t)n * sizeof(c128));
  f64 *rwork = (f64 *)malloc((size_t)(8 * n) * sizeof(f64));
  int *bwork = (int *)malloc((size_t)n * sizeof(int));
  if (alpha == NULL || beta == NULL || rwork == NULL || bwork == NULL) {
    free(alpha);
    free(beta);
    free(rwork);
    free(bwork);
    return KLEIN_QZ_ALLOC_FAIL;
  }

  /* Workspace query (lwork = -1): zgges writes the optimal complex work size to
   * wq.re. */
  c128 wq = c128_make(0.0, 0.0);
  int lwork = -1;
  zgges(&jobvsl, &jobvsr, &sort, &klein_ouc, &n32, s, &n32, t, &n32, &sdim,
        alpha, beta, &vsl_dummy, &ldvsl, z, &n32, &wq, &lwork, rwork, bwork,
        &info);

  lwork = (int)wq.re;
  if (lwork < 1) {
    lwork = 1;
  }
  c128 *work = (c128 *)malloc((size_t)lwork * sizeof(c128));
  if (work == NULL) {
    free(alpha);
    free(beta);
    free(rwork);
    free(bwork);
    return KLEIN_QZ_ALLOC_FAIL;
  }

  zgges(&jobvsl, &jobvsr, &sort, &klein_ouc, &n32, s, &n32, t, &n32, &sdim,
        alpha, beta, &vsl_dummy, &ldvsl, z, &n32, work, &lwork, rwork, bwork,
        &info);

  free(work);
  free(alpha);
  free(beta);
  free(rwork);
  free(bwork);

  return (info != 0) ? KLEIN_QZ_LAPACK_FAIL : KLEIN_QZ_OK;
}
