#ifndef KLEIN_QZ_H
#define KLEIN_QZ_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"

/* Klein 'outside unit circle' selctg for zgges: LOGICAL FUNCTION SELCTG(ALPHA,
 * BETA), receiving two complex*16 by pointer and returning a Fortran LOGICAL
 * (int). Matches the pyx `_klein_ouc`. */
typedef int (*klein_zselect2_fn)(const c128 *alpha, const c128 *beta);

/* LAPACK zgges, reached through a runtime function-pointer address (pulled from
 * scipy.linalg.cython_lapack.__pyx_capi__['zgges'] on the Python side), so this
 * translation unit links against no LAPACK at build time. All INTEGER arguments
 * are 32-bit `int` (LAPACK default INTEGER; scipy's cython_lapack is not ILP64);
 * `bwork` is a Fortran LOGICAL array, i.e. `int*`. */
typedef void (*klein_zgges_fn)(const char *jobvsl, const char *jobvsr,
                               const char *sort, klein_zselect2_fn selctg,
                               const int *n, c128 *a, const int *lda, c128 *b,
                               const int *ldb, int *sdim, c128 *alpha,
                               c128 *beta, c128 *vsl, const int *ldvsl,
                               c128 *vsr, const int *ldvsr, c128 *work,
                               const int *lwork, f64 *rwork, int *bwork,
                               int *info);

/* Native generalized Schur (QZ) with the Klein 'ouc' ordering, equivalent to
 * scipy.linalg.ordqz(a, b, sort="ouc", output="complex") indices [0, 1, 5].
 *
 * Buffers are column-major (Fortran order), n*n complex128 each:
 *   s : IN  the A pencil        -> OUT ordered Schur factor S
 *   t : IN  the B pencil        -> OUT ordered Schur factor T
 *   z : OUT right Schur vectors Z  (need not be initialized)
 * s and t are overwritten in place; the caller materializes the complex pencil
 * into them. All scratch (alpha/beta/rwork/bwork/work) is allocated internally.
 *
 * Returns KLEIN_QZ_OK, KLEIN_QZ_ALLOC_FAIL, or KLEIN_QZ_LAPACK_FAIL (zgges
 * info != 0). n == 0 is a no-op returning KLEIN_QZ_OK. */
i64 klein_qz(klein_zgges_fn zgges, i64 n, c128 *SDSGE_RESTRICT s,
             c128 *SDSGE_RESTRICT t, c128 *SDSGE_RESTRICT z);

#define KLEIN_QZ_OK 0
#define KLEIN_QZ_ALLOC_FAIL -1
#define KLEIN_QZ_LAPACK_FAIL -2

#endif /* KLEIN_QZ_H */
