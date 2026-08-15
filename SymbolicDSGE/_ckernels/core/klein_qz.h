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
 * are 32-bit `int` (LAPACK default INTEGER; scipy's cython_lapack is not
 * ILP64); `bwork` is a Fortran LOGICAL array, i.e. `int*`. */
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
 * into them. alpha/beta/rwork/work come off `arena`, bwork off `iarena`.
 *
 * Returns KLEIN_QZ_OK or KLEIN_QZ_LAPACK_FAIL (zgges info != 0). n == 0 is a
 * no-op returning KLEIN_QZ_OK. */
arena_size klein_qz_arena_size(i64 n);

i64 klein_qz(klein_zgges_fn zgges, i64 n, c128 *SDSGE_RESTRICT s,
             c128 *SDSGE_RESTRICT t, c128 *SDSGE_RESTRICT z,
             f64 *SDSGE_RESTRICT arena, i64 *SDSGE_RESTRICT iarena);

#define KLEIN_QZ_OK 0
#define KLEIN_QZ_LAPACK_FAIL -401

/* Provisioned zgges complex workspace, in units of n. LAPACK's documented
 optimum is `n*(nb+1)` for block size nb; 64 is the largest `nb` shipped by
 reference builds. We allocate the entire n*(64 + 1) unconditionally as the
 maximum possible `nb` encountered in the reference spec. The arena space is
 therefore guaranteed to be runtime-optimal but have no memory-optimality
 guarantees. */
#define KLEIN_QZ_LWORK_PER_N 65

#endif /* KLEIN_QZ_H */
