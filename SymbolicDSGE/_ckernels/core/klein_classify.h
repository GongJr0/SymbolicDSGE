#ifndef SDSGE_KLEIN_CLASSIFY_H
#define SDSGE_KLEIN_CLASSIFY_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"

/* LAPACK ztgexc, reached through a runtime function-pointer address (pulled
 * from scipy.linalg.cython_lapack.__pyx_capi__['ztgexc'] on the Python side),
 * so this translation unit links against no LAPACK at build time. All INTEGER
 * arguments are 32-bit `int` (LAPACK default INTEGER; scipy's cython_lapack is
 * not ILP64). */
typedef void (*sdsge_ztgexc_fn)(const int *wantq, const int *wantz,
                                const int *n, c128 *a, const int *lda, c128 *b,
                                const int *ldb, c128 *q, const int *ldq,
                                c128 *z, const int *ldz, const int *ifst,
                                int *ilst, int *info);

/* Whether the generalized eigenvalue at diagonal slot `i` of an n by n Schur
 * pencil is finite. The eig fill and the leading-block screen are the same
 * question asked twice, so both ask it here and a NaN magnitude reports
 * infinite in each. The diagonal sits at i*n + i under either layout, which is
 * what lets the callers run before the transposes. */
static inline int klein_root_finite(const c128 *SDSGE_RESTRICT s, const i64 i,
                                    const i64 n) {
  return c128_abs(s[i * n + i]) > 1e-12;
}

/* Diagonals only, so this may run on either layout. */
static inline void klein_fill_eig(const c128 *SDSGE_RESTRICT s,
                                  const c128 *SDSGE_RESTRICT t,
                                  c128 *SDSGE_RESTRICT eig, const i64 n) {
  for (i64 i = 0; i < n; ++i) {
    if (klein_root_finite(s, i, n)) {
      eig[i] = c128_div(t[i * n + i], s[i * n + i]);
    } else {
      eig[i] = c128_make(INFINITY, 0.0);
    }
  }
}

static inline void klein_fill_eye(c128 *SDSGE_RESTRICT eye, const i64 n) {
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < n; ++j) {
      eye[i * n + j] = c128_make(i == j ? 1.0 : 0.0, 0.0);
    }
  }
}

/* z11 (row-major, out of a column-major z) and z11i = z11^-1 for the leading
 * n_s block. OK when the rule can be read off it, INFINITE_ROOT when a selected
 * root is not finite, RANK_FAIL when z11 is singular or its inverse carries no
 * digits.
 *
 * `eye` arrives prefilled as identity. */
i64 klein_z11_pair(const c128 *SDSGE_RESTRICT s, const c128 *SDSGE_RESTRICT z,
                   const i64 nd, const i64 n_s, c128 *SDSGE_RESTRICT z11,
                   c128 *SDSGE_RESTRICT z11i, c128 *SDSGE_RESTRICT tmp,
                   const c128 *SDSGE_RESTRICT eye, i64 *SDSGE_RESTRICT piv,
                   f64 *SDSGE_RESTRICT rcond);

/* Scratch for klein_reorder_argmax: the pristine s/t/z it restores each
 * candidate from, and the candidate's own z11 pair. */
arena_size klein_classify_arena_size(i64 nd, i64 n_s);

/* Best-conditioned leading n_s block over the single swaps of the sdim stable
 * roots, by ztgexc. Assumes sdim > n_s and a live ztgexc; the caller owns that
 * branch. Leaves s/t/z in the winning configuration with z11/z11i holding its
 * pair, and returns the unsearched block's verdict when nothing beats it.
 *
 * The swap that evicts leading slot k for excess slot j is two exchanges, both
 * landing on n_s: one call only ever reaches slot n_s, since the blocks it
 * walks past shift back one each. */
i64 klein_reorder_argmax(sdsge_ztgexc_fn ztgexc, c128 *SDSGE_RESTRICT s,
                         c128 *SDSGE_RESTRICT t, c128 *SDSGE_RESTRICT z,
                         const i64 nd, const i64 n_s, const i64 sdim,
                         c128 *SDSGE_RESTRICT z11, c128 *SDSGE_RESTRICT z11i,
                         c128 *SDSGE_RESTRICT tmp,
                         const c128 *SDSGE_RESTRICT eye,
                         i64 *SDSGE_RESTRICT piv, f64 *SDSGE_RESTRICT arena);

#define SDSGE_KLEIN_OK 0
#define SDSGE_KLEIN_RANK_FAIL -301
#define SDSGE_KLEIN_INFINITE_ROOT -302

#endif // !SDSGE_KLEIN_CLASSIFY_H
