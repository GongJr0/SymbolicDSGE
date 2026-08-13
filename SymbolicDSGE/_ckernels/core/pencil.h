#ifndef SDSGE_PENCIL_H
#define SDSGE_PENCIL_H

#include "../_common/sdsge_common.h"

/* Structural incidence: which dates a variable occurs at, one i8 per variable.
 *
 * A property of the model, not of any solver. The partition below is one
 * algorithm's view of it; a quadratic-matrix-equation solver would derive a
 * different grouping from the same array, which is why the incidence rather
 * than the grouping is what crosses the boundary.
 *
 * It must be the union over the reference equations and every regime's
 * replacements. A regime that drops the only occurrence of some `v(t+1)` would
 * otherwise classify `v` differently from the reference, and the regime pencils
 * are stacked by bitmask into one array that admits a single shape. */
#define SDSGE_INC_LAG 1  /* occurs at t-1 */
#define SDSGE_INC_CUR 2  /* occurs at t   */
#define SDSGE_INC_LEAD 4 /* occurs at t+1 */

/* Size of the pencil the incidence implies: `ndynamic + n_both`.
 *
 * `n_var` does not bound this. A variable carrying both a lag and a lead needs
 * a companion row, so an all-mixed model reaches `2 * n_var`. Callers own the
 * pencil, Schur and eigenvalue buffers, so they size them from here. */
i64 sdsge_pencil_dim(const i8 *SDSGE_RESTRICT incidence, i64 n_var);

/* Classify every variable and write the decision-rule ordering.
 *
 * `order` is `n_var` long and holds original indices grouped
 * `[static | pred | both | fwd]`, Dynare's `order_var`, stable within each
 * group. Every index set the assembly needs is a slice of it given the counts,
 * so none of them are stored.
 *
 * Only the outer two bits classify: a variable with neither a lag nor a lead is
 * static whatever it does at `t`. `SDSGE_INC_CUR` is what makes a variable
 * occurring at no date at all distinguishable from one occurring only at `t`,
 * and that is an absent variable rather than a static one. */
i64 sdsge_pencil_partition(const i8 *SDSGE_RESTRICT incidence, i64 n_var,
                           i64 *SDSGE_RESTRICT order,
                           i64 *SDSGE_RESTRICT n_static,
                           i64 *SDSGE_RESTRICT n_pred,
                           i64 *SDSGE_RESTRICT n_both,
                           i64 *SDSGE_RESTRICT n_fwd);

/* LAPACK QR, reached through runtime function-pointer addresses pulled from
 * scipy.linalg.cython_lapack on the Python side, exactly as `klein_zgges_fn`
 * is, so this translation unit links against no LAPACK at build time. INTEGER
 * arguments are 32-bit `int`: scipy's cython_lapack is not ILP64. */
typedef void (*sdsge_dgeqrf_fn)(const int *m, const int *n, f64 *a,
                                const int *lda, f64 *tau, f64 *work,
                                const int *lwork, int *info);

typedef void (*sdsge_dormqr_fn)(const char *side, const char *trans,
                                const int *m, const int *n, const int *k,
                                const f64 *a, const int *lda, const f64 *tau,
                                f64 *c, const int *ldc, f64 *work,
                                const int *lwork, int *info);

/* Scratch `sdsge_pencil_rotate_static` needs for a shape. */
arena_size sdsge_pencil_rotate_arena_size(i64 n_var, i64 n_static, i64 n_cols);

/* Rotate the static equations to the top so the dynamic rows carry no static
 * column, in place across every block.
 *
 * `blocks` are `n_block` matrices of `(n_var, n_cols)` sharing one row space:
 * the Jacobian blocks and the shock loading, all rotated by the same Q so they
 * stay one system. The rotation is orthogonal on the equations, so the model it
 * describes is unchanged; it only puts the static block where the pencil can
 * skip it and the back-substitution can recover it.
 *
 * Q is never formed. The reflectors come from a QR of the static columns of
 * `b`, and `dormqr` applies Q' straight to each block. A row-major `(n_var,
 * n_cols)` buffer is a column-major `(n_cols, n_var)` view of its transpose, so
 * `Q' A` is `A^T Q` from LAPACK's side, which is `side='R', trans='N'` and needs
 * no transpose of the block itself.
 *
 * A no-op when `n_static == 0`. */
i64 sdsge_pencil_rotate_static(sdsge_dgeqrf_fn dgeqrf, sdsge_dormqr_fn dormqr,
                               const f64 *SDSGE_RESTRICT b,
                               const i64 *SDSGE_RESTRICT order, i64 n_var,
                               i64 n_static, f64 *const *blocks,
                               const i64 *n_cols, i64 n_block,
                               f64 *SDSGE_RESTRICT arena);

/* Assemble the generalized eigenvalue pencil `E z = D z'` the QZ runs on.
 *
 * Takes the Jacobian blocks in the kernel's own signs, `a = dF/dfwd`,
 * `b = -dF/dcur`, `c = -dF/dprev`, each `(n_var, n_var)` in canonical variable
 * order, and reads them through `order` so the pencil comes out in
 * decision-rule order. Both matrices are `nd = ndynamic + n_both` square, with
 * `nd` as `sdsge_pencil_dim` reports it.
 *
 * Column space is `[pred | both | both' | fwd]`: the leading `nspred =
 * n_pred + n_both` are the predetermined block the post-proc splits on, and the
 * trailing `nsfwrd = n_both + n_fwd` are the led block. A variable carrying both
 * a lag and a lead appears in each, tied together by the identity rows the
 * companion block contributes; when `n_both` is zero there is no companion block
 * and `nd` is just `ndynamic`.
 *
 * The static rows must already be rotated out: rows `[n_static, n_var)` of the
 * blocks are the ones read, which is only the dynamic system once a QR has
 * cleared the static columns from them. With `n_static == 0` that is every row
 * and no rotation is needed. */
void sdsge_pencil_assemble(const f64 *SDSGE_RESTRICT a,
                           const f64 *SDSGE_RESTRICT b,
                           const f64 *SDSGE_RESTRICT c,
                           const i64 *SDSGE_RESTRICT order, i64 n_var,
                           i64 n_static, i64 n_pred, i64 n_both, i64 n_fwd,
                           f64 *SDSGE_RESTRICT E, f64 *SDSGE_RESTRICT D);

/* ERROR CODES */
#define SDSGE_PENCIL_OK 0
#define SDSGE_PENCIL_ABSENT_VAR -601 /* a variable occurs at no date */
#define SDSGE_PENCIL_QR_FAIL -602    /* LAPACK dgeqrf/dormqr info != 0 */

#endif /* SDSGE_PENCIL_H */
