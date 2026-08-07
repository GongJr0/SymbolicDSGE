#ifndef SDSGE_KLEIN_PREPROC_H
#define SDSGE_KLEIN_PREPROC_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"

typedef void (*sdsge_residual_fn)(const c128 *fwd, const c128 *cur,
                                  const c128 *par, c128 *out);

arena_size klein_preproc_arena_size(i64 n_var, i64 n_par, i64 n_eq);

/* Complex-step pencil a = dF/dfwd, b = -dF/dcur at `ss`. Total: the sweep has no
 * failure mode once its scratch comes from the caller. */
void klein_preproc(sdsge_residual_fn residual, const f64 *SDSGE_RESTRICT ss,
                   const f64 *SDSGE_RESTRICT par, const i64 n_var,
                   const i64 n_par, const i64 n_eq, f64 *SDSGE_RESTRICT a,
                   f64 *SDSGE_RESTRICT b, f64 *SDSGE_RESTRICT arena);

#endif /* SDSGE_KLEIN_PREPROC_H */
