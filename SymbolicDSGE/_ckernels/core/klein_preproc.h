#ifndef SDSGE_KLEIN_PREPROC_H
#define SDSGE_KLEIN_PREPROC_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"

typedef void (*sdsge_residual_fn)(const c128 *fwd, const c128 *cur,
                                  const c128 *par, c128 *out);

i64 klein_preproc(sdsge_residual_fn residual, const f64 *SDSGE_RESTRICT ss,
                  const f64 *SDSGE_RESTRICT par, const i64 n_var,
                  const i64 n_par, const i64 n_eq, const i64 log_linear,
                  f64 *SDSGE_RESTRICT a, f64 *SDSGE_RESTRICT b);

/* ERROR CODES */
#define SDSGE_PREKLEIN_OK 0
#define SDSGE_PREKLEIN_ALLOC_FAIL -1

#endif /* SDSGE_KLEIN_PREPROC_H */
