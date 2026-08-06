#ifndef SDSGE_OCCBIN_H
#define SDSGE_OCCBIN_H

#include "../_common/sdsge_common.h"

typedef void (*sdsge_constraint_fn)(f64 *cur, f64 *par, i8 *flags);

i64 sdsge_constraint_path(sdsge_constraint_fn cond,
                          f64 *SDSGE_RESTRICT path, // (T, n_var)
                          f64 *SDSGE_RESTRICT par,  // (n_par,)
                          const i8 *regime_in,      // (T,)
                          i8 *regime_out,           // (T,)
                          i64 T, i64 n_var, i64 n_constraint);

#endif // SDSGE_OCCBIN_H
