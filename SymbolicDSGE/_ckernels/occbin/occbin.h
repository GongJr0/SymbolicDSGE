#ifndef SDSGE_OCCBIN_H
#define SDSGE_OCCBIN_H

#include "../_common/sdsge_common.h"
#include "regime_pencil.h"

typedef void (*sdsge_constraint_fn)(f64 *cur, f64 *par, f64 *err);

i64 sdsge_constraint_path(sdsge_constraint_fn cond,
                          f64 *SDSGE_RESTRICT path, // (T, n_var)
                          f64 *SDSGE_RESTRICT par,  // (n_par,)
                          const i8 *regime_in,      // (T,)
                          i8 *regime_out,           // (T,)
                          i64 inclusive, // Bitmask for inequality strictness
                          f64 *SDSGE_RESTRICT max_err, i64 T, i64 n_var,
                          i64 n_constraint);
typedef struct {
  const regime_ctx *table;
  const f64 *f_ref;
  i64 n_var;
  i64 n_state;
  i64 n_ctrl;
} occbin_ctx;

arena_size sdsge_occbin_recursion_arena_size(i64 n_var, i64 n_state,
                                             i64 n_ctrl);

i64 sdsge_occbin_recursion(
    const occbin_ctx *ctx, const i8 *SDSGE_RESTRICT mask, i64 T,
    f64 *SDSGE_RESTRICT out, // (T, n_var, n_state + 1)
    i64 *SDSGE_RESTRICT
        singular_date, // `t` if the recursion is singular, -1 otherwise
    f64 *SDSGE_RESTRICT arena, i64 *SDSGE_RESTRICT iarena);

#define SDSGE_OCCBIN_RECURSION_OK 0
#define SDSGE_OCCBIN_RECURSION_SINGULAR -2 // match code to LU factorization

#endif // SDSGE_OCCBIN_H
