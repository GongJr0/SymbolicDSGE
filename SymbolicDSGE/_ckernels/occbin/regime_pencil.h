#ifndef SDSGE_REGIME_PENCIL_H
#define SDSGE_REGIME_PENCIL_H

#include "../_common/sdsge_common.h"

typedef void (*sdsge_regime_pencil_fn)(const f64 *cur, const f64 *par,
                                       f64 *out);

typedef struct {
  sdsge_regime_pencil_fn pencil; // NULL for reference regime
  i64 n_row;                     // number of rows a regime swaps
  const i64 *rows;               // n_row; indices of the rows a regime swaps
  f64 *a;                        // n_var*n_var
  f64 *b;                        // n_var*n_var
  f64 *c;                        // n_var
} regime_ctx;

arena_size sdsge_regime_pencil_arena_size(i64 n_var, i64 n_row);
void sdsge_regime_pencil(
    regime_ctx *regime, const f64 *SDSGE_RESTRICT ss,
    const f64 *SDSGE_RESTRICT par,
    const f64 *SDSGE_RESTRICT a_ref, // a at reference regime
    const f64 *SDSGE_RESTRICT b_ref, // b at reference regime
    i64 n_var, f64 *SDSGE_RESTRICT arena);

void regime_table(regime_ctx *table, // indexed by regime bitmask
                  i64 n_regime, const f64 *SDSGE_RESTRICT ss,
                  const f64 *SDSGE_RESTRICT par,
                  const f64 *SDSGE_RESTRICT a_ref,
                  const f64 *SDSGE_RESTRICT b_ref, i64 n_var,
                  f64 *SDSGE_RESTRICT arena);

#endif // SDSGE_REGIME_PENCIL_H
