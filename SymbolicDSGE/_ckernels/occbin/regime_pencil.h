#ifndef SDSGE_REGIME_PENCIL_H
#define SDSGE_REGIME_PENCIL_H

#include "../_common/sdsge_common.h"

/* Replaced rows of one regime's pencil, written as
 * `[jac_a; jac_b; jac_c; jac_d; constants]` into a single buffer, each block
 * whole and row-major and ordered like `rows`. */
typedef void (*sdsge_regime_pencil_fn)(const f64 *cur, const f64 *par,
                                       f64 *out);

/* The blocks follow klein_preproc's naming and signs, so a row reads
 * `a y' = b y + c y_prev + d eps - cst`. The constant is `cst` rather than `c`:
 * `c` is the lag Jacobian everywhere else, and the two sitting in one struct
 * under one name is a trap rather than a shorthand. */
typedef struct {
  sdsge_regime_pencil_fn pencil; // NULL for reference regime
  i64 n_row;                     // number of rows a regime swaps
  const i64 *rows;               // n_row; indices of the rows a regime swaps
  f64 *a;                        // n_var*n_var
  f64 *b;                        // n_var*n_var
  f64 *c;                        // n_var*n_var
  f64 *d;                        // n_var*n_exog
  f64 *cst;                      // n_var
} regime_ctx;

arena_size sdsge_regime_pencil_arena_size(i64 n_var, i64 n_exog, i64 n_row);
void sdsge_regime_pencil(
    regime_ctx *regime, const f64 *SDSGE_RESTRICT ss,
    const f64 *SDSGE_RESTRICT par,
    const f64 *SDSGE_RESTRICT a_ref, // a at reference regime
    const f64 *SDSGE_RESTRICT b_ref, // b at reference regime
    const f64 *SDSGE_RESTRICT c_ref, // c at reference regime
    const f64 *SDSGE_RESTRICT d_ref, // d at reference regime
    i64 n_var, i64 n_exog, f64 *SDSGE_RESTRICT arena);

void regime_table(regime_ctx *table, // indexed by regime bitmask
                  i64 n_regime, const f64 *SDSGE_RESTRICT ss,
                  const f64 *SDSGE_RESTRICT par,
                  const f64 *SDSGE_RESTRICT a_ref,
                  const f64 *SDSGE_RESTRICT b_ref,
                  const f64 *SDSGE_RESTRICT c_ref,
                  const f64 *SDSGE_RESTRICT d_ref, i64 n_var, i64 n_exog,
                  f64 *SDSGE_RESTRICT arena);

#endif // SDSGE_REGIME_PENCIL_H
