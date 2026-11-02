#include "regime_pencil.h"
#include "../_common/sdsge_common.h"
#include <stddef.h> // NULL
#include <string.h> // memcpy, memset

arena_size sdsge_regime_pencil_arena_size(i64 n_var, i64 n_exog, i64 n_row) {
  return make_sizer(n_row * (3 * n_var + n_exog + 1), 0);
}

void sdsge_regime_pencil(regime_ctx *regime, const f64 *SDSGE_RESTRICT ss,
                         const f64 *SDSGE_RESTRICT par,
                         const f64 *SDSGE_RESTRICT a_ref,
                         const f64 *SDSGE_RESTRICT b_ref,
                         const f64 *SDSGE_RESTRICT c_ref,
                         const f64 *SDSGE_RESTRICT d_ref, i64 n_var, i64 n_exog,
                         f64 *SDSGE_RESTRICT arena) {
  const size_t row_bytes = n_var * sizeof(f64);
  const size_t shock_row_bytes = n_exog * sizeof(f64);

  // Set reference regime
  memcpy(regime->a, a_ref, n_var * row_bytes);
  memcpy(regime->b, b_ref, n_var * row_bytes);
  memcpy(regime->c, c_ref, n_var * row_bytes);
  memcpy(regime->d, d_ref, n_var * shock_row_bytes);
  memset(regime->cst, 0, // cst == 0 since ss is solved for the reference.
         row_bytes);

  if (regime->pencil == NULL) {
    return; // reference regime
  }

  // regime != reference, pencil produces a/b/c/d/cst for swapped rows
  // we then scatter the swapped rows into the reference regime's copies
  const i64 n_row = regime->n_row;

  // pencil into flat arena [a; b; c; d; cst]
  regime->pencil(ss, par, arena);
  const f64 *blk_a = arena;
  const f64 *blk_b = blk_a + n_row * n_var;
  const f64 *blk_c = blk_b + n_row * n_var;
  const f64 *blk_d = blk_c + n_row * n_var;
  const f64 *blk_cst = blk_d + n_row * n_exog;

  for (i64 i = 0; i < n_row; ++i) {
    const i64 row = regime->rows[i];
    memcpy(&regime->a[row * n_var], &blk_a[i * n_var], row_bytes);
    memcpy(&regime->b[row * n_var], &blk_b[i * n_var], row_bytes);
    memcpy(&regime->c[row * n_var], &blk_c[i * n_var], row_bytes);
    memcpy(&regime->d[row * n_exog], &blk_d[i * n_exog], shock_row_bytes);
    regime->cst[row] = blk_cst[i];
  }
}

void regime_table(regime_ctx *table, i64 n_regime, const f64 *SDSGE_RESTRICT ss,
                  const f64 *SDSGE_RESTRICT par,
                  const f64 *SDSGE_RESTRICT a_ref,
                  const f64 *SDSGE_RESTRICT b_ref,
                  const f64 *SDSGE_RESTRICT c_ref,
                  const f64 *SDSGE_RESTRICT d_ref, i64 n_var, i64 n_exog,
                  f64 *SDSGE_RESTRICT arena) {
  // Caller sizes arena by max(n_row) in the table so every regime can use the
  // same arena.
  for (i64 mask = 0; mask < n_regime; ++mask) {
    sdsge_regime_pencil(&table[mask], ss, par, a_ref, b_ref, c_ref, d_ref,
                        n_var, n_exog, arena);
  }
}
