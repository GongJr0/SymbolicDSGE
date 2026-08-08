#include "regime_pencil.h"
#include "../_common/sdsge_common.h"
#include <stddef.h> // NULL
#include <string.h> // memcpy, memset

arena_size sdsge_regime_pencil_arena_size(i64 n_var, i64 n_row) {
  return make_sizer(2 * n_row * n_var + n_row, 0);
}

void sdsge_regime_pencil(regime_ctx *regime, const f64 *SDSGE_RESTRICT ss,
                         const f64 *SDSGE_RESTRICT par,
                         const f64 *SDSGE_RESTRICT a_ref,
                         const f64 *SDSGE_RESTRICT b_ref, i64 n_var,
                         f64 *SDSGE_RESTRICT arena) {
  const size_t row_bytes = n_var * sizeof(f64);

  // Set reference regime
  memcpy(regime->a, a_ref, n_var * row_bytes);
  memcpy(regime->b, b_ref, n_var * row_bytes);
  memset(regime->c, 0, // c == 0 since ss is solved for the reference.
         row_bytes);

  if (regime->pencil == NULL) {
    return; // reference regime
  }

  // regime != reference, pencil produces a/b/c for swapped rows
  // we then scatter the swapped rows into the reference regime a/b/c
  const i64 n_row = regime->n_row;

  // pencil into flat arena [a; b; c]
  regime->pencil(ss, par, arena);
  const f64 *blk_a = arena;
  const f64 *blk_b = blk_a + n_row * n_var;
  const f64 *blk_c = blk_b + n_row * n_var;

  for (i64 i = 0; i < n_row; ++i) {
    const i64 row = regime->rows[i];
    memcpy(&regime->a[row * n_var], &blk_a[i * n_var], row_bytes);
    memcpy(&regime->b[row * n_var], &blk_b[i * n_var], row_bytes);
    regime->c[row] = blk_c[i];
  }
}

void regime_table(regime_ctx *table, i64 n_regime, const f64 *SDSGE_RESTRICT ss,
                  const f64 *SDSGE_RESTRICT par,
                  const f64 *SDSGE_RESTRICT a_ref,
                  const f64 *SDSGE_RESTRICT b_ref, i64 n_var,
                  f64 *SDSGE_RESTRICT arena) {
  // Caller sizes arena by max(n_row) in the table so every regime can use the
  // same arena.
  for (i64 mask = 0; mask < n_regime; ++mask) {
    sdsge_regime_pencil(&table[mask], ss, par, a_ref, b_ref, n_var, arena);
  }
}
