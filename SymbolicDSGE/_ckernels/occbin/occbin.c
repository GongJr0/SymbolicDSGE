#include "occbin.h"
#include "../_common/sdsge_linalg.h"

i64 sdsge_constraint_path(sdsge_constraint_fn cond, f64 *SDSGE_RESTRICT path,
                          f64 *SDSGE_RESTRICT par, const i8 *regime_in,
                          i8 *regime_out, i64 T, i64 n_var, i64 n_constraint) {
  i8 flags[4]; // 2 * MAX_CONSTRAINTS (a constraint is a relax/bind pair)
  i64 changed = 0;

  for (i64 t = 0; t < T; ++t) {
    cond(&path[t * n_var], par, flags);
    const i8 prev = regime_in[t];
    i8 next = 0;
    for (i64 i = 0; i < n_constraint; ++i) {
      // If the previous regime was binding, check if it should relax; if it was
      // relaxing, check if it should bind
      next |= (((prev >> i) & 1) ? !flags[2 * i + 1] : flags[2 * i]) << i;
    }
    regime_out[t] = next;
    changed += (next != prev);
  }
  return changed; // No status code, call can't fail
}

arena_size sdsge_occbin_recursion_arena_size(i64 n_var, i64 n_state,
                                             i64 n_ctrl) {
  const i64 n_rhs = n_state + 1;
  return make_sizer(n_var * n_var + n_var * n_rhs + n_ctrl * n_rhs, n_var);
}

i64 sdsge_occbin_recursion(const occbin_ctx *ctx, const i8 *SDSGE_RESTRICT mask,
                           i64 T, f64 *SDSGE_RESTRICT out,
                           i64 *SDSGE_RESTRICT singular_date,
                           f64 *SDSGE_RESTRICT arena,
                           i64 *SDSGE_RESTRICT iarena) {
  const i64 n_var = ctx->n_var;
  const i64 n_state = ctx->n_state;
  const i64 n_ctrl = ctx->n_ctrl;
  const i64 n_rhs = n_state + 1;

  f64 *m = arena;
  f64 *rhs = m + n_var * n_var;
  f64 *seed = rhs + n_var * n_rhs;
  i64 *piv = iarena;

  *singular_date = -1;

  /* Date T closes on the reference rule, restrided to match a solved block. */
  for (i64 l = 0; l < n_ctrl; ++l) {
    for (i64 j = 0; j < n_state; ++j) {
      seed[l * n_rhs + j] = ctx->f_ref[l * n_state + j];
    }
    seed[l * n_rhs + n_state] = 0.0;
  }
  const f64 *u_next = seed;

  for (i64 t = T - 1; t >= 0; --t) {
    const regime_ctx *reg = &ctx->table[mask[t]];

    for (i64 i = 0; i < n_var; ++i) {
      const f64 *a_row = reg->a + i * n_var;
      const f64 *b_row = reg->b + i * n_var;
      f64 *m_row = m + i * n_var;
      f64 *r_row = rhs + i * n_rhs;
      f64 rc = reg->c[i];

      for (i64 j = 0; j < n_state; ++j) {
        m_row[j] = a_row[j];
        r_row[j] = b_row[j];
      }
      for (i64 j = 0; j < n_ctrl; ++j) {
        m_row[n_state + j] = -b_row[n_state + j];
      }

      for (i64 l = 0; l < n_ctrl; ++l) {
        const f64 a_ul = a_row[n_state + l];
        const f64 *u_row = u_next + l * n_rhs;
        for (i64 j = 0; j < n_state; ++j) {
          m_row[j] += a_ul * u_row[j];
        }
        rc += a_ul * u_row[n_state];
      }
      r_row[n_state] = -rc;
    }

    if (sdsge_lu_factor_inplace(m, piv, n_var) != SDSGE_LU_SUCCESS) {
      *singular_date = t;
      return SDSGE_OCCBIN_RECURSION_SINGULAR;
    }

    f64 *rule = out + t * n_var * n_rhs;
    sdsge_lu_solve(m, piv, rhs, rule, n_var, n_rhs);
    u_next = rule + n_state * n_rhs;
  }
  return SDSGE_OCCBIN_RECURSION_OK;
}
