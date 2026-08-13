#include "bicomplex_hessian.h"

/* Component slots of a bc256: real = a.re, i-unit = a.im, j-unit = b.re,
 * ij = b.im. Perturbations set the i/j slots of a stacked arg (fwd for idx <
 * n_var, cur otherwise); the base holds those at 0, so set-then-set-0 restores
 * it exactly. */
static void set_i_unit(bc256 *SDSGE_RESTRICT fwd, bc256 *SDSGE_RESTRICT cur,
                       i64 n_var, i64 idx, f64 v) {
  if (idx < n_var) {
    fwd[idx].a.im = v;
  } else {
    cur[idx - n_var].a.im = v;
  }
}

static void set_j_unit(bc256 *SDSGE_RESTRICT fwd, bc256 *SDSGE_RESTRICT cur,
                       i64 n_var, i64 idx, f64 v) {
  if (idx < n_var) {
    fwd[idx].b.re = v;
  } else {
    cur[idx - n_var].b.re = v;
  }
}

arena_size sdsge_bicomplex_hessian_arena_size(const i64 n_var, const i64 n_par,
                                              const i64 n_exog,
                                              const i64 n_eq) {
  return make_sizer(
      4 * (3 * n_var + n_par + n_exog + n_eq), /* fwd, cur, prev, par, eps, out */
      0);
}

void sdsge_bicomplex_hessian(bc_residual_fn residual,
                             const f64 *SDSGE_RESTRICT ss,
                             const f64 *SDSGE_RESTRICT par, i64 n_var, i64 n_par,
                             i64 n_exog, i64 n_eq, f64 *SDSGE_RESTRICT hessian,
                             f64 *SDSGE_RESTRICT arena) {
  const i64 n2 = 2 * n_var;

  bc256 *bp = (bc256 *)arena;
  bc256 *fwd = bp;
  bp += n_var;
  bc256 *cur = bp;
  bp += n_var;
  bc256 *prev = bp;
  bp += n_var;
  bc256 *par_c = bp;
  bp += n_par;
  bc256 *eps = bp;
  bp += n_exog;
  bc256 *out = bp;

  /* Base: real steady state at every date, zero innovation, params real. Set
   * once. `prev` and `eps` are never perturbed, so they stay at the base for
   * the whole sweep. */
  for (i64 k = 0; k < n_var; ++k) {
    fwd[k] = bc256_from_real(ss[k]);
    cur[k] = bc256_from_real(ss[k]);
    prev[k] = bc256_from_real(ss[k]);
  }
  for (i64 k = 0; k < n_par; ++k) {
    par_c[k] = bc256_from_real(par[k]);
  }
  for (i64 k = 0; k < n_exog; ++k) {
    eps[k] = bc256_from_real(0.0);
  }

  for (i64 i = 0; i < n2; ++i) {
    for (i64 j = i; j < n2; ++j) {
      set_i_unit(fwd, cur, n_var, i, SDSGE_HESSIAN_STEP);
      set_j_unit(fwd, cur, n_var, j, SDSGE_HESSIAN_STEP);

      residual(fwd, cur, prev, eps, par_c, out);

      for (i64 eq = 0; eq < n_eq; ++eq) {
        const f64 val = out[eq].b.im * SDSGE_HESSIAN_INV_STEP2;
        hessian[eq * n2 * n2 + i * n2 + j] = val;
        hessian[eq * n2 * n2 + j * n2 + i] = val;
      }

      /* Restore the real base for the next pair. */
      set_i_unit(fwd, cur, n_var, i, 0.0);
      set_j_unit(fwd, cur, n_var, j, 0.0);
    }
  }
}
