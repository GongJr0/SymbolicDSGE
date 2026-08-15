#include "bicomplex_hessian.h"

/* Component slots of a bc256: real = a.re, i-unit = a.im, j-unit = b.re,
 * ij = b.im. The base holds the i/j slots at 0, so set-then-set-0 restores it
 * exactly.
 *
 * z is (lag, cur, lead, eps), matching the column order Dynare's second-order
 * solver selects with `kk1`, so a contraction against this Hessian reads the
 * same way its does. */
static inline bc256 *sdsge_bc_slot(bc256 *SDSGE_RESTRICT prev,
                                   bc256 *SDSGE_RESTRICT cur,
                                   bc256 *SDSGE_RESTRICT fwd,
                                   bc256 *SDSGE_RESTRICT eps, const i64 n_var,
                                   const i64 idx) {
  if (idx < n_var) {
    return &prev[idx];
  }
  if (idx < 2 * n_var) {
    return &cur[idx - n_var];
  }
  if (idx < 3 * n_var) {
    return &fwd[idx - 2 * n_var];
  }
  return &eps[idx - 3 * n_var];
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
  const i64 nz = 3 * n_var + n_exog;

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

  /* Base: real steady state at every date, zero innovation, params real. */
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

  for (i64 i = 0; i < nz; ++i) {
    bc256 *const zi = sdsge_bc_slot(prev, cur, fwd, eps, n_var, i);
    for (i64 j = i; j < nz; ++j) {
      bc256 *const zj = sdsge_bc_slot(prev, cur, fwd, eps, n_var, j);
      zi->a.im = SDSGE_HESSIAN_STEP;
      zj->b.re = SDSGE_HESSIAN_STEP;

      residual(fwd, cur, prev, eps, par_c, out);

      for (i64 eq = 0; eq < n_eq; ++eq) {
        const f64 val = out[eq].b.im * SDSGE_HESSIAN_INV_STEP2;
        hessian[eq * nz * nz + i * nz + j] = val;
        hessian[eq * nz * nz + j * nz + i] = val;
      }

      /* Restore the real base for the next pair. */
      zi->a.im = 0.0;
      zj->b.re = 0.0;
    }
  }
}
