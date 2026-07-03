#include "bicomplex_hessian.h"
#include <stdlib.h>

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

i64 sdsge_bicomplex_hessian(bc_residual_fn residual, const f64 *SDSGE_RESTRICT ss,
                            const f64 *SDSGE_RESTRICT par, i64 n_var, i64 n_par,
                            i64 n_eq, f64 step, f64 *SDSGE_RESTRICT hessian) {
  const i64 n2 = 2 * n_var;
  const f64 inv_h2 = 1.0 / (step * step);

  bc256 *fwd = (bc256 *)malloc((size_t)(n_var > 0 ? n_var : 1) * sizeof(bc256));
  bc256 *cur = (bc256 *)malloc((size_t)(n_var > 0 ? n_var : 1) * sizeof(bc256));
  bc256 *par_c = (bc256 *)malloc((size_t)(n_par > 0 ? n_par : 1) * sizeof(bc256));
  bc256 *out = (bc256 *)malloc((size_t)(n_eq > 0 ? n_eq : 1) * sizeof(bc256));
  if (!fwd || !cur || !par_c || !out) {
    free(fwd);
    free(cur);
    free(par_c);
    free(out);
    return SDSGE_HESSIAN_ALLOC_FAIL;
  }

  /* Base: real steady state at both t+1 and t; params real. Set once. */
  for (i64 k = 0; k < n_var; ++k) {
    fwd[k] = bc256_from_real(ss[k]);
    cur[k] = bc256_from_real(ss[k]);
  }
  for (i64 k = 0; k < n_par; ++k) {
    par_c[k] = bc256_from_real(par[k]);
  }

  for (i64 i = 0; i < n2; ++i) {
    for (i64 j = i; j < n2; ++j) {
      set_i_unit(fwd, cur, n_var, i, step);
      set_j_unit(fwd, cur, n_var, j, step);

      residual(fwd, cur, par_c, out);

      for (i64 eq = 0; eq < n_eq; ++eq) {
        const f64 val = out[eq].b.im * inv_h2;
        hessian[eq * n2 * n2 + i * n2 + j] = val;
        hessian[eq * n2 * n2 + j * n2 + i] = val;
      }

      /* Restore the real base for the next pair. */
      set_i_unit(fwd, cur, n_var, i, 0.0);
      set_j_unit(fwd, cur, n_var, j, 0.0);
    }
  }

  free(fwd);
  free(cur);
  free(par_c);
  free(out);
  return SDSGE_HESSIAN_OK;
}
