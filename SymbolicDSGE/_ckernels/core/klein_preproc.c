#include "klein_preproc.h"

#define CJAC_STEP 1e-30

static void perturb_sweep(sdsge_residual_fn resid,
                          const c128 *SDSGE_RESTRICT base,
                          c128 *SDSGE_RESTRICT fwd, c128 *SDSGE_RESTRICT cur,
                          const c128 *SDSGE_RESTRICT par, const i64 n_var,
                          const i64 n_eq, const int perturb_fwd, f64 sign,
                          f64 *SDSGE_RESTRICT mat, c128 *SDSGE_RESTRICT out) {
  const c128 istep = c128_make(0.0, CJAC_STEP); // complex-step
  for (i64 j = 0; j < n_var; ++j) {
    for (i64 i = 0; i < n_var; ++i) {
      fwd[i] = base[i];
      cur[i] = base[i];
    }
    if (perturb_fwd) {
      fwd[j] = c128_add(base[j], istep);
    } else {
      cur[j] = c128_add(base[j], istep);
    }

    resid(fwd, cur, par, out);

    for (i64 k = 0; k < n_eq; ++k) {
      mat[k * n_var + j] = sign * c128_imag(out[k]) / CJAC_STEP;
    }
  }
}

arena_size klein_preproc_arena_size(const i64 n_var, const i64 n_par,
                                    const i64 n_eq) {
  return make_sizer(2 * (3 * n_var + n_par + n_eq), /* base, fwd, cur, par, out */
                    0);
}

void klein_preproc(sdsge_residual_fn resid, const f64 *SDSGE_RESTRICT ss,
                   const f64 *SDSGE_RESTRICT par, const i64 n_var,
                   const i64 n_par, const i64 n_eq, f64 *SDSGE_RESTRICT a,
                   f64 *SDSGE_RESTRICT b, f64 *SDSGE_RESTRICT arena) {
  c128 *p = (c128 *)arena;
  c128 *base = p;
  p += n_var;
  c128 *par_c = p;
  p += n_par;
  c128 *fwd = p;
  p += n_var;
  c128 *cur = p;
  p += n_var;
  c128 *out = p;

  for (i64 i = 0; i < n_var; ++i) {
    base[i] = c128_from_real(ss[i]);
  }

  for (i64 j = 0; j < n_par; ++j) {
    par_c[j] = c128_from_real(par[j]);
  }

  perturb_sweep(resid, base, fwd, cur, par_c, n_var, n_eq, 1, 1.0, a, out);
  perturb_sweep(resid, base, fwd, cur, par_c, n_var, n_eq, 0, -1.0, b, out);
}
