#include "klein_preproc.h"

#define CJAC_STEP 1e-30

typedef enum {
  SWEEP_FWD,
  SWEEP_CUR,
  SWEEP_PREV,
  SWEEP_EPS,
} sweep_target;

static void perturb_sweep(sdsge_residual_fn resid,
                          const c128 *SDSGE_RESTRICT base,
                          c128 *SDSGE_RESTRICT fwd, c128 *SDSGE_RESTRICT cur,
                          c128 *SDSGE_RESTRICT prev, c128 *SDSGE_RESTRICT eps,
                          const c128 *SDSGE_RESTRICT par, const i64 n_var,
                          const i64 n_exog, const i64 n_eq,
                          const sweep_target target, f64 sign,
                          f64 *SDSGE_RESTRICT mat, c128 *SDSGE_RESTRICT out) {
  const c128 istep = c128_make(0.0, CJAC_STEP); // complex-step
  /* eps is the one block that is not one column per variable. */
  const i64 n_col = (target == SWEEP_EPS) ? n_exog : n_var;

  for (i64 j = 0; j < n_col; ++j) {
    for (i64 i = 0; i < n_var; ++i) {
      fwd[i] = base[i];
      cur[i] = base[i];
      prev[i] = base[i];
    }
    for (i64 i = 0; i < n_exog; ++i) {
      eps[i] = c128_from_real(0.0);
    }
    switch (target) {
    case SWEEP_FWD:
      fwd[j] = c128_add(base[j], istep);
      break;
    case SWEEP_CUR:
      cur[j] = c128_add(base[j], istep);
      break;
    case SWEEP_PREV:
      prev[j] = c128_add(base[j], istep);
      break;
    case SWEEP_EPS:
      eps[j] = istep; /* the expansion point is a zero innovation. */
      break;
    }

    resid(fwd, cur, prev, eps, par, out);

    for (i64 k = 0; k < n_eq; ++k) {
      mat[k * n_col + j] = sign * c128_imag(out[k]) / CJAC_STEP;
    }
  }
}

arena_size klein_preproc_arena_size(const i64 n_var, const i64 n_par,
                                    const i64 n_exog, const i64 n_eq) {
  return make_sizer(
      2 * (4 * n_var + n_par + n_exog + n_eq), /* base, par, fwd, cur, prev,
                                                  eps, out */
      0);
}

void klein_preproc(sdsge_residual_fn resid, const f64 *SDSGE_RESTRICT ss,
                   const f64 *SDSGE_RESTRICT par, const i64 n_var,
                   const i64 n_par, const i64 n_exog, const i64 n_eq,
                   f64 *SDSGE_RESTRICT a, f64 *SDSGE_RESTRICT b,
                   f64 *SDSGE_RESTRICT c, f64 *SDSGE_RESTRICT d,
                   f64 *SDSGE_RESTRICT arena) {

  c128 *base = (c128 *)arena;
  c128 *par_c = base + n_var;
  c128 *fwd = par_c + n_par;
  c128 *cur = fwd + n_var;
  c128 *prev = cur + n_var;
  c128 *eps = prev + n_var;
  c128 *out = eps + n_exog;

  for (i64 i = 0; i < n_var; ++i) {
    base[i] = c128_from_real(ss[i]);
  }

  for (i64 j = 0; j < n_par; ++j) {
    par_c[j] = c128_from_real(par[j]);
  }

  perturb_sweep(resid, base, fwd, cur, prev, eps, par_c, n_var, n_exog, n_eq,
                SWEEP_FWD, 1.0, a, out);
  perturb_sweep(resid, base, fwd, cur, prev, eps, par_c, n_var, n_exog, n_eq,
                SWEEP_CUR, -1.0, b, out);
  perturb_sweep(resid, base, fwd, cur, prev, eps, par_c, n_var, n_exog, n_eq,
                SWEEP_PREV, -1.0, c, out);
  perturb_sweep(resid, base, fwd, cur, prev, eps, par_c, n_var, n_exog, n_eq,
                SWEEP_EPS, -1.0, d, out);
}
