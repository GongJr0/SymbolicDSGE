#include "klein_preproc.h"
#include <stdlib.h>

#define CJAC_STEP 1e-30

static i64 eval_resid(sdsge_residual_fn resid, const c128 *SDSGE_RESTRICT fwd,
                      const c128 *SDSGE_RESTRICT cur,
                      const c128 *SDSGE_RESTRICT par, const i64 n_var,
                      const i64 n_eq, const i64 log_linear,
                      c128 *SDSGE_RESTRICT out) {

  if (!log_linear) {
    resid(fwd, cur, par, out);
    return SDSGE_PREKLEIN_OK;
  }

  c128 *SDSGE_RESTRICT efwd = malloc((size_t)n_var * sizeof(c128));
  c128 *SDSGE_RESTRICT ecur = malloc((size_t)n_var * sizeof(c128));

  if (!efwd || !ecur) {
    free(efwd);
    free(ecur);
    return SDSGE_PREKLEIN_ALLOC_FAIL;
  }

  for (i64 i = 0; i < n_var; ++i) {
    efwd[i] = c128_exp(fwd[i]);
    ecur[i] = c128_exp(cur[i]);
  }
  resid(efwd, ecur, par, out);
  for (i64 k = 0; k < n_eq; ++k) {
    out[k] = c128_log(c128_add(out[k], c128_from_real(1.0)));
  }

  free(efwd);
  free(ecur);
  return SDSGE_PREKLEIN_OK;
}

static i64 perturb_sweep(sdsge_residual_fn resid,
                         const c128 *SDSGE_RESTRICT base,
                         c128 *SDSGE_RESTRICT fwd, c128 *SDSGE_RESTRICT cur,
                         const c128 *SDSGE_RESTRICT par, const i64 n_var,
                         const i64 n_eq, const i64 log_linear,
                         const int perturb_fwd, f64 sign,
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

    const i64 err =
        eval_resid(resid, fwd, cur, par, n_var, n_eq, log_linear, out);
    if (err != SDSGE_PREKLEIN_OK) {
      return err;
    }

    for (i64 k = 0; k < n_eq; ++k) {
      mat[k * n_var + j] = sign * c128_imag(out[k]) / CJAC_STEP;
    }
  }
  return SDSGE_PREKLEIN_OK;
}

i64 klein_preproc(sdsge_residual_fn resid, const f64 *SDSGE_RESTRICT ss,
                  const f64 *SDSGE_RESTRICT par, const i64 n_var,
                  const i64 n_par, const i64 n_eq, const i64 log_linear,
                  f64 *SDSGE_RESTRICT a, f64 *SDSGE_RESTRICT b) {
  c128 *base = (c128 *)malloc((size_t)(n_var > 0 ? n_var : 1) * sizeof(c128));
  c128 *par_c = (c128 *)malloc((size_t)(n_par > 0 ? n_par : 1) * sizeof(c128));
  c128 *fwd = (c128 *)malloc((size_t)(n_var > 0 ? n_var : 1) * sizeof(c128));
  c128 *cur = (c128 *)malloc((size_t)(n_var > 0 ? n_var : 1) * sizeof(c128));
  c128 *out = (c128 *)malloc((size_t)(n_eq > 0 ? n_eq : 1) * sizeof(c128));
  if (!base || !par_c || !fwd || !cur || !out) {
    free(base);
    free(par_c);
    free(fwd);
    free(cur);
    free(out);
    return SDSGE_PREKLEIN_ALLOC_FAIL;
  }

  for (i64 i = 0; i < n_var; ++i) {
    const c128 ssi = c128_from_real(ss[i]);
    base[i] = log_linear ? c128_log(ssi) : ssi;
  }

  for (i64 j = 0; j < n_par; ++j) {
    par_c[j] = c128_from_real(par[j]);
  }

  i64 err = perturb_sweep(resid, base, fwd, cur, par_c, n_var, n_eq, log_linear,
                          1, 1.0, a, out);
  if (err == SDSGE_PREKLEIN_OK) {
    err = perturb_sweep(resid, base, fwd, cur, par_c, n_var, n_eq, log_linear,
                        0, -1.0, b, out);
  }

  free(base);
  free(par_c);
  free(fwd);
  free(cur);
  free(out);
  return err;
}
