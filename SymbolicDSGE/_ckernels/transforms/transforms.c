#include "transforms.h"
#include "../_common/as241.h"
#include <math.h>

/* LOG TRANSFORM */

void sdsge_log_fwd(const f64 *x, f64 *y) { *y = log(*x); }

void sdsge_log_fwd_arr(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT y,
                       i64 n) {
  for (i64 i = 0; i < n; i++) {
    y[i] = log(x[i]);
  }
}

void sdsge_log_inv(const f64 *y, f64 *x) { *x = exp(*y); }

void sdsge_log_inv_arr(const f64 *SDSGE_RESTRICT y, f64 *SDSGE_RESTRICT x,
                       i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = exp(y[i]);
  }
}

void sdsge_log_grad_fwd(const f64 *x, f64 *y) { *y = 1.0 / *x; }

void sdsge_log_grad_fwd_arr(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT y,
                            i64 n) {
  for (i64 i = 0; i < n; i++) {
    y[i] = 1.0 / x[i];
  }
}

void sdsge_log_grad_inv(const f64 *y, f64 *x) { *x = exp(*y); }

void sdsge_log_grad_inv_arr(const f64 *SDSGE_RESTRICT y, f64 *SDSGE_RESTRICT x,
                            i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = exp(y[i]);
  }
}

void sdsge_log_ldet_abs_jac_fwd(const f64 *x, f64 *y) { *y = -log(*x); }

void sdsge_log_ldet_abs_jac_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                    f64 *SDSGE_RESTRICT y, i64 n) {
  for (i64 i = 0; i < n; i++) {
    y[i] = -log(x[i]);
  }
}

void sdsge_log_ldet_abs_jac_inv(const f64 *y, f64 *x) { *x = *y; }

void sdsge_log_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                    f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = y[i];
  }
}

void sdsge_log_grad_ldet_abs_jac_inv(const f64 *y, f64 *x) { *x = 1.0; }

void sdsge_log_grad_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                         f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = 1.0;
  }
}

/* LOGIT TRANSFORM */

void sdsge_logit_fwd(const f64 *x, f64 *y) { *y = log(*x / (1.0 - *x)); }

void sdsge_logit_fwd_arr(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT y,
                         i64 n) {
  for (i64 i = 0; i < n; i++) {
    y[i] = log(x[i] / (1.0 - x[i]));
  }
}

void sdsge_logit_inv(const f64 *y, f64 *x) { *x = 1.0 / (1.0 + exp(-*y)); }

void sdsge_logit_inv_arr(const f64 *SDSGE_RESTRICT y, f64 *SDSGE_RESTRICT x,
                         i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = 1.0 / (1.0 + exp(-y[i]));
  }
}

void sdsge_logit_grad_fwd(const f64 *x, f64 *y) {
  *y = 1.0 / (*x * (1.0 - *x));
}

void sdsge_logit_grad_fwd_arr(const f64 *SDSGE_RESTRICT x,
                              f64 *SDSGE_RESTRICT y, i64 n) {
  for (i64 i = 0; i < n; i++) {
    y[i] = 1.0 / (x[i] * (1.0 - x[i]));
  }
}

void sdsge_logit_grad_inv(const f64 *y, f64 *x) {
  f64 p;
  sdsge_logit_inv(y, &p);
  *x = p * (1.0 - p);
}

void sdsge_logit_grad_inv_arr(const f64 *SDSGE_RESTRICT y,
                              f64 *SDSGE_RESTRICT x, i64 n) {
  f64 p;
  for (i64 i = 0; i < n; i++) {
    sdsge_logit_inv(&y[i], &p);
    x[i] = p * (1.0 - p);
  }
}

void sdsge_logit_ldet_abs_jac_fwd(const f64 *x, f64 *y) {
  *y = -log(*x) - log(1.0 - *x);
}

void sdsge_logit_ldet_abs_jac_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                      f64 *SDSGE_RESTRICT y, i64 n) {
  for (i64 i = 0; i < n; i++) {
    y[i] = -log(x[i]) - log(1.0 - x[i]);
  }
}

void sdsge_logit_ldet_abs_jac_inv(const f64 *y, f64 *x) {
  *x = -*y - 2.0 * log(1.0 + exp(-*y));
}

void sdsge_logit_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                      f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = -y[i] - 2.0 * log(1.0 + exp(-y[i]));
  }
}

void sdsge_logit_grad_ldet_abs_jac_inv(const f64 *y, f64 *x) {
  *x = 1.0 - 2.0 / (1.0 + exp(-*y)); /* 1 - 2*expit(y) */
}

void sdsge_logit_grad_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                           f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = 1.0 - 2.0 / (1.0 + exp(-y[i]));
  }
}

/* PROBIT TRANSFORM */

/* 1/sqrt(2*pi), for the standard-normal pdf. */
#define SDSGE_INV_SQRT_2PI 0.3989422804014327

/* Standard-normal pdf and CDF. The forward map's inverse-CDF (ndtri) is
   Wichura AS 241 from _common; these two mirror the scipy norm.pdf / norm.cdf
   the reference ProbitTransform used. */
static inline f64 sdsge_std_norm_pdf(f64 z) {
  return SDSGE_INV_SQRT_2PI * exp(-0.5 * z * z);
}

static inline f64 sdsge_std_norm_cdf(f64 z) { return 0.5 * erfc(-z / SQRT2); }

void sdsge_probit_fwd(const f64 *x, f64 *y) { *y = sdsge_ndtri_as241(*x); }

void sdsge_probit_fwd_arr(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT y,
                          i64 n) {
  for (i64 i = 0; i < n; i++) {
    y[i] = sdsge_ndtri_as241(x[i]);
  }
}

void sdsge_probit_inv(const f64 *y, f64 *x) { *x = sdsge_std_norm_cdf(*y); }

void sdsge_probit_inv_arr(const f64 *SDSGE_RESTRICT y, f64 *SDSGE_RESTRICT x,
                          i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = sdsge_std_norm_cdf(y[i]);
  }
}

void sdsge_probit_grad_fwd(const f64 *x, f64 *y) {
  f64 p = sdsge_ndtri_as241(*x);
  *y = 1.0 / sdsge_std_norm_pdf(p);
}

void sdsge_probit_grad_fwd_arr(const f64 *SDSGE_RESTRICT x,
                               f64 *SDSGE_RESTRICT y, i64 n) {
  f64 p;
  for (i64 i = 0; i < n; i++) {
    p = sdsge_ndtri_as241(x[i]);
    y[i] = 1.0 / sdsge_std_norm_pdf(p);
  }
}

void sdsge_probit_grad_inv(const f64 *y, f64 *x) {
  *x = sdsge_std_norm_pdf(*y);
}

void sdsge_probit_grad_inv_arr(const f64 *SDSGE_RESTRICT y,
                               f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = sdsge_std_norm_pdf(y[i]);
  }
}

void sdsge_probit_ldet_abs_jac_fwd(const f64 *x, f64 *y) {
  f64 p = sdsge_ndtri_as241(*x);
  *y = -log(sdsge_std_norm_pdf(p));
}

void sdsge_probit_ldet_abs_jac_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                       f64 *SDSGE_RESTRICT y, i64 n) {
  f64 p;
  for (i64 i = 0; i < n; i++) {
    p = sdsge_ndtri_as241(x[i]);
    y[i] = -log(sdsge_std_norm_pdf(p));
  }
}

void sdsge_probit_ldet_abs_jac_inv(const f64 *y, f64 *x) {
  *x = log(sdsge_std_norm_pdf(*y));
}

void sdsge_probit_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                       f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = log(sdsge_std_norm_pdf(y[i]));
  }
}

void sdsge_probit_grad_ldet_abs_jac_inv(const f64 *y, f64 *x) { *x = -*y; }

void sdsge_probit_grad_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                            f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = -y[i];
  }
}

/* AFFINE LOGIT TRANSFORM */
void sdsge_aff_logit_fwd(const f64 *x, f64 *y, const f64 *a, const f64 *b) {
  f64 z;
  sdsge_affine_to_unit(x, &z, a, b);
  sdsge_logit_fwd(&z, y);
}

void sdsge_aff_logit_fwd_arr(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT y,
                             i64 n, const f64 *a, const f64 *b) {
  f64 z;
  for (i64 i = 0; i < n; i++) {
    sdsge_affine_to_unit(&x[i], &z, a, b);
    sdsge_logit_fwd(&z, &y[i]);
  }
}

void sdsge_aff_logit_inv(const f64 *y, f64 *x, const f64 *a, const f64 *b) {
  f64 z;
  sdsge_logit_inv(y, &z);
  sdsge_unit_to_affine(&z, x, a, b);
}

void sdsge_aff_logit_inv_arr(const f64 *SDSGE_RESTRICT y, f64 *SDSGE_RESTRICT x,
                             i64 n, const f64 *a, const f64 *b) {
  f64 z;
  for (i64 i = 0; i < n; i++) {
    sdsge_logit_inv(&y[i], &z);
    sdsge_unit_to_affine(&z, &x[i], a, b);
  }
}

void sdsge_aff_logit_grad_fwd(const f64 *x, f64 *y, const f64 *a,
                              const f64 *b) {
  f64 z;
  sdsge_affine_to_unit(x, &z, a, b);
  sdsge_logit_grad_fwd(&z, y);
  *y /= (*b - *a);
}

void sdsge_aff_logit_grad_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                  f64 *SDSGE_RESTRICT y, i64 n, const f64 *a,
                                  const f64 *b) {
  f64 z;
  for (i64 i = 0; i < n; i++) {
    sdsge_affine_to_unit(&x[i], &z, a, b);
    sdsge_logit_grad_fwd(&z, &y[i]);
    y[i] /= (*b - *a);
  }
}

void sdsge_aff_logit_grad_inv(const f64 *y, f64 *x, const f64 *a,
                              const f64 *b) {
  sdsge_logit_grad_inv(y, x);
  *x *= (*b - *a);
}

void sdsge_aff_logit_grad_inv_arr(const f64 *SDSGE_RESTRICT y,
                                  f64 *SDSGE_RESTRICT x, i64 n, const f64 *a,
                                  const f64 *b) {
  for (i64 i = 0; i < n; i++) {
    sdsge_logit_grad_inv(&y[i], &x[i]);
    x[i] *= (*b - *a);
  }
}

void sdsge_aff_logit_ldet_abs_jac_fwd(const f64 *x, f64 *y, const f64 *a,
                                      const f64 *b) {
  f64 z;
  sdsge_affine_to_unit(x, &z, a, b);
  sdsge_logit_ldet_abs_jac_fwd(&z, y);
  *y -= log(*b - *a);
}

void sdsge_aff_logit_ldet_abs_jac_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                          f64 *SDSGE_RESTRICT y, i64 n,
                                          const f64 *a, const f64 *b) {
  f64 z;
  for (i64 i = 0; i < n; i++) {
    sdsge_affine_to_unit(&x[i], &z, a, b);
    sdsge_logit_ldet_abs_jac_fwd(&z, &y[i]);
    y[i] -= log(*b - *a);
  }
}

void sdsge_aff_logit_ldet_abs_jac_inv(const f64 *y, f64 *x, const f64 *a,
                                      const f64 *b) {
  sdsge_logit_ldet_abs_jac_inv(y, x);
  *x += log(*b - *a);
}

void sdsge_aff_logit_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                          f64 *SDSGE_RESTRICT x, i64 n,
                                          const f64 *a, const f64 *b) {
  for (i64 i = 0; i < n; i++) {
    sdsge_logit_ldet_abs_jac_inv(&y[i], &x[i]);
    x[i] += log(*b - *a);
  }
}

void sdsge_aff_logit_grad_ldet_abs_jac_inv(const f64 *y, f64 *x) {
  sdsge_logit_grad_ldet_abs_jac_inv(y, x);
}

void sdsge_aff_logit_grad_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                               f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    sdsge_logit_grad_ldet_abs_jac_inv(&y[i], &x[i]);
  }
}

/* AFFINE PROBIT TRANSFORM */
void sdsge_aff_probit_fwd(const f64 *x, f64 *y, const f64 *a, const f64 *b) {
  f64 z;
  sdsge_affine_to_unit(x, &z, a, b);
  sdsge_probit_fwd(&z, y);
}

void sdsge_aff_probit_fwd_arr(const f64 *SDSGE_RESTRICT x,
                              f64 *SDSGE_RESTRICT y, i64 n, const f64 *a,
                              const f64 *b) {
  f64 z;
  for (i64 i = 0; i < n; i++) {
    sdsge_affine_to_unit(&x[i], &z, a, b);
    sdsge_probit_fwd(&z, &y[i]);
  }
}

void sdsge_aff_probit_inv(const f64 *y, f64 *x, const f64 *a, const f64 *b) {
  f64 z;
  sdsge_probit_inv(y, &z);
  sdsge_unit_to_affine(&z, x, a, b);
}

void sdsge_aff_probit_inv_arr(const f64 *SDSGE_RESTRICT y,
                              f64 *SDSGE_RESTRICT x, i64 n, const f64 *a,
                              const f64 *b) {
  f64 z;
  for (i64 i = 0; i < n; i++) {
    sdsge_probit_inv(&y[i], &z);
    sdsge_unit_to_affine(&z, &x[i], a, b);
  }
}

void sdsge_aff_probit_grad_fwd(const f64 *x, f64 *y, const f64 *a,
                               const f64 *b) {
  f64 z;
  sdsge_affine_to_unit(x, &z, a, b);
  sdsge_probit_grad_fwd(&z, y);
  *y /= (*b - *a);
}

void sdsge_aff_probit_grad_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                   f64 *SDSGE_RESTRICT y, i64 n, const f64 *a,
                                   const f64 *b) {
  f64 z;
  for (i64 i = 0; i < n; i++) {
    sdsge_affine_to_unit(&x[i], &z, a, b);
    sdsge_probit_grad_fwd(&z, &y[i]);
    y[i] /= (*b - *a);
  }
}

void sdsge_aff_probit_grad_inv(const f64 *y, f64 *x, const f64 *a,
                               const f64 *b) {
  sdsge_probit_grad_inv(y, x);
  *x *= (*b - *a);
}

void sdsge_aff_probit_grad_inv_arr(const f64 *SDSGE_RESTRICT y,
                                   f64 *SDSGE_RESTRICT x, i64 n, const f64 *a,
                                   const f64 *b) {
  for (i64 i = 0; i < n; i++) {
    sdsge_probit_grad_inv(&y[i], &x[i]);
    x[i] *= (*b - *a);
  }
}

void sdsge_aff_probit_ldet_abs_jac_fwd(const f64 *x, f64 *y, const f64 *a,
                                       const f64 *b) {
  f64 z;
  sdsge_affine_to_unit(x, &z, a, b);
  sdsge_probit_ldet_abs_jac_fwd(&z, y);
  *y -= log(*b - *a);
}

void sdsge_aff_probit_ldet_abs_jac_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                           f64 *SDSGE_RESTRICT y, i64 n,
                                           const f64 *a, const f64 *b) {
  f64 z;
  for (i64 i = 0; i < n; i++) {
    sdsge_affine_to_unit(&x[i], &z, a, b);
    sdsge_probit_ldet_abs_jac_fwd(&z, &y[i]);
    y[i] -= log(*b - *a);
  }
}

void sdsge_aff_probit_ldet_abs_jac_inv(const f64 *y, f64 *x, const f64 *a,
                                       const f64 *b) {
  sdsge_probit_ldet_abs_jac_inv(y, x);
  *x += log(*b - *a);
}

void sdsge_aff_probit_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                           f64 *SDSGE_RESTRICT x, i64 n,
                                           const f64 *a, const f64 *b) {
  for (i64 i = 0; i < n; i++) {
    sdsge_probit_ldet_abs_jac_inv(&y[i], &x[i]);
    x[i] += log(*b - *a);
  }
}

void sdsge_aff_probit_grad_ldet_abs_jac_inv(const f64 *y, f64 *x) {
  sdsge_probit_grad_ldet_abs_jac_inv(y, x);
}

void sdsge_aff_probit_grad_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                                f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    sdsge_probit_grad_ldet_abs_jac_inv(&y[i], &x[i]);
  }
}

/* SOFTPLUS TRANSFORM */

static inline f64 sdsge_softplus_val(f64 y) {
  return y > 0.0 ? y + log1p(exp(-y)) : log1p(exp(y));
}

void sdsge_softplus_fwd(const f64 *x, f64 *y) { *y = log(expm1(*x)); }

void sdsge_softplus_fwd_arr(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT y,
                            i64 n) {
  for (i64 i = 0; i < n; i++) {
    y[i] = log(expm1(x[i]));
  }
}

void sdsge_softplus_inv(const f64 *y, f64 *x) { *x = sdsge_softplus_val(*y); }

void sdsge_softplus_inv_arr(const f64 *SDSGE_RESTRICT y, f64 *SDSGE_RESTRICT x,
                            i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = sdsge_softplus_val(y[i]);
  }
}

void sdsge_softplus_grad_fwd(const f64 *x, f64 *y) {
  *y = 1.0 + 1.0 / expm1(*x);
}

void sdsge_softplus_grad_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                 f64 *SDSGE_RESTRICT y, i64 n) {
  for (i64 i = 0; i < n; i++) {
    y[i] = 1.0 + 1.0 / expm1(x[i]);
  }
}

void sdsge_softplus_grad_inv(const f64 *y, f64 *x) {
  *x = 1.0 / (1.0 + exp(-*y));
}

void sdsge_softplus_grad_inv_arr(const f64 *SDSGE_RESTRICT y,
                                 f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = 1.0 / (1.0 + exp(-y[i]));
  }
}

void sdsge_softplus_ldet_abs_jac_fwd(const f64 *x, f64 *y) {
  *y = *x - log(expm1(*x));
}

void sdsge_softplus_ldet_abs_jac_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                         f64 *SDSGE_RESTRICT y, i64 n) {
  for (i64 i = 0; i < n; i++) {
    y[i] = x[i] - log(expm1(x[i]));
  }
}

void sdsge_softplus_ldet_abs_jac_inv(const f64 *y, f64 *x) {
  *x = -sdsge_softplus_val(-*y);
}

void sdsge_softplus_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                         f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = -sdsge_softplus_val(-y[i]);
  }
}

void sdsge_softplus_grad_ldet_abs_jac_inv(const f64 *y, f64 *x) {
  *x = 1.0 - 1.0 / (1.0 + exp(-*y));
}

void sdsge_softplus_grad_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                              f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = 1.0 - 1.0 / (1.0 + exp(-y[i]));
  }
}

/* LOWER BOUNDED TRANSFORM */

void sdsge_lower_fwd(const f64 *x, f64 *y, const f64 *low) {
  *y = log(*x - *low);
}

void sdsge_lower_fwd_arr(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT y,
                         i64 n, const f64 *low) {
  for (i64 i = 0; i < n; i++) {
    y[i] = log(x[i] - *low);
  }
}

void sdsge_lower_inv(const f64 *y, f64 *x, const f64 *low) {
  *x = *low + exp(*y);
}

void sdsge_lower_inv_arr(const f64 *SDSGE_RESTRICT y, f64 *SDSGE_RESTRICT x,
                         i64 n, const f64 *low) {
  for (i64 i = 0; i < n; i++) {
    x[i] = *low + exp(y[i]);
  }
}

void sdsge_lower_grad_fwd(const f64 *x, f64 *y, const f64 *low) {
  *y = 1.0 / (*x - *low);
}

void sdsge_lower_grad_fwd_arr(const f64 *SDSGE_RESTRICT x,
                              f64 *SDSGE_RESTRICT y, i64 n, const f64 *low) {
  for (i64 i = 0; i < n; i++) {
    y[i] = 1.0 / (x[i] - *low);
  }
}

void sdsge_lower_grad_inv(const f64 *y, f64 *x) { *x = exp(*y); }

void sdsge_lower_grad_inv_arr(const f64 *SDSGE_RESTRICT y,
                              f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = exp(y[i]);
  }
}

void sdsge_lower_ldet_abs_jac_fwd(const f64 *x, f64 *y, const f64 *low) {
  *y = -log(*x - *low);
}

void sdsge_lower_ldet_abs_jac_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                      f64 *SDSGE_RESTRICT y, i64 n,
                                      const f64 *low) {
  for (i64 i = 0; i < n; i++) {
    y[i] = -log(x[i] - *low);
  }
}

void sdsge_lower_ldet_abs_jac_inv(const f64 *y, f64 *x) { *x = *y; }

void sdsge_lower_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                      f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = y[i];
  }
}

void sdsge_lower_grad_ldet_abs_jac_inv(const f64 *y, f64 *x) { *x = 1.0; }

void sdsge_lower_grad_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                           f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = 1.0;
  }
}

/* UPPER BOUNDED TRANSFORM */

void sdsge_upper_fwd(const f64 *x, f64 *y, const f64 *high) {
  *y = log(*high - *x);
}

void sdsge_upper_fwd_arr(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT y,
                         i64 n, const f64 *high) {
  for (i64 i = 0; i < n; i++) {
    y[i] = log(*high - x[i]);
  }
}

void sdsge_upper_inv(const f64 *y, f64 *x, const f64 *high) {
  *x = *high - exp(*y);
}

void sdsge_upper_inv_arr(const f64 *SDSGE_RESTRICT y, f64 *SDSGE_RESTRICT x,
                         i64 n, const f64 *high) {
  for (i64 i = 0; i < n; i++) {
    x[i] = *high - exp(y[i]);
  }
}

void sdsge_upper_grad_fwd(const f64 *x, f64 *y, const f64 *high) {
  *y = -1.0 / (*high - *x);
}

void sdsge_upper_grad_fwd_arr(const f64 *SDSGE_RESTRICT x,
                              f64 *SDSGE_RESTRICT y, i64 n, const f64 *high) {
  for (i64 i = 0; i < n; i++) {
    y[i] = -1.0 / (*high - x[i]);
  }
}

void sdsge_upper_grad_inv(const f64 *y, f64 *x) { *x = -exp(*y); }

void sdsge_upper_grad_inv_arr(const f64 *SDSGE_RESTRICT y,
                              f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = -exp(y[i]);
  }
}

void sdsge_upper_ldet_abs_jac_fwd(const f64 *x, f64 *y, const f64 *high) {
  *y = -log(*high - *x);
}

void sdsge_upper_ldet_abs_jac_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                      f64 *SDSGE_RESTRICT y, i64 n,
                                      const f64 *high) {
  for (i64 i = 0; i < n; i++) {
    y[i] = -log(*high - x[i]);
  }
}

void sdsge_upper_ldet_abs_jac_inv(const f64 *y, f64 *x) { *x = *y; }

void sdsge_upper_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                      f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = y[i];
  }
}

void sdsge_upper_grad_ldet_abs_jac_inv(const f64 *y, f64 *x) { *x = 1.0; }

void sdsge_upper_grad_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                           f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = 1.0;
  }
}

/* TANH TRANSFORM */

/* log(2), for the stable log-sech^2 below. */
#define SDSGE_LN2 0.6931471805599453

/* log(sech^2(y)) = log(1 - tanh^2(y)), written to avoid both the 1 - tanh^2
   cancellation as tanh -> +-1 and the cosh overflow for large |y|. */
static inline f64 sdsge_log_sech2(f64 y) {
  f64 ay = fabs(y);
  return 2.0 * (SDSGE_LN2 - ay - log1p(exp(-2.0 * ay)));
}

void sdsge_tanh_fwd(const f64 *x, f64 *y) { *y = atanh(*x); }

void sdsge_tanh_fwd_arr(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT y,
                        i64 n) {
  for (i64 i = 0; i < n; i++) {
    y[i] = atanh(x[i]);
  }
}

void sdsge_tanh_inv(const f64 *y, f64 *x) { *x = tanh(*y); }

void sdsge_tanh_inv_arr(const f64 *SDSGE_RESTRICT y, f64 *SDSGE_RESTRICT x,
                        i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = tanh(y[i]);
  }
}

/* dy/dx = 1 / (1 - x^2), factored as (1-x)(1+x) to keep precision near +-1. */
void sdsge_tanh_grad_fwd(const f64 *x, f64 *y) {
  *y = 1.0 / ((1.0 - *x) * (1.0 + *x));
}

void sdsge_tanh_grad_fwd_arr(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT y,
                             i64 n) {
  for (i64 i = 0; i < n; i++) {
    y[i] = 1.0 / ((1.0 - x[i]) * (1.0 + x[i]));
  }
}

/* dx/dy = 1 - tanh^2(y) = sech^2(y); 1/cosh^2 avoids the cancellation. */
void sdsge_tanh_grad_inv(const f64 *y, f64 *x) {
  f64 c = cosh(*y);
  *x = 1.0 / (c * c);
}

void sdsge_tanh_grad_inv_arr(const f64 *SDSGE_RESTRICT y, f64 *SDSGE_RESTRICT x,
                             i64 n) {
  f64 c;
  for (i64 i = 0; i < n; i++) {
    c = cosh(y[i]);
    x[i] = 1.0 / (c * c);
  }
}

void sdsge_tanh_ldet_abs_jac_fwd(const f64 *x, f64 *y) {
  *y = -log((1.0 - *x) * (1.0 + *x));
}

void sdsge_tanh_ldet_abs_jac_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                     f64 *SDSGE_RESTRICT y, i64 n) {
  for (i64 i = 0; i < n; i++) {
    y[i] = -log((1.0 - x[i]) * (1.0 + x[i]));
  }
}

void sdsge_tanh_ldet_abs_jac_inv(const f64 *y, f64 *x) {
  *x = sdsge_log_sech2(*y);
}

void sdsge_tanh_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                     f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = sdsge_log_sech2(y[i]);
  }
}

void sdsge_tanh_grad_ldet_abs_jac_inv(const f64 *y, f64 *x) {
  *x = -2.0 * tanh(*y);
}

void sdsge_tanh_grad_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                          f64 *SDSGE_RESTRICT x, i64 n) {
  for (i64 i = 0; i < n; i++) {
    x[i] = -2.0 * tanh(y[i]);
  }
}
