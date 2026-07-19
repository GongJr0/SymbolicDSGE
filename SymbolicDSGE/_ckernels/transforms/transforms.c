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

void sdsge_probit_grad_inv(const f64 *y, f64 *x) { *x = sdsge_std_norm_pdf(*y); }

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
