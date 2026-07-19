#include "transforms.h"
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
