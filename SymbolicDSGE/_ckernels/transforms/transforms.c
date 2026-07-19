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
