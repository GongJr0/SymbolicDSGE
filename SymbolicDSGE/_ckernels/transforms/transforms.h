#ifndef SDSGE_TRANSFORMS_H
#define SDSGE_TRANSFORMS_H

#include "../_common/sdsge_common.h"

/* LOG TRANSFORM */

void sdsge_log_fwd(const f64 *x, f64 *y);
void sdsge_log_fwd_arr(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT y,
                       i64 n);

void sdsge_log_inv(const f64 *y, f64 *x);
void sdsge_log_inv_arr(const f64 *SDSGE_RESTRICT y, f64 *SDSGE_RESTRICT x,
                       i64 n);

void sdsge_log_grad_fwd(const f64 *x, f64 *y);
void sdsge_log_grad_fwd_arr(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT y,
                            i64 n);

void sdsge_log_grad_inv(const f64 *y, f64 *x);
void sdsge_log_grad_inv_arr(const f64 *SDSGE_RESTRICT y, f64 *SDSGE_RESTRICT x,
                            i64 n);

void sdsge_log_ldet_abs_jac_fwd(const f64 *x, f64 *y);
void sdsge_log_ldet_abs_jac_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                    f64 *SDSGE_RESTRICT y, i64 n);

void sdsge_log_ldet_abs_jac_inv(const f64 *y, f64 *x);
void sdsge_log_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                    f64 *SDSGE_RESTRICT x, i64 n);

void sdsge_log_grad_ldet_abs_jac_inv(const f64 *y, f64 *x);
void sdsge_log_grad_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                         f64 *SDSGE_RESTRICT x, i64 n);

/* LOGIT TRANSFORM */

void sdsge_logit_fwd(const f64 *x, f64 *y);
void sdsge_logit_fwd_arr(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT y,
                         i64 n);

void sdsge_logit_inv(const f64 *y, f64 *x);
void sdsge_logit_inv_arr(const f64 *SDSGE_RESTRICT y, f64 *SDSGE_RESTRICT x,
                         i64 n);

void sdsge_logit_grad_fwd(const f64 *x, f64 *y);
void sdsge_logit_grad_fwd_arr(const f64 *SDSGE_RESTRICT x,
                              f64 *SDSGE_RESTRICT y, i64 n);

void sdsge_logit_grad_inv(const f64 *y, f64 *x);
void sdsge_logit_grad_inv_arr(const f64 *SDSGE_RESTRICT y,
                              f64 *SDSGE_RESTRICT x, i64 n);

void sdsge_logit_ldet_abs_jac_fwd(const f64 *x, f64 *y);
void sdsge_logit_ldet_abs_jac_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                      f64 *SDSGE_RESTRICT y, i64 n);

void sdsge_logit_ldet_abs_jac_inv(const f64 *y, f64 *x);
void sdsge_logit_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                      f64 *SDSGE_RESTRICT x, i64 n);

void sdsge_logit_grad_ldet_abs_jac_inv(const f64 *y, f64 *x);
void sdsge_logit_grad_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                           f64 *SDSGE_RESTRICT x, i64 n);

/* PROBIT TRANSFORM */

void sdsge_probit_fwd(const f64 *x, f64 *y);
void sdsge_probit_fwd_arr(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT y,
                          i64 n);

void sdsge_probit_inv(const f64 *y, f64 *x);
void sdsge_probit_inv_arr(const f64 *SDSGE_RESTRICT y, f64 *SDSGE_RESTRICT x,
                          i64 n);

void sdsge_probit_grad_fwd(const f64 *x, f64 *y);
void sdsge_probit_grad_fwd_arr(const f64 *SDSGE_RESTRICT x,
                               f64 *SDSGE_RESTRICT y, i64 n);

void sdsge_probit_grad_inv(const f64 *y, f64 *x);
void sdsge_probit_grad_inv_arr(const f64 *SDSGE_RESTRICT y,
                               f64 *SDSGE_RESTRICT x, i64 n);

void sdsge_probit_ldet_abs_jac_fwd(const f64 *x, f64 *y);
void sdsge_probit_ldet_abs_jac_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                       f64 *SDSGE_RESTRICT y, i64 n);

void sdsge_probit_ldet_abs_jac_inv(const f64 *y, f64 *x);
void sdsge_probit_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                       f64 *SDSGE_RESTRICT x, i64 n);

void sdsge_probit_grad_ldet_abs_jac_inv(const f64 *y, f64 *x);
void sdsge_probit_grad_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                            f64 *SDSGE_RESTRICT x, i64 n);

/* INLINE AFFINE HELPERS */
static inline void sdsge_affine_to_unit(const f64 *x, f64 *y, const f64 *a,
                                        const f64 *b) {
  *y = (*x - *a) / (*b - *a);
}

static inline void sdsge_unit_to_affine(const f64 *y, f64 *x, const f64 *a,
                                        const f64 *b) {
  *x = *y * (*b - *a) + *a;
}

/* AFFINE LOGIT TRANSFORM */

void sdsge_aff_logit_fwd(const f64 *x, f64 *y, const f64 *a, const f64 *b);
void sdsge_aff_logit_fwd_arr(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT y,
                             i64 n, const f64 *a, const f64 *b);

void sdsge_aff_logit_inv(const f64 *y, f64 *x, const f64 *a, const f64 *b);
void sdsge_aff_logit_inv_arr(const f64 *SDSGE_RESTRICT y, f64 *SDSGE_RESTRICT x,
                             i64 n, const f64 *a, const f64 *b);

void sdsge_aff_logit_grad_fwd(const f64 *x, f64 *y, const f64 *a, const f64 *b);
void sdsge_aff_logit_grad_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                  f64 *SDSGE_RESTRICT y, i64 n, const f64 *a,
                                  const f64 *b);

void sdsge_aff_logit_grad_inv(const f64 *y, f64 *x, const f64 *a, const f64 *b);
void sdsge_aff_logit_grad_inv_arr(const f64 *SDSGE_RESTRICT y,
                                  f64 *SDSGE_RESTRICT x, i64 n, const f64 *a,
                                  const f64 *b);

void sdsge_aff_logit_ldet_abs_jac_fwd(const f64 *x, f64 *y, const f64 *a,
                                      const f64 *b);
void sdsge_aff_logit_ldet_abs_jac_fwd_arr(const f64 *SDSGE_RESTRICT x,
                                          f64 *SDSGE_RESTRICT y, i64 n,
                                          const f64 *a, const f64 *b);

void sdsge_aff_logit_ldet_abs_jac_inv(const f64 *y, f64 *x, const f64 *a,
                                      const f64 *b);
void sdsge_aff_logit_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                          f64 *SDSGE_RESTRICT x, i64 n,
                                          const f64 *a, const f64 *b);

void sdsge_aff_logit_grad_ldet_abs_jac_inv(const f64 *y, f64 *x);
void sdsge_aff_logit_grad_ldet_abs_jac_inv_arr(const f64 *SDSGE_RESTRICT y,
                                               f64 *SDSGE_RESTRICT x, i64 n);

#endif /* SDSGE_TRANSFORMS_H */
