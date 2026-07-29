#include "tests.h"
#include <stddef.h>

static int sdsge_mc_test_status(const int status, i64 *SDSGE_RESTRICT int_out) {
  if (int_out != NULL) {
    int_out[0] = status;
  }
  return status;
}

int sdsge_mc_wald_test_runner(const i64 rep_idx,
                              f64 *SDSGE_RESTRICT float_in_work,
                              f64 *SDSGE_RESTRICT float_out,
                              i64 *SDSGE_RESTRICT int_work,
                              i64 *SDSGE_RESTRICT int_out, const void *ctx) {
  (void)rep_idx;
  const sdsge_mc_wald_test_ctx *config = ctx;
  const f64 *sample = float_in_work;
  f64 *arena = float_in_work + config->n * config->q;
  int status;

  if (config->kind == SDSGE_MC_WALD_MEAN) {
    status = sdsge_wald_mean_hac(sample, config->target, config->n, config->q,
                                 config->kernel_id, config->bandwidth_mode,
                                 config->manual_bandwidth, arena, int_work,
                                 float_out);
  } else if (config->kind == SDSGE_MC_WALD_COVARIANCE) {
    status = sdsge_wald_covariance_hac(
        sample, config->target, config->n, config->q, config->kernel_id,
        config->bandwidth_mode, config->manual_bandwidth, arena, int_work,
        float_out);
  } else {
    status = sdsge_wald_second_moment_hac(
        sample, config->target, config->n, config->q, config->kernel_id,
        config->bandwidth_mode, config->manual_bandwidth, arena, int_work,
        float_out);
  }
  return sdsge_mc_test_status(status, int_out);
}

int sdsge_mc_ljung_box_test_runner(const i64 rep_idx,
                                   f64 *SDSGE_RESTRICT float_in_work,
                                   f64 *SDSGE_RESTRICT float_out,
                                   i64 *SDSGE_RESTRICT int_work,
                                   i64 *SDSGE_RESTRICT int_out,
                                   const void *ctx) {
  (void)rep_idx;
  (void)int_work;
  const sdsge_mc_ljung_box_test_ctx *config = ctx;
  const f64 *x = float_in_work;
  f64 *arena = float_in_work + config->n;
  const int status = sdsge_lb_stat(x, config->n, config->lags, arena,
                                   arena + config->n, float_out);
  return sdsge_mc_test_status(status, int_out);
}

int sdsge_mc_jarque_bera_test_runner(const i64 rep_idx,
                                     f64 *SDSGE_RESTRICT float_in_work,
                                     f64 *SDSGE_RESTRICT float_out,
                                     i64 *SDSGE_RESTRICT int_work,
                                     i64 *SDSGE_RESTRICT int_out,
                                     const void *ctx) {
  (void)rep_idx;
  (void)int_work;
  const sdsge_mc_jarque_bera_test_ctx *config = ctx;
  return sdsge_mc_test_status(
      sdsge_jb_stat(float_in_work, config->n, float_out), int_out);
}

int sdsge_mc_breusch_pagan_test_runner(const i64 rep_idx,
                                       f64 *SDSGE_RESTRICT float_in_work,
                                       f64 *SDSGE_RESTRICT float_out,
                                       i64 *SDSGE_RESTRICT int_work,
                                       i64 *SDSGE_RESTRICT int_out,
                                       const void *ctx) {
  (void)rep_idx;
  (void)int_work;
  const sdsge_mc_breusch_pagan_test_ctx *config = ctx;
  const f64 *eps = float_in_work;
  f64 *X = float_in_work + config->n;
  const i64 p = config->k + 1;
  f64 *X_aug = X + config->n * config->k;
  f64 *arena = X_aug + config->n * p;

  for (i64 i = 0; i < config->n; ++i) {
    X_aug[i * p] = 1.0;
    for (i64 j = 0; j < config->k; ++j) {
      X_aug[i * p + 1 + j] = X[i * config->k + j];
    }
  }

  return sdsge_mc_test_status(
      sdsge_bp_stat(eps, X_aug, config->n, p, config->robust, arena, float_out),
      int_out);
}

int sdsge_mc_breusch_godfrey_test_runner(const i64 rep_idx,
                                         f64 *SDSGE_RESTRICT float_in_work,
                                         f64 *SDSGE_RESTRICT float_out,
                                         i64 *SDSGE_RESTRICT int_work,
                                         i64 *SDSGE_RESTRICT int_out,
                                         const void *ctx) {
  (void)rep_idx;
  (void)int_work;
  const sdsge_mc_breusch_godfrey_test_ctx *config = ctx;
  const f64 *eps = float_in_work;
  f64 *X = float_in_work + config->n;
  f64 *arena = X + config->n * config->k;
  return sdsge_mc_test_status(sdsge_bg_stat(eps, X, config->n, config->k,
                                            config->lags, arena, float_out),
                              int_out);
}

int sdsge_mc_cusum_test_runner(const i64 rep_idx,
                               f64 *SDSGE_RESTRICT float_in_work,
                               f64 *SDSGE_RESTRICT float_out,
                               i64 *SDSGE_RESTRICT int_work,
                               i64 *SDSGE_RESTRICT int_out, const void *ctx) {
  (void)rep_idx;
  (void)int_work;
  const sdsge_mc_cusum_test_ctx *config = ctx;
  const f64 *y = float_in_work;
  f64 *X = float_in_work + config->n;
  f64 *arena = X + config->n * config->p;
  return sdsge_mc_test_status(
      sdsge_cusum_stat(y, X, config->n, config->p, arena, float_out), int_out);
}

int sdsge_mc_cusumsq_test_runner(const i64 rep_idx,
                                 f64 *SDSGE_RESTRICT float_in_work,
                                 f64 *SDSGE_RESTRICT float_out,
                                 i64 *SDSGE_RESTRICT int_work,
                                 i64 *SDSGE_RESTRICT int_out, const void *ctx) {
  (void)rep_idx;
  (void)int_work;
  const sdsge_mc_cusumsq_test_ctx *config = ctx;
  const f64 *y = float_in_work;
  f64 *X = float_in_work + config->n;
  f64 *arena = X + config->n * config->p;
  i64 n_out;
  return sdsge_mc_test_status(
      sdsge_cusumsq_stat(y, X, config->n, config->p, &n_out, arena, float_out),
      int_out);
}

int sdsge_mc_chow_test_runner(const i64 rep_idx,
                              f64 *SDSGE_RESTRICT float_in_work,
                              f64 *SDSGE_RESTRICT float_out,
                              i64 *SDSGE_RESTRICT int_work,
                              i64 *SDSGE_RESTRICT int_out, const void *ctx) {
  (void)rep_idx;
  (void)int_work;
  const sdsge_mc_chow_test_ctx *config = ctx;
  const f64 *y = float_in_work;
  f64 *X = float_in_work + config->n;
  f64 *arena = X + config->n * config->p;
  return sdsge_mc_test_status(sdsge_chow_stat(y, X, config->n, config->p,
                                              config->t_break, arena,
                                              float_out),
                              int_out);
}
