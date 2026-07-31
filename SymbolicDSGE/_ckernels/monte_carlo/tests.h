#ifndef SDSGE_MC_TESTS_H
#define SDSGE_MC_TESTS_H

#include "../diag/diag.h"
#include "../diag/diag_wald.h"
#include "runner.h"

typedef enum {
  SDSGE_MC_WALD_MEAN = 0,
  SDSGE_MC_WALD_COVARIANCE = 1,
  SDSGE_MC_WALD_SECOND_MOMENT = 2,
} sdsge_mc_wald_kind;

typedef struct {
  const f64 *target;
  i64 n;
  i64 q;
  i64 manual_bandwidth;
  KernelID kernel_id;
  WaldBandwidthMode bandwidth_mode;
  sdsge_mc_wald_kind kind;
} sdsge_mc_wald_test_ctx;

typedef struct {
  i64 n;
  i64 lags;
} sdsge_mc_ljung_box_test_ctx;

typedef struct {
  i64 n;
} sdsge_mc_jarque_bera_test_ctx;

typedef struct {
  i64 n;
  i64 k;
  int robust;
} sdsge_mc_breusch_pagan_test_ctx;

typedef struct {
  i64 n;
  i64 k;
  i64 lags;
} sdsge_mc_breusch_godfrey_test_ctx;

typedef struct {
  i64 n;
  i64 p;
} sdsge_mc_cusum_test_ctx;

typedef sdsge_mc_cusum_test_ctx sdsge_mc_cusumsq_test_ctx;

typedef struct {
  i64 n;
  i64 p;
  i64 t_break;
} sdsge_mc_chow_test_ctx;

/* Generic native Monte Carlo test adapters. Every adapter writes a scalar
 * statistic to ``float_out[0]`` and its diagnostic status to ``int_out[0]``
 * when the latter is supplied, then returns ``SDSGE_MC_RUN_OK``. Diagnostic
 * statuses are result data, not runner failures. ``float_in_work`` starts with
 * the operation inputs and continues with all temporary float storage required
 * by the underlying diagnostic kernel. */
int sdsge_mc_wald_test_runner(
    i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_ljung_box_test_runner(
    i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_jarque_bera_test_runner(
    i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_breusch_pagan_test_runner(
    i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_breusch_godfrey_test_runner(
    i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_cusum_test_runner(
    i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_cusumsq_test_runner(
    i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_chow_test_runner(
    i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
    f64 *SDSGE_RESTRICT float_out, i64 *SDSGE_RESTRICT int_work,
    i64 *SDSGE_RESTRICT int_out, const void *ctx);

#endif /* SDSGE_MC_TESTS_H */
