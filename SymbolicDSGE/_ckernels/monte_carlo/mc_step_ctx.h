#ifndef MC_STEP_CTX_H
#define MC_STEP_CTX_H

#include "core_steps.h"
#include "regression.h"
#include "tests.h"
#include "transforms.h"

typedef union {
  // core_steps.h
  sdsge_mc_payload_step_ctx payload;
  sdsge_mc_passthrough_step_ctx passthrough;
  sdsge_mc_raw_model_data_step_ctx raw_model_data;

  sdsge_mc_simulate_order1_step_ctx simulate_order1;
  sdsge_mc_simulate_order2_step_ctx simulate_order2;

  sdsge_mc_filter_linear_step_ctx filter_linear;
  sdsge_mc_filter_extended_step_ctx filter_extended;
  sdsge_mc_filter_unscented_step_ctx filter_unscented;

  // regression.h
  sdsge_mc_ols_step_ctx ols;

  sdsge_mc_ridge_step_ctx ridge;
  sdsge_mc_ridge_gs_step_ctx ridge_gs;

  sdsge_mc_lasso_step_ctx lasso;
  sdsge_mc_lasso_gs_step_ctx lasso_gs;

  sdsge_mc_elastic_net_step_ctx elastic_net;
  sdsge_mc_elastic_net_gs_step_ctx elastic_net_gs;

  // tests.h
  sdsge_mc_wald_test_ctx wald;
  sdsge_mc_ljung_box_test_ctx ljung_box;
  sdsge_mc_jarque_bera_test_ctx jarque_bera;
  sdsge_mc_breusch_pagan_test_ctx breusch_pagan;
  sdsge_mc_breusch_godfrey_test_ctx breusch_godfrey;
  sdsge_mc_cusum_test_ctx cusum;
  sdsge_mc_cusumsq_test_ctx cusumsq;
  sdsge_mc_chow_test_ctx chow;

  // transforms.h
  sdsge_mc_standardize_step_ctx standardize;
  sdsge_mc_log_step_ctx log;
  sdsge_mc_log_diff_step_ctx log_diff;
  sdsge_mc_diff_step_ctx diff;
  sdsge_mc_rolling_mean_step_ctx rolling_mean;
  sdsge_mc_rolling_var_step_ctx rolling_var;
  sdsge_mc_rolling_std_step_ctx rolling_std;
  sdsge_mc_user_transform_step_ctx user_transform;

} sdsge_mc_step_ctx;

#endif // MC_STEP_CTX_H
