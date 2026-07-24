#ifndef SDSGE_MCMC_H
#define SDSGE_MCMC_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_linalg.h"
#include "../optim/optim.h"
#include "../rng/rng.h"

/* MCMC INPUTS */

typedef struct {
  i64 n_draws;
  i64 burn_in;
  i64 thin;
  int adapt;
  i64 adapt_start;
  f64 adapt_epsilon;
  f64 proposal_scale;
} sdsge_mcmc_options;

typedef struct {
  f64 *kept;
  f64 *kept_lp;
} sdsge_mcmc_buffers;

/* MCMC OUTPUTS */

typedef struct {
  i64 n_accepted;
  i64 total_steps;
  i64 bk_violations;
  i64 status; /* ERROR CODE */
  const char *message;
} sdsge_mcmc_result;

/* HOT LOOP DRIVER */
i64 sdsge_mcmc_run(sdsge_objective_fn logpost, void *obj_ctx, bitgen *bg,
                   const f64 *theta0, i64 d, const sdsge_mcmc_options *opt,
                   sdsge_mcmc_buffers *buf, sdsge_mcmc_result *out);

#endif /* SDSGE_MCMC_H */
