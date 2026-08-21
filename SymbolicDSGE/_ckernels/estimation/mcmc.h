#ifndef SDSGE_MCMC_H
#define SDSGE_MCMC_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_linalg.h"
#include "../optim/optim.h"
#include "../rng/rng.h"
#include "estimation.h"

/* MCMC INPUTS */

typedef struct {
  i64 n_draws;
  i64 burn_in;
  i64 thin;
  int needs_map; // False if user supplied a mode for the hessian proposal
                 // factor. Otherwise, MAP is computed inside the MCMC driver.
  int adapt;
  i64 adapt_start;
  f64 adapt_epsilon;
  f64 proposal_scale;
  f64 hessian_fd_step_scale;
  f64 hessian_fd_absolute_floor;
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

/* status codes for sdsge_mcmc_result.status */
#define SDSGE_MCMC_OK 0
#define SDSGE_MCMC_EALLOC (-1601)
#define SDSGE_MCMC_EMAP (-1604)

/* HOT LOOP DRIVER */
i64 sdsge_mcmc_run(sdsge_objective_fn logpost, void *obj_ctx, bitgen_t *bg,
                   const f64 *SDSGE_RESTRICT theta0, i64 d,
                   const sdsge_mcmc_options *opt,
                   const sdsge_estimation_options *map_opt,
                   sdsge_mcmc_buffers *buf, sdsge_mcmc_result *out);

#endif /* SDSGE_MCMC_H */
