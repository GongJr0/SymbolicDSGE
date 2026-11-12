#ifndef SDSGE_MCMC_H
#define SDSGE_MCMC_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_linalg.h"
#include "../optim/optim.h"
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
  i64 adapt_end;
  i64 adapt_interval;
  f64 adapt_epsilon;
  f64 proposal_scale;
  int needs_hessian; // False if user supplied a hessian for the hessian
                     // proposal factor. Otherwise, Hessian is computed inside
                     // the MCMC driver.

  f64 hessian_fd_step_scale;
  f64 hessian_fd_absolute_floor;
  u64 seed;
} sdsge_mcmc_options;

typedef struct {
  f64 *kept;
  f64 *kept_lp;
  f64 *kept_lj;
} sdsge_mcmc_buffers;

/* MCMC OUTPUTS */

typedef struct {
  // Acceptance
  i64 accepted_burn_in;
  i64 accepted_in_sample;
  // Step count
  i64 total_steps;
  // Errors
  i64 bk_violations;
  i64 status; /* ERROR CODE */
  const char *message;
  // Timing
  f64 burn_in_t;
  f64 in_sample_t;
} sdsge_mcmc_result;

/* Module Code for Philox RNG */
#define SDSGE_MCMC_MODCODE ((u64)1600 << 48)

/* status codes for sdsge_mcmc_result.status */
#define SDSGE_MCMC_OK 0
#define SDSGE_MCMC_EMAP (-1604)
#define SDSGE_MCMC_EMULTICHAIN (-1605)
/* HOT LOOP DRIVER */
arena_size sdsge_mcmc_arena_size(i64 d, const void *obj_ctx);
i64 sdsge_mcmc_run(sdsge_objective_fn logpost, void *obj_ctx,
                   const f64 *SDSGE_RESTRICT theta0, i64 d,
                   const f64 *SDSGE_RESTRICT pcov,
                   const sdsge_mcmc_options *opt,
                   const sdsge_estimation_options *map_opt,
                   sdsge_mcmc_buffers *buf, sdsge_mcmc_result *out,
                   f64 *SDSGE_RESTRICT work);

#endif /* SDSGE_MCMC_H */
