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
  int adapt;
  i64 adapt_start;
  i64 adapt_interval; /* update the proposal cov every interval-th step (>= 1)*/
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
#define SDSGE_MCMC_EHESSIAN (-1602)
#define SDSGE_MCMC_ENOTSPD (-1603)
#define SDSGE_MCMC_EMAP (-1604)

/* Build an unscaled proposal factor at `theta` from the negative-log-posterior
 * Hessian. The factor satisfies factor * factor^T = H^-1, where
 * H = -d^2 logpost(theta). `fd_step_scale` and `fd_absolute_floor` define
 * h_i = max(abs(theta_i), fd_absolute_floor) * DBL_EPSILON^(1/6) *
 * fd_step_scale. The caller owns the d*d row-major `factor` output. */
i64 sdsge_mcmc_hessian_proposal_factor(sdsge_objective_fn logpost,
                                       void *obj_ctx, const f64 *theta, i64 d,
                                       f64 fd_step_scale, f64 fd_absolute_floor,
                                       f64 *factor);

/* HOT LOOP DRIVER */
i64 sdsge_mcmc_run(sdsge_objective_fn logpost, void *obj_ctx, bitgen_t *bg,
                   const f64 *theta0, i64 d, const sdsge_mcmc_options *opt,
                   const sdsge_estimation_options *map_opt,
                   sdsge_mcmc_buffers *buf, sdsge_mcmc_result *out);

#endif /* SDSGE_MCMC_H */
