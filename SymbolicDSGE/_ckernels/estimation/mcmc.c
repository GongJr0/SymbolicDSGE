#include "mcmc.h"
#include <math.h>
#include <string.h>

/* Direct includes for the primitives used here (native-include hygiene). */
#include "../_common/sdsge_common.h" /* max_i64 */
#include "../_common/sdsge_linalg.h" /* sdsge_chol, sdsge_matmul_abt,
                                      * sdsge_matvec, sdsge_matvec_plus_vec */
#include "../optim/optim.h"          /* sdsge_optim_result */
#include "../rng/philox.h" /* sdsge_philox_standard_{normal,uniform}_fill,
                            * sdsge_philox_seed */
#include "estimation.h"    /* sdsge_run_estimation, sdsge_objective_fn */
#include "prior_program.h"
#include "sdsge_time.h" /* sdsge_monotime_sec */

/* Native adaptive random-walk Metropolis mainloop (issue #331).
 *
 * proposal:   prop = current + L @ z,  z ~ N(0, I), where L is the lower
 *             Cholesky of the proposal covariance.
 * adaptation: the Haario et al. (2001) covariance recursion, seeded from
 *             the mode-Hessian covariance and carried forward at (n-1)/n.
 *
 * Memory is O(n_draws*d + d^2) and the loop has no d==1 special case (a 1x1
 * covariance subsumes it).
 *
 * `logpost` returns +logpost (the estimation trampoline passes the objective
 * WITHOUT the optimizer's negation); a BK violation / non-finite eval surfaces
 * as -inf and auto-rejects through the finiteness gate. The BK-violation count
 * lives on the objective's own ctx (the objective owns that counter); the
 * Cython caller reads it off the ctx after the run. */

static inline void sdsge_haario_update(const f64 *SDSGE_RESTRICT x, i64 d,
                                       i64 n, f64 eps, f64 *SDSGE_RESTRICT m,
                                       f64 *SDSGE_RESTRICT dev,
                                       f64 *SDSGE_RESTRICT C) {
  const f64 inv = 1.0 / (f64)n;
  const f64 carry = (f64)(n - 1) * inv;

  /* (x - m_old)(x - m_new)' == carry * (x - m_old)(x - m_old)' exactly, which
   * collapses the scatter increment to a rank-1 update on one centered
   * vector. */
  const f64 w = carry * inv;

  for (i64 i = 0; i < d; ++i) {
    dev[i] = x[i] - m[i];
    m[i] += dev[i] * inv;
  }

  /* Lower triangle only: C is read solely by sdsge_chol, which touches
   * j <= i. The upper triangle keeps the seed's values and goes stale. */
  for (i64 i = 0; i < d; ++i) {
    f64 *SDSGE_RESTRICT Ci = C + i * d;
    const f64 wi = w * dev[i];

    for (i64 j = 0; j <= i; ++j) {
      Ci[j] = carry * Ci[j] + wi * dev[j];
    }
    Ci[i] += inv * eps;
  }
}

static inline void sdsge_proposal_factor(const f64 *SDSGE_RESTRICT C, i64 d,
                                         f64 factor_scale,
                                         f64 *SDSGE_RESTRICT Ltmp,
                                         f64 *SDSGE_RESTRICT L) {
  if (sdsge_chol(C, 0.0, Ltmp, d) != SDSGE_OK) {
    return;
  }
  for (i64 i = 0; i < d * d; ++i) {
    Ltmp[i] *= factor_scale;
  }
  memcpy(L, Ltmp, d * d * sizeof(f64));
}

/* Sampler's unconstrained theta into the parameter space via inverse
 * transforms. Returns the log-jacobian the prior carries at this theta. The
 * chain folds it into the logpost (include_logjac == 1) and parameter space
 * reporting needs it subtracted out.
 *
 * A std with no prior still walks in log space, and a CPC block with
 * no LKJ is still a reparameterization, both have logjac=0.0 but are
 * reparameterizations.
 *
 * ``Lblk`` is scratch for the widest block's Cholesky factor. */
static inline f64 sdsge_unconstrained_to_params(const sdsge_obj_common *b,
                                                const f64 *SDSGE_RESTRICT theta,
                                                f64 *SDSGE_RESTRICT Lblk,
                                                f64 *SDSGE_RESTRICT out) {
  const sdsge_prior_tables *pr = &b->prior;

  f64 x, logjac, lj = 0.0;
  i64 len;
  for (i64 i = 0; i < pr->n_theta; ++i) {
    len = sdsge_block_run_len(pr, i);
    if (len) {
      i += len - 1; // skip the block fill
      continue;
    }
    sdsge_transform_inverse_and_logjac(pr->transform_codes[i],
                                       pr->transform_params +
                                           i * SDSGE_N_TRANSFORM_PARAMS,
                                       theta[i], &x, &logjac);
    out[i] = x;
    lj += logjac;
  }

  const sdsge_cov_spec *specs[2] = {&b->q.spec, &b->r.spec};
  for (i64 s = 0; s < 2; ++s) {
    const sdsge_cov_spec *sp = specs[s];
    if (!sdsge_spec_has_block(sp)) {
      continue;
    }
    f64 block_lj;
    sdsge_corr_entries_from_unconstrained(theta + sp->block_theta_off, sp->K,
                                          Lblk, out + sp->block_theta_off,
                                          &block_lj);
    lj += block_lj;
  }

  return lj;
}

static void sdsge_mcmc_init_out(sdsge_mcmc_result *out,
                                const sdsge_mcmc_options *opt) {
  out->total_steps = opt->burn_in + opt->n_draws * opt->thin;
  out->accepted_burn_in = 0;
  out->accepted_in_sample = 0;
  out->burn_in_t = 0.0;
  out->in_sample_t = 0.0;
  out->bk_violations = 0;
  out->status = SDSGE_MCMC_OK;
  out->message = "ok";
}

static void sdsge_mcmc_nan_buffers(sdsge_mcmc_buffers *buf, i64 d,
                                   i64 n_draws) {
  memset(buf->kept, 0xFF, n_draws * d * sizeof(f64));
  memset(buf->kept_lp, 0xFF, n_draws * sizeof(f64));
  memset(buf->kept_lj, 0xFF, n_draws * sizeof(f64));
  memset(buf->init_theta, 0xFF, d * sizeof(f64));
}

static void sdsge_mcmc_map_point(void *obj_ctx, i64 n_theta,
                                 f64 *SDSGE_RESTRICT theta,
                                 const sdsge_estimation_options *map_opt,
                                 sdsge_mcmc_result *out) {
  sdsge_estimation_result map_out;
  /* Defensive, blocks the covariance solve (compute_cov=0 in cython). */
  map_out.vcov = NULL;
  sdsge_run_estimation(obj_ctx, n_theta, theta, map_opt, &map_out);
  if (!map_out.base.success) {
    out->status = SDSGE_MCMC_EMAP;
    out->message =
        map_out.base.message != NULL ? map_out.base.message : "MAP failed";
  }
}

static void
sdsge_mcmc_resolve_pcov(sdsge_objective_fn logpost, void *obj_ctx,
                        const sdsge_mcmc_options *opt, i64 d,
                        const f64 *SDSGE_RESTRICT pcov,
                        f64 *SDSGE_RESTRICT current, f64 *SDSGE_RESTRICT L,
                        f64 *C, // aliases hessian_work by design, no *restrict
                        f64 *hessian_work, sdsge_mcmc_result *out) {
  if (opt->needs_hessian) {
    i64 status = sdsge_estimation_cov_factor(
        logpost, obj_ctx, current, d, opt->hessian_fd_step_scale,
        opt->hessian_fd_absolute_floor, L, hessian_work);
    if (status != SDSGE_ESTIMATION_OK) {
      out->status = status;
      out->message = status == SDSGE_ESTIMATION_ENOTSPD
                         ? "MAP Hessian is not positive definite"
                         : "MCMC Hessian construction failed";
      return;
    }
    sdsge_matmul_abt(L, L, C, d, d, d); /* C = L @ L^T */
  } else {
    if (sdsge_chol(pcov, 0.0, L, d) != SDSGE_OK) {
      out->status = SDSGE_ESTIMATION_ENOTSPD;
      out->message =
          "User-supplied proposal covariance is not positive definite";
      return;
    }
    memcpy(C, pcov, d * d * sizeof(f64));
  }
}

static void sdsge_mcmc_init_point(sdsge_objective_fn logpost, void *obj_ctx,
                                  const sdsge_mcmc_options *opt,
                                  sdsge_philox_state *st, i64 d, f64 *current,
                                  const f64 *L, f64 *z, sdsge_mcmc_result *out,
                                  sdsge_mcmc_buffers *buf) {
  f64 scale = opt->init_scale;
  i64 scale_deflation_tries = 10;
  i64 nodeflate_iter = 100;

  /* The deflation at a factor of 0.8 produces 0.8^10 = 0.107 if the final
   * reduction is reached. That is, if we don't find a finite logpost at ~10% of
   * the user-supplied scale, the raise is taken to the user and they should
   * decide to update their init_scale accordingly. */

  while (1) {
    sdsge_philox_standard_normal_fill(st, d, z);
    for (i64 i = 0; i < d; ++i) {
      z[i] *= scale;
    }
    sdsge_matvec_plus_vec(L, z, current, buf->init_theta, d, d);
    const f64 cand_lp = logpost(buf->init_theta, obj_ctx);

    if (isfinite(cand_lp)) {
      memcpy(current, buf->init_theta, d * sizeof(f64));
      return;
    }

    if (--nodeflate_iter == 0) {
      if (scale_deflation_tries == 0) {
        out->status = SDSGE_MCMC_EPOINTINIT;
        out->message =
            "MCMC initial point search failed: no finite logpost found";
        return;
      }

      scale *= 0.8;
      --scale_deflation_tries;
      nodeflate_iter = 10;
    }
  }
}

static void sdsge_mcmc_kernel(
    sdsge_objective_fn logpost, void *obj_ctx, const sdsge_mcmc_options *opt,
    sdsge_philox_state *st, i64 d, f64 *SDSGE_RESTRICT current,
    f64 *SDSGE_RESTRICT L, f64 *SDSGE_RESTRICT prop, f64 *SDSGE_RESTRICT z,
    f64 *SDSGE_RESTRICT m, f64 *SDSGE_RESTRICT dev, f64 *SDSGE_RESTRICT C,
    f64 *SDSGE_RESTRICT Ltmp, sdsge_mcmc_buffers *buf, sdsge_mcmc_result *out) {

  const sdsge_obj_common *cb = (const sdsge_obj_common *)obj_ctx;
  const i64 total_steps = opt->burn_in + opt->n_draws * opt->thin;
  const f64 factor_scale = sqrt((2.38 * 2.38) / (f64)d) * opt->proposal_scale;
  for (i64 i = 0; i < d * d; ++i) {
    L[i] *= factor_scale;
  }

  memset(m, 0, d * sizeof(f64));
  f64 cur_lp = logpost(current, obj_ctx);

  i64 accepted = 0;
  i64 accepted_burn_in = 0;
  i64 keep_i = 0;
  i64 next_keep = opt->burn_in;      /* init with post-burn-in */
  i64 next_adapt = opt->adapt_start; /* init with first adaptation */

  /* clock stores */
  const f64 start_cp = sdsge_monotime_sec();
  f64 burn_cp = start_cp; /* == start_cp if burn_in == 0, else set at first
                          post-burn-in draw */
  f64 halt_cp;

  for (i64 t = 0; t < total_steps; ++t) {
    /* Timing and acceptance accounting: start the clock at first
     * post-burn-in draw and record the current accepted count for warmup
     * diagnostics. */
    if (t == opt->burn_in) {
      burn_cp = sdsge_monotime_sec();
      accepted_burn_in = accepted;
    }

    /* Propose: prop = current + L @ z,  z ~ N(0, I). */
    sdsge_philox_standard_normal_fill(st, d, z);
    sdsge_matvec(L, z, prop, d, d);
    for (i64 i = 0; i < d; ++i) {
      prop[i] += current[i];
    }

    const f64 prop_lp = logpost(prop, obj_ctx);
    if (isfinite(prop_lp)) {
      const f64 log_alpha = prop_lp - cur_lp;
      f64 u;
      sdsge_philox_standard_uniform_fill(st, 1, &u);
      if (log(u) < log_alpha) {
        memcpy(current, prop, d * sizeof(f64));
        cur_lp = prop_lp;
        ++accepted;
      }
    }

    /* Haario adaptation over the specified span. */
    if (opt->adapt && t < opt->adapt_end) {
      const i64 n = t + 1;
      if (t < opt->adapt_start) {
        const f64 inv = 1.0 / (f64)n;
        for (i64 i = 0; i < d; ++i) {
          m[i] += (current[i] - m[i]) * inv;
        }
      } else {
        sdsge_haario_update(current, d, n, opt->adapt_epsilon, m, dev, C);
        if (t == next_adapt) {
          next_adapt += opt->adapt_interval;
          sdsge_proposal_factor(C, d, factor_scale, Ltmp, L);
        }
      }
    }

    /* Keep post-burn-in draws at the thinning cadence. */
    if (t == next_keep) {
      f64 lj = sdsge_unconstrained_to_params(cb, current, Ltmp,
                                             &buf->kept[keep_i * d]);
      buf->kept_lp[keep_i] = cur_lp - lj;
      buf->kept_lj[keep_i] = lj;
      ++keep_i;
      next_keep += opt->thin;
    }
  }
  halt_cp = sdsge_monotime_sec();
  out->accepted_burn_in = accepted_burn_in;
  out->accepted_in_sample = accepted - accepted_burn_in;
  out->burn_in_t = burn_cp - start_cp;
  out->in_sample_t = halt_cp - burn_cp;
  out->bk_violations = cb->bk_violations;
}

arena_size sdsge_chain_arena_size(i64 d, const void *obj_ctx) {
  const sdsge_obj_common *cb = (const sdsge_obj_common *)obj_ctx;

  const i64 kblk =
      max_i64(sdsge_spec_has_block(&cb->q.spec) ? cb->q.spec.K : 0,
              sdsge_spec_has_block(&cb->r.spec) ? cb->r.spec.K : 0);
  const size_t nm = d * d;
  const size_t nlast = (size_t)max_i64(d * d, kblk * kblk);
  return make_sizer(5 * d + 2 * nm + nlast, 0);
}
arena_size sdsge_mcmc_arena_size(i64 d, i64 n_chains, const void *obj_ctx) {
  arena_size single = sdsge_chain_arena_size(d, obj_ctx);
  return make_sizer(single.n_float * n_chains, 0);
}
i64 sdsge_mcmc_run(sdsge_objective_fn logpost, sdsge_obj_ctx *obj_ctx,
                   const f64 *SDSGE_RESTRICT theta0, i64 d, i64 n_chains,
                   const f64 *SDSGE_RESTRICT pcov,
                   const sdsge_mcmc_options *opt,
                   const sdsge_estimation_options *map_opt,
                   sdsge_mcmc_buffers *buf, sdsge_mcmc_result *out, f64 *wall_t,
                   f64 *init_t, f64 *SDSGE_RESTRICT work) {
  *init_t = 0.0;
  *wall_t = 0.0;
  f64 start_cp = sdsge_monotime_sec();

  /* Initialize workpace for the first chain. */
  f64 *base_current = work;
  f64 *base_L = base_current + d;
  f64 *base_C = base_L + d * d;
  memcpy(base_current, theta0, d * sizeof(f64));

  sdsge_mcmc_result *base_out = out;
  sdsge_mcmc_init_out(base_out, opt);

  /* MAP point as theta0 on first chain if requested. Following chains are
   * standard normal perturbations of the first chain's initial point. */
  if (opt->needs_map) {
    sdsge_mcmc_map_point(&obj_ctx[0], d, base_current, map_opt, base_out);
    if (base_out->status != SDSGE_MCMC_OK) {
      sdsge_mcmc_nan_buffers(&buf[0], d, opt->n_draws);

      for (i64 c = 1; c < n_chains; ++c) {
        sdsge_mcmc_init_out(&out[c], opt);
        out[c].status = base_out->status;
        out[c].message = base_out->message;
        sdsge_mcmc_nan_buffers(&buf[c], d, opt->n_draws);
      }
      return base_out->status;
    }
  }
  memcpy(buf[0].init_theta, base_current, d * sizeof(f64));

  /* Proposal covariance of base, feeds successive chains directly. */
  sdsge_mcmc_resolve_pcov(logpost, &obj_ctx[0], opt, d, pcov, base_current,
                          base_L, base_C, base_C, base_out);

  if (base_out->status != SDSGE_MCMC_OK) {
    sdsge_mcmc_nan_buffers(&buf[0], d, opt->n_draws);

    for (i64 c = 1; c < n_chains; ++c) {
      sdsge_mcmc_init_out(&out[c], opt);
      out[c].status = base_out->status;
      out[c].message = base_out->message;
      sdsge_mcmc_nan_buffers(&buf[c], d, opt->n_draws);
    }
    return base_out->status;
  }

  /* Initialize the remaining chains' workspace and out. */
  arena_size work_per = sdsge_chain_arena_size(d, &obj_ctx[0]);
  for (i64 c = 1; c < n_chains; ++c) {
    memcpy(work + c * work_per.n_float, work, work_per.n_float * sizeof(f64));
    sdsge_mcmc_init_out(&out[c], opt);
  }

  /* Get perturbed theta0 per each non-first chain. */
  if (n_chains > 1) {
#pragma omp parallel for schedule(static)
    for (i64 c = 1; c < n_chains; ++c) {
      sdsge_philox_state st;
      // (seed, modcode, chain, <0:sampler, 1:init_point>)
      sdsge_philox_seed(&st, opt->seed, SDSGE_MCMC_MODCODE, c, 1);
      f64 *current = work + c * work_per.n_float;
      f64 *L = current + d;
      f64 *z = L + 2 * d * d + d;
      sdsge_mcmc_init_point(logpost, &obj_ctx[c], opt, &st, d, current, L, z,
                            &out[c], &buf[c]);
    }
  }

  i64 err = 0;
  for (i64 c = 0; c < n_chains; ++c) {
    if (out[c].status != SDSGE_MCMC_OK) {
      // generic, chains that failed are located with the out[c].status
      err = 1;
    }
  }
  if (err) {
    for (i64 c = 0; c < n_chains; ++c) {
      sdsge_mcmc_nan_buffers(&buf[c], d, opt->n_draws);
    }
    return SDSGE_MCMC_EPOINTINIT;
  }

  f64 init_cp = sdsge_monotime_sec();

  /* Kernel entry */
#pragma omp parallel for if (n_chains > 1) schedule(static)
  for (i64 c = 0; c < n_chains; ++c) {
    sdsge_philox_state st;
    sdsge_philox_seed(&st, opt->seed, SDSGE_MCMC_MODCODE, c, 0);
    f64 *current = work + c * work_per.n_float;
    f64 *L = current + d;
    f64 *C = L + d * d;
    f64 *prop = C + d * d;
    f64 *z = prop + d;
    f64 *m = z + d;
    f64 *dev = m + d;
    f64 *Ltmp = dev + d;

    sdsge_mcmc_kernel(logpost, &obj_ctx[c], opt, &st, d, current, L, prop, z, m,
                      dev, C, Ltmp, &buf[c], &out[c]);
  }
  for (i64 c = 0; c < n_chains; ++c) {
    if (out[c].status != SDSGE_MCMC_OK) {
      return SDSGE_MCMC_EMULTICHAIN;
    }
  }
  *init_t = init_cp - start_cp;
  *wall_t = sdsge_monotime_sec() - start_cp;
  return SDSGE_MCMC_OK;
}
