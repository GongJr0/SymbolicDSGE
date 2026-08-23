#include "mcmc.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Direct includes for the primitives used here (native-include hygiene). */
#include "../_common/sdsge_common.h" /* max_i64 */
#include "../_common/sdsge_linalg.h" /* sdsge_chol, sdsge_backward_subst_chol_t,
                                        sdsge_matmul_abt, sdsge_matvec */
#include "../optim/optim.h"          /* sdsge_optim_result */
#include "../rng/rng.h" /* sdsge_rng_standard_{normal,uniform}_fill */
#include "estimation.h" /* sdsge_run_estimation, sdsge_objective_fn */
#include "prior_program.h"

/* Native adaptive random-walk Metropolis mainloop (issue #331).
 *
 * A clean transcription of the numpy-era Estimator.mcmc loop, taken fully
 * native under the statistical-equivalence contract (option b): numpy's draws
 * stay bit-exact (the standard-normal / uniform fills advance numpy's own PCG64
 * via `bg`), but the two pieces of deterministic linear algebra are ours, NOT
 * numpy's:
 *
 *   - proposal:   prop = current + L @ z,  z ~ N(0, I), where L is OUR lower
 *                 Cholesky of the proposal covariance (not numpy's SVD-based
 *                 multivariate_normal map).
 *   - adaptation: the Haario et al. (2001) covariance recursion, seeded from
 *                 the mode-Hessian covariance and carried forward at (n-1)/n,
 *                 not a batch np.cov recomputed from stored history.
 *
 * The recursion spans the whole chain rather than burn-in alone, so the
 * proposal is never frozen and an early, under-dispersed history is damped by
 * the seed at weight adapt_start/n instead of replacing it outright.
 *
 * Consequences of dropping stored history: memory is O(n_draws*d + d^2) and the
 * loop has no d==1 special case (a 1x1 covariance subsumes it). The resulting
 * chain is a principled, reproducible native stream, statistically equivalent
 * to the numpy chain (matched stationary marginals + acceptance), not a
 * draw-for-draw reproduction of it.
 *
 * `logpost` returns +logpost (the estimation trampoline passes the objective
 * WITHOUT the optimizer's negation); a BK violation / non-finite eval surfaces
 * as -inf and auto-rejects through the finiteness gate. The BK-violation count
 * lives on the objective's own ctx (the objective owns that counter); the
 * Cython caller reads it off the ctx after the run, so `out->bk_violations` is
 * left 0 here to keep this loop a generic sampler over an opaque `obj_ctx`. */

static inline void sdsge_haario_update(const f64 *SDSGE_RESTRICT x, i64 d,
                                       i64 n, f64 eps, f64 *SDSGE_RESTRICT m,
                                       f64 *SDSGE_RESTRICT m0,
                                       f64 *SDSGE_RESTRICT C) {
  const f64 inv = 1.0 / (f64)n;
  memcpy(m0, m, d * sizeof(f64));
  for (i64 i = 0; i < d; ++i) {
    m[i] = (m0[i] * (f64)(n - 1) + x[i]) * inv;
  }
  const f64 carry = (f64)(n - 1) * inv;
  for (i64 i = 0; i < d; ++i) {
    for (i64 j = 0; j < d; ++j) {
      const f64 inc = (f64)n * m0[i] * m0[j] - (f64)(n + 1) * m[i] * m[j] +
                      x[i] * x[j] + (i == j ? eps : 0.0);

      C[i * d + j] = carry * C[i * d + j] + inc * inv;
    }
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

/* A spec whose correlation is a live CPC block, the only shape the keep-time
 * inversion has work to do on and the only one that needs block scratch. */
static inline int sdsge_spec_has_block(const sdsge_cov_spec *sp) {
  return !sp->is_constant && sp->corr_from_block;
}

/* One draw, out of the sampler's unconstrained theta and into the parameters
 * themselves: every estimated scalar through its inverse transform, every CPC
 * block through the correlation its z encodes. Returns the log-jacobian the
 * prior carries at this theta, which is what the chain's logpost folded in
 * (include_logjac == 1) and what the caller takes back out to report a density
 * over the parameters.
 *
 * The two passes read different tables on purpose. Values come off the param
 * map and the cov specs, which cover every estimated entry; the jacobian comes
 * off the prior tables, which cover only the entries carrying a density. A std
 * with no prior still walks in log space, and a CPC block with no LKJ is still
 * a reparameterization, so neither is reported raw, and neither owes a
 * jacobian. ``Lblk`` is scratch for the widest block's Cholesky factor. */
static inline f64 sdsge_unconstrained_to_params(const sdsge_obj_common *b,
                                                const f64 *SDSGE_RESTRICT theta,
                                                f64 *SDSGE_RESTRICT Lblk,
                                                f64 *SDSGE_RESTRICT out) {
  const sdsge_prior_tables *pr = &b->prior;
  f64 lj = 0.0;
  f64 x, logjac;

  for (i64 i = 0; i < b->pmap.n_scalars; ++i) {
    const sdsge_scalar_scatter *sc = &b->pmap.scalars[i];
    sdsge_transform_inverse_and_logjac(sc->transform_code, sc->transform_params,
                                       theta[sc->theta_idx], &x, &logjac);
    out[sc->theta_idx] = x;
  }

  for (i64 i = 0; i < pr->n_scalar; ++i) {
    sdsge_transform_inverse_and_logjac(
        pr->scalar_transform_codes[i],
        pr->scalar_transform_params + i * SDSGE_N_TRANSFORM_PARAMS,
        theta[pr->scalar_indices[i]], &x, &logjac);
    lj += logjac;
  }

  const sdsge_cov_spec *specs[2] = {&b->q_spec, &b->r_spec};
  for (i64 s = 0; s < 2; ++s) {
    const sdsge_cov_spec *sp = specs[s];
    if (!sdsge_spec_has_block(sp)) {
      continue;
    }
    f64 block_lj;
    sdsge_corr_entries_from_unconstrained(theta + sp->block_theta_off, sp->K,
                                          Lblk, out + sp->block_theta_off,
                                          &block_lj);
    /* Only a block carrying an LKJ put a jacobian into the logpost, and the
     * prior tables list exactly those. */
    for (i64 k = 0; k < pr->n_blocks; ++k) {
      if (pr->matrix_offsets[k] == sp->block_theta_off) {
        lj += block_lj;
        break;
      }
    }
  }

  return lj;
}

i64 sdsge_mcmc_run(sdsge_objective_fn logpost, void *obj_ctx, bitgen_t *bg,
                   const f64 *SDSGE_RESTRICT theta0, i64 d,
                   const f64 *SDSGE_RESTRICT hessian,
                   const sdsge_mcmc_options *opt,
                   const sdsge_estimation_options *map_opt,
                   sdsge_mcmc_buffers *buf, sdsge_mcmc_result *out) {
  const i64 total_steps = opt->burn_in + opt->n_draws * opt->thin;
  out->total_steps = total_steps;
  out->n_accepted = 0;
  out->bk_violations = 0; /* filled by the caller off the objective ctx */
  out->status = SDSGE_MCMC_OK;
  out->message = "ok";

  /* Widest CPC block the keep-time inversion has to factor. */
  const sdsge_obj_common *cb = (const sdsge_obj_common *)obj_ctx;
  const i64 kblk =
      max_i64(sdsge_spec_has_block(&cb->q_spec) ? cb->q_spec.K : 0,
              sdsge_spec_has_block(&cb->r_spec) ? cb->r_spec.K : 0);

  /* One workspace allocation up front, freed on return.
   * 5 vectors of d and 3 matrices of d*d. The last matrix is the refactor
   * scratch, dead by the time a draw is kept, so the block inversion borrows
   * it; it is widened only when K*K outgrows d*d, which takes a block estimated
   * with nothing alongside it. */
  const size_t nm = d * d;
  const size_t nlast = (size_t)max_i64(d * d, kblk * kblk);
  f64 *work = (f64 *)malloc((5 * d + 2 * nm + nlast) * sizeof(f64));
  if (work == NULL) {
    out->status = SDSGE_MCMC_EALLOC;
    out->message = "mcmc workspace allocation failed";
    return SDSGE_MCMC_EALLOC;
  }

  f64 *current = work;
  f64 *L = current + d;
  f64 *prop = L + nm;
  f64 *z = prop + d;
  f64 *m = z + d;
  f64 *m0 = m + d;
  f64 *C = m0 + d;
  f64 *Ltmp = C + nm;

  const f64 scale = (2.38 * 2.38) / (f64)d;

  /* Initial proposal: cov0 = scale * H^-1, L = chol(cov0). The Hessian is
   * computed at the MAP as a finite-difference approximation of the negative
   * log-posterior.
   *
   * MAP is only computed if the user did not supply a mode for the Hessian
   * proposal factor.
   */

  memcpy(current, theta0, d * sizeof(f64));
  if (opt->needs_map) {
    /* The sampler builds its own factor below, so the MAP call is asked for
     * the point only; vcov stays NULL. */
    sdsge_estimation_result map_out;
    map_out.vcov = NULL;
    sdsge_run_estimation(obj_ctx, d, current, map_opt, &map_out);
    if (!map_out.base.success) {
      out->status = SDSGE_MCMC_EMAP;
      out->message =
          map_out.base.message != NULL ? map_out.base.message : "MAP failed";
      free(work);
      return SDSGE_MCMC_EMAP;
    }
  }

  // needs_hessian is guaranteed to be True if needs_map is
  // True. This block being separately gated allows a user to skip the MAP and
  // get the Hessian at a user-supplied point, never to skip both unless the
  // hessian itself is supplied.
  if (opt->needs_hessian) {
    // The allocation past `L` is dead at cov_factor time, and it is the exact
    // size the factor needs. `prop` is therefore passed as the work buffer, and
    // `L` is the output.
    f64 *hessian_work = prop;
    const i64 hessian_status = sdsge_estimation_cov_factor(
        logpost, obj_ctx, current, d, opt->hessian_fd_step_scale,
        opt->hessian_fd_absolute_floor, L, hessian_work);
    if (hessian_status != SDSGE_ESTIMATION_OK) {
      out->status = hessian_status;
      out->message = hessian_status == SDSGE_ESTIMATION_ENOTSPD
                         ? "MAP Hessian is not positive definite"
                         : "MCMC Hessian construction failed";
      free(work);
      return hessian_status;
    }
    sdsge_matmul_abt(L, L, C, d, d, d);

  } else {
    // User supplied a Hessian; factor it for the proposal and copy directly
    // into the covariance buffer.
    if (sdsge_chol_upper(hessian, 0.0, L, d) != SDSGE_OK) {
      out->status = SDSGE_ESTIMATION_ENOTSPD;
      out->message = "User-supplied Hessian is not positive definite";
      free(work);
      return SDSGE_ESTIMATION_ENOTSPD;
    }
    memcpy(C, hessian, nm * sizeof(f64));
  }

  const f64 factor_scale = sqrt(scale);
  for (i64 i = 0; i < d * d; ++i) {
    L[i] *= factor_scale;
  }

  for (i64 i = 0; i < d; ++i) {
    m[i] = 0.0;
  }
  i64 n = 0;

  f64 cur_lp = logpost(current, obj_ctx);

  i64 accepted = 0;
  i64 keep_i = 0;

  for (i64 t = 0; t < total_steps; ++t) {
    /* Propose: prop = current + L @ z,  z ~ N(0, I). */
    sdsge_rng_standard_normal_fill(bg, d, z);
    sdsge_matvec(L, z, prop, d, d);
    for (i64 i = 0; i < d; ++i) {
      prop[i] += current[i];
    }

    const f64 prop_lp = logpost(prop, obj_ctx);
    if (isfinite(prop_lp)) {
      const f64 log_alpha = prop_lp - cur_lp;
      f64 u;
      sdsge_rng_standard_uniform_fill(bg, 1, &u);
      if (log(u) < log_alpha) {
        memcpy(current, prop, d * sizeof(f64));
        cur_lp = prop_lp;
        ++accepted;
      }
    }

    /* Haario adaptation over the whole chain: below adapt_start only the
     * running mean accumulates, above it the covariance recursion runs and
     * the factor is refactored every step. */
    if (opt->adapt) {
      ++n;
      if (n <= opt->adapt_start) {
        for (i64 i = 0; i < d; ++i) {
          m[i] += (current[i] - m[i]) / (f64)n;
        }
      } else {
        sdsge_haario_update(current, d, n, opt->adapt_epsilon, m, m0, C);
        sdsge_proposal_factor(C, d, factor_scale, Ltmp, L);
      }
    }

    /* Keep post-burn-in draws at the thinning cadence. */
    if (t >= opt->burn_in && (t - opt->burn_in) % opt->thin == 0) {
      f64 lj = sdsge_unconstrained_to_params(cb, current, Ltmp,
                                             &buf->kept[keep_i * d]);
      buf->kept_lp[keep_i] = cur_lp - lj;
      buf->kept_lj[keep_i] = lj;
      ++keep_i;
    }
  }

  out->n_accepted = accepted;
  free(work);
  return SDSGE_MCMC_OK;
}
