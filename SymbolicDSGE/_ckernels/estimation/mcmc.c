#include "mcmc.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Direct includes for the primitives used here (native-include hygiene). */
#include "../_common/sdsge_linalg.h" /* sdsge_chol, sdsge_backward_subst_chol_t,
                                        sdsge_matmul_abt, sdsge_matvec */
#include "../optim/optim.h"          /* sdsge_optim_result */
#include "../rng/rng.h" /* sdsge_rng_standard_{normal,uniform}_fill */
#include "estimation.h" /* sdsge_run_estimation, sdsge_objective_fn */

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

/* Hessian-derived initial proposal factor. This is intentionally separate from
 * the sampling loop: callers may prepare it once at a MAP point and reuse it
 * across chains. The objective returns +logpost, so every finite-difference
 * expression below is written directly for H = -d^2 logpost.
 *
 * The off-diagonal stencil matches Dynare's hessian.m: it reuses the two
 * coordinate-direction evaluations and needs only the (++), (--) pair for
 * each i < j. After H = L L^T, solving L^T X = I gives X X^T = H^-1. */
i64 sdsge_mcmc_hessian_proposal_factor(sdsge_objective_fn logpost,
                                       void *obj_ctx,
                                       const f64 *SDSGE_RESTRICT theta, i64 d,
                                       f64 fd_step_scale, f64 fd_absolute_floor,
                                       f64 *SDSGE_RESTRICT factor,
                                       f64 *SDSGE_RESTRICT work) {
  if (d <= 0 || fd_step_scale <= 0.0 || fd_absolute_floor <= 0.0) {
    return SDSGE_MCMC_EHESSIAN;
  }
  const f64 fd_relative_step = sqrt(cbrt(DBL_EPSILON)) * fd_step_scale;
  if (!isfinite(fd_relative_step) || fd_relative_step <= 0.0) {
    return SDSGE_MCMC_EHESSIAN;
  }

  const size_t nv = (size_t)d;
  const size_t nm = nv * nv;

  f64 *x = work;
  f64 *h = x + nv;
  f64 *plus = h + nv;
  f64 *minus = plus + nv;
  f64 *H = minus + nv;
  f64 *L = H + nm;

  memcpy(x, theta, nv * sizeof(f64));
  const f64 lp0 = logpost(theta, obj_ctx);
  if (!isfinite(lp0)) {
    return SDSGE_MCMC_EHESSIAN;
  }

  for (i64 i = 0; i < d; ++i) {
    h[i] = fmax(fabs(theta[i]), fd_absolute_floor) * fd_relative_step;
    if (!isfinite(h[i]) || h[i] == 0.0) {
      return SDSGE_MCMC_EHESSIAN;
    }

    x[i] = theta[i] + h[i];
    plus[i] = logpost(x, obj_ctx);
    x[i] = theta[i] - h[i];
    minus[i] = logpost(x, obj_ctx);
    x[i] = theta[i];
    if (!isfinite(plus[i]) || !isfinite(minus[i])) {
      return SDSGE_MCMC_EHESSIAN;
    }
  }

  for (i64 i = 0; i < d; ++i) {
    H[i * d + i] = (2.0 * lp0 - plus[i] - minus[i]) / (h[i] * h[i]);
    for (i64 j = i + 1; j < d; ++j) {
      x[i] = theta[i] + h[i];
      x[j] = theta[j] + h[j];
      const f64 lp_pp = logpost(x, obj_ctx);
      x[i] = theta[i] - h[i];
      x[j] = theta[j] - h[j];
      const f64 lp_mm = logpost(x, obj_ctx);
      x[i] = theta[i];
      x[j] = theta[j];
      if (!isfinite(lp_pp) || !isfinite(lp_mm)) {
        return SDSGE_MCMC_EHESSIAN;
      }

      const f64 hij = (-lp_pp - lp_mm + plus[i] + minus[i] + plus[j] +
                       minus[j] - 2.0 * lp0) /
                      (2.0 * h[i] * h[j]);
      H[i * d + j] = hij;
      H[j * d + i] = hij;
    }
  }

  if (sdsge_chol(H, 0.0, L, d) != SDSGE_OK) {
    return SDSGE_MCMC_ENOTSPD;
  }

  for (i64 j = 0; j < d; ++j) {
    for (i64 i = 0; i < d; ++i) {
      x[i] = i == j ? 1.0 : 0.0;
    }
    sdsge_backward_subst_chol_t(L, x, x, d);
    for (i64 i = 0; i < d; ++i) {
      factor[i * d + j] = x[i];
    }
  }

  return SDSGE_MCMC_OK;
}

i64 sdsge_mcmc_run(sdsge_objective_fn logpost, void *obj_ctx, bitgen_t *bg,
                   const f64 *theta0, i64 d, const sdsge_mcmc_options *opt,
                   const sdsge_estimation_options *map_opt,
                   sdsge_mcmc_buffers *buf, sdsge_mcmc_result *out) {
  const i64 total_steps = opt->burn_in + opt->n_draws * opt->thin;
  out->total_steps = total_steps;
  out->n_accepted = 0;
  out->bk_violations = 0; /* filled by the caller off the objective ctx */
  out->status = SDSGE_MCMC_OK;
  out->message = "ok";

  /* One workspace allocation up front (never inside the loop), freed on return.
   * 6 vectors of d + 4 matrices of d*d. */
  const size_t nm = d * d;
  f64 *work = (f64 *)malloc((5 * d + 3 * nm) * sizeof(f64));
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
   */
  sdsge_optim_result map_out;
  memcpy(current, theta0, d * sizeof(f64));
  sdsge_run_estimation(obj_ctx, d, current, map_opt, &map_out);
  if (!map_out.success) {
    out->status = SDSGE_MCMC_EMAP;
    out->message = map_out.message != NULL ? map_out.message : "MAP failed";
    free(work);
    return SDSGE_MCMC_EMAP;
  }

  // The allocation past `L` is dead at hessian_proposal_factor, and it is the
  // exact size needed for the hessian. `prop` is therefore passed to the
  // hessian proposal factor as the work buffer, and `L` is the output.
  f64 *hessian_work = prop;
  const i64 hessian_status = sdsge_mcmc_hessian_proposal_factor(
      logpost, obj_ctx, current, d, opt->hessian_fd_step_scale,
      opt->hessian_fd_absolute_floor, L, hessian_work);
  if (hessian_status != SDSGE_MCMC_OK) {
    out->status = hessian_status;
    out->message = hessian_status == SDSGE_MCMC_EALLOC
                       ? "MCMC Hessian workspace allocation failed"
                   : hessian_status == SDSGE_MCMC_ENOTSPD
                       ? "MAP Hessian is not positive definite"
                       : "MCMC Hessian construction failed";
    free(work);
    return hessian_status;
  }
  sdsge_matmul_abt(L, L, C, d, d, d);

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
     * running mean accumulates, above it the covariance recursion runs and the
     * factor is refactored every step. */
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
      memcpy(&buf->kept[keep_i * d], current, d * sizeof(f64));
      buf->kept_lp[keep_i] = cur_lp;
      ++keep_i;
    }
  }

  out->n_accepted = accepted;
  free(work);
  return SDSGE_MCMC_OK;
}
