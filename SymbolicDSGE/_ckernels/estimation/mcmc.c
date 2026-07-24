#include "mcmc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Direct includes for the primitives used here (native-include hygiene). */
#include "../_common/sdsge_linalg.h" /* sdsge_chol, sdsge_matvec, sdsge_zero_mat */
#include "../rng/rng.h"              /* sdsge_rng_standard_{normal,uniform}_fill */

/* Native adaptive random-walk Metropolis mainloop (issue #331).
 *
 * A clean transcription of the numpy-era Estimator.mcmc loop, taken fully native
 * under the statistical-equivalence contract (option b): numpy's draws stay
 * bit-exact (the standard-normal / uniform fills advance numpy's own PCG64 via
 * `bg`), but the two pieces of deterministic linear algebra are ours, NOT numpy's:
 *
 *   - proposal:   prop = current + L @ z,  z ~ N(0, I), where L is OUR lower
 *                 Cholesky of the proposal covariance (not numpy's SVD-based
 *                 multivariate_normal map).
 *   - adaptation: a running (Welford) empirical covariance, refactored in place,
 *                 not a batch np.cov recomputed from stored history.
 *
 * Consequences of dropping stored history: memory is O(n_draws*d + d^2) and the
 * loop has no d==1 special case (a 1x1 covariance subsumes it). The running
 * moments are updated every step, but the proposal factor is rebuilt only every
 * adapt_interval steps within burn-in, on the same schedule as the numpy
 * reference. The resulting chain is a principled, reproducible native
 * stream, statistically equivalent to the numpy chain (matched stationary
 * marginals + acceptance), not a draw-for-draw reproduction of it.
 *
 * `logpost` returns +logpost (the estimation trampoline passes the objective
 * WITHOUT the optimizer's negation); a BK violation / non-finite eval surfaces as
 * -inf and auto-rejects through the finiteness gate. The BK-violation count lives
 * on the objective's own ctx (the objective owns that counter); the Cython caller
 * reads it off the ctx after the run, so `out->bk_violations` is left 0 here to
 * keep this loop a generic sampler over an opaque `obj_ctx`. */

/* L(d,d) := v * I (lower factor of the initial diagonal proposal covariance). */
static inline void sdsge_diag_fill(f64 *SDSGE_RESTRICT L, i64 d, f64 v) {
  sdsge_zero_mat(L, d, d);
  for (i64 i = 0; i < d; ++i) {
    L[i * d + i] = v;
  }
}

/* Welford update of the running mean / co-moment with a new point x(d). `count`
 * is the sample count AFTER including x (>= 1). M2 accumulates the outer product
 * of the pre- and post-update deltas; the empirical covariance is M2/(count-1). */
static inline void sdsge_welford_update(const f64 *SDSGE_RESTRICT x, i64 d,
                                        i64 count, f64 *SDSGE_RESTRICT mean,
                                        f64 *SDSGE_RESTRICT M2,
                                        f64 *SDSGE_RESTRICT delta,
                                        f64 *SDSGE_RESTRICT delta2) {
  const f64 inv = 1.0 / (f64)count;
  for (i64 i = 0; i < d; ++i) {
    delta[i] = x[i] - mean[i];
  }
  for (i64 i = 0; i < d; ++i) {
    mean[i] += delta[i] * inv;
  }
  for (i64 i = 0; i < d; ++i) {
    delta2[i] = x[i] - mean[i];
  }
  for (i64 i = 0; i < d; ++i) {
    for (i64 j = 0; j < d; ++j) {
      M2[i * d + j] += delta[i] * delta2[j];
    }
  }
}

/* Rebuild the proposal factor from the running co-moment:
 *   S = scale * (M2/(count-1) + eps * I),   L <- chol(S) if S is PD.
 * Mirrors the numpy adaptation `cov = scale * (emp + eps*I)`. A non-PD S (rare,
 * near-degenerate early cov) leaves the previous L untouched, so the sampler
 * keeps its last good proposal rather than breaking. */
static inline void sdsge_adapt_factor(const f64 *SDSGE_RESTRICT M2, i64 d,
                                      i64 count, f64 scale, f64 eps,
                                      f64 *SDSGE_RESTRICT S,
                                      f64 *SDSGE_RESTRICT Ltmp,
                                      f64 *SDSGE_RESTRICT L) {
  const f64 inv = 1.0 / (f64)(count - 1);
  for (i64 i = 0; i < d; ++i) {
    for (i64 j = 0; j < d; ++j) {
      f64 emp = M2[i * d + j] * inv;
      if (i == j) {
        emp += eps;
      }
      S[i * d + j] = scale * emp;
    }
  }
  if (sdsge_chol(S, 0.0, Ltmp, d) == SDSGE_OK) {
    memcpy(L, Ltmp, (size_t)(d * d) * sizeof(f64));
  }
}

i64 sdsge_mcmc_run(sdsge_objective_fn logpost, void *obj_ctx, bitgen_t *bg,
                   const f64 *theta0, i64 d, const sdsge_mcmc_options *opt,
                   sdsge_mcmc_buffers *buf, sdsge_mcmc_result *out) {
  const i64 total_steps = opt->burn_in + opt->n_draws * opt->thin;
  out->total_steps = total_steps;
  out->n_accepted = 0;
  out->bk_violations = 0; /* filled by the caller off the objective ctx */
  out->status = SDSGE_MCMC_OK;
  out->message = "ok";

  /* One workspace allocation up front (never inside the loop), freed on return.
   * 6 vectors of d + 4 matrices of d*d. */
  const size_t nv = (size_t)d;
  const size_t nm = (size_t)d * (size_t)d;
  f64 *work = (f64 *)malloc((6 * nv + 4 * nm) * sizeof(f64));
  if (work == NULL) {
    out->status = SDSGE_MCMC_EALLOC;
    out->message = "mcmc workspace allocation failed";
    return SDSGE_MCMC_EALLOC;
  }
  f64 *current = work;
  f64 *prop = current + nv;
  f64 *z = prop + nv;
  f64 *mean = z + nv;
  f64 *delta = mean + nv;
  f64 *delta2 = delta + nv;
  f64 *L = delta2 + nv;
  f64 *M2 = L + nm;
  f64 *S = M2 + nm;
  f64 *Ltmp = S + nm;

  const f64 scale = (2.38 * 2.38) / (f64)d;

  /* Initial proposal: cov0 = proposal_scale^2 * I  =>  L0 = proposal_scale * I. */
  sdsge_diag_fill(L, d, opt->proposal_scale);

  /* Running-moment accumulators start empty. */
  for (i64 i = 0; i < d; ++i) {
    mean[i] = 0.0;
  }
  memset(M2, 0, nm * sizeof(f64));
  i64 count = 0;

  memcpy(current, theta0, nv * sizeof(f64));
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
        memcpy(current, prop, nv * sizeof(f64));
        cur_lp = prop_lp;
        ++accepted;
      }
    }

    /* Haario adaptation over the running empirical covariance, during burn-in
     * only; the proposal is frozen once burn-in ends. Every accepted-or-repeated
     * state is folded into the running moments from t=0, so a recompute spans the
     * full history to that point (matching the numpy chain's np.cov over
     * history[:t+1]); the proposal factor is only rebuilt every adapt_interval
     * steps on the same schedule as the reference. */
    if (opt->adapt && t < opt->burn_in) {
      ++count;
      sdsge_welford_update(current, d, count, mean, M2, delta, delta2);
      if (t >= opt->adapt_start && (t + 1) % opt->adapt_interval == 0 &&
          count > 1) {
        sdsge_adapt_factor(M2, d, count, scale, opt->adapt_epsilon, S, Ltmp, L);
      }
    }

    /* Keep post-burn-in draws at the thinning cadence. */
    if (t >= opt->burn_in && (t - opt->burn_in) % opt->thin == 0) {
      memcpy(&buf->kept[keep_i * d], current, nv * sizeof(f64));
      buf->kept_lp[keep_i] = cur_lp;
      ++keep_i;
    }
  }

  out->n_accepted = accepted;
  free(work);
  return SDSGE_MCMC_OK;
}
