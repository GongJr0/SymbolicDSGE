#include "estimation.h"
#include "../core/klein_solve.h"
#include "../kalman/kalman.h"
#include "../optim/nelder_mead.h"
#include "../optim/optim.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Direct includes for the primitives used here (native-include hygiene). */
#include "../_common/sdsge_linalg.h" /* sdsge_chol, sdsge_backward_subst_chol_t,
                                        sdsge_matmul_abt */
#include "prior_program.h" /* sdsge_transform_inverse_and_logjac,
                              sdsge_corr_entries_from_unconstrained */

/* sdsge_classify outcomes. */
#define SDSGE_SOLVE_OK 0
#define SDSGE_SOLVE_BK                                                         \
  1 /* stab != 0 or QZ breakdown; caller counts a BK violation */
#define SDSGE_SOLVE_INFEASIBLE                                                 \
  2 /* the draw has no solution to filter; sentinel, not a BK count */

/* theta -> params fill. params is in calib_params order and is the residual
 * argument vector directly (no gather). Non-estimated slots never move across
 * evals, so they are seeded once here; the per-eval fill touches only the
 * estimated entries. */

void sdsge_init_params(f64 *SDSGE_RESTRICT params,
                       const f64 *SDSGE_RESTRICT base_params, i64 n_par) {
  for (i64 i = 0; i < n_par; ++i) {
    params[i] = base_params[i];
  }
}

static inline void sdsge_fill_params(sdsge_obj_common *base,
                                     const f64 *SDSGE_RESTRICT theta) {
  f64 x, logjac;
  for (i64 s = 0; s < base->pmap.n_scalars; ++s) {
    const sdsge_scalar_scatter *sc = &base->pmap.scalars[s];
    sdsge_transform_inverse_and_logjac(sc->transform_code,
                                       (f64 *)sc->transform_params,
                                       theta[sc->theta_idx], &x, &logjac);
    base->params[sc->param_slot] = x;
  }
}

/* Public wrapper: scatter a theta into base->params (e.g. resolve x_best after
 * the optimizer returns). params then holds the named parameter vector. */
void sdsge_scatter_params(sdsge_obj_common *SDSGE_RESTRICT base,
                          const f64 *SDSGE_RESTRICT theta) {
  sdsge_fill_params(base, theta);
}

static inline void sdsge_z0_from_x0(const f64 *SDSGE_RESTRICT x0,
                                    const i64 n_state, f64 *SDSGE_RESTRICT z0) {
  /* z0 = [x0; 0] */
  for (i64 i = 0; i < n_state; ++i) {
    z0[i] = x0[i];
    z0[n_state + i] = 0.0;
  }
}

/* corr(K*K) := I, then off-diagonal pairs corr[i,j]=corr[j,i]=params[slot]. */
static inline void sdsge_assemble_corr(const i64 *SDSGE_RESTRICT pair_i,
                                       const i64 *SDSGE_RESTRICT pair_j,
                                       const i64 *SDSGE_RESTRICT pair_slot,
                                       i64 n_pairs,
                                       const f64 *SDSGE_RESTRICT params, i64 K,
                                       f64 *SDSGE_RESTRICT corr) {
  for (i64 r = 0; r < K; ++r) {
    for (i64 c = 0; c < K; ++c) {
      corr[r * K + c] = (r == c) ? 1.0 : 0.0;
    }
  }
  for (i64 p = 0; p < n_pairs; ++p) {
    const i64 i = pair_i[p];
    const i64 j = pair_j[p];
    const f64 v = params[pair_slot[p]];
    corr[i * K + j] = v;
    corr[j * K + i] = v;
  }
}

/* out(K*K) := outer(std, std) * corr, with std[k] = params[std_slots[k]]. */
static inline void sdsge_cov_from_std_corr(const i64 *SDSGE_RESTRICT std_slots,
                                           const f64 *SDSGE_RESTRICT params,
                                           const f64 *SDSGE_RESTRICT corr,
                                           i64 K, f64 *SDSGE_RESTRICT out) {
  for (i64 i = 0; i < K; ++i) {
    const f64 si = params[std_slots[i]];
    for (i64 j = 0; j < K; ++j) {
      const f64 sj = params[std_slots[j]];
      out[i * K + j] = si * sj * corr[i * K + j];
    }
  }
}

/* Build one covariance (Q or R); returns the matrix the filter should read.
 * `out`/`corr_scratch` are K*K, `std_scratch` is K. */
static inline const f64 *sdsge_build_cov(const sdsge_cov_spec *spec,
                                         const f64 *SDSGE_RESTRICT theta,
                                         const f64 *SDSGE_RESTRICT params,
                                         f64 *SDSGE_RESTRICT std_scratch,
                                         f64 *SDSGE_RESTRICT corr_scratch,
                                         f64 *SDSGE_RESTRICT out) {
  if (spec->is_constant) {
    return spec->constant;
  }
  const i64 K = spec->K;
  if (spec->corr_from_block) {
    for (i64 k = 0; k < K; ++k) {
      std_scratch[k] = params[spec->std_slots[k]];
    }
    sdsge_cov_from_unconstrained(theta + spec->block_theta_off, std_scratch, K,
                                 corr_scratch, out);
  } else {
    sdsge_assemble_corr(spec->pair_i, spec->pair_j, spec->pair_slot,
                        spec->n_pairs, params, K, corr_scratch);
    sdsge_cov_from_std_corr(spec->std_slots, params, corr_scratch, K, out);
  }
  return out;
}

static inline klein_spec sdsge_spec_from(const sdsge_obj_common *b) {
  const klein_spec spec = {.residual = b->residual,
                           .zgges = b->zgges,
                           .dgeqrf = b->dgeqrf,
                           .dormqr = b->dormqr,
                           .ss_seed = b->ss_seed,
                           .params = b->params,
                           .incidence = b->incidence,
                           .n_var = b->dims.n_var,
                           .n_state = b->dims.n_state,
                           .n_ctrl = b->dims.n_ctrl,
                           .n_exog = b->dims.n_exog,
                           .n_par = b->dims.n_par};
  return spec;
}

/* Estimation's reading of a core solve verdict: every way the pencil half can
 * fail leaves f/p/stab unusable, and so does a nonzero stab, so all of them
 * reject the draw as a Blanchard-Kahn violation. A missing steady state, an
 * allocation failure and a second-order breakdown make the draw infeasible
 * rather than indeterminate, so none of them are counted as violations. */
static inline int sdsge_classify(const i64 rc, const i64 stab) {
  switch (rc) {
  case SDSGE_KLEIN_SOLVE_OK:
    return (stab != 0) ? SDSGE_SOLVE_BK : SDSGE_SOLVE_OK;
  case SDSGE_KLEIN_SOLVE_QZ:
  case SDSGE_KLEIN_SOLVE_SINGULAR:
  case SDSGE_KLEIN_SOLVE_NO_STATES:
    return SDSGE_SOLVE_BK;
  default:
    return SDSGE_SOLVE_INFEASIBLE;
  }
}

static inline int sdsge_solve1_run(sdsge_obj_common *b, sdsge_solve1 *s) {
  const klein_spec spec = sdsge_spec_from(b);
  const i64 rc = sdsge_klein_solve1(&spec, s, b->solve_arena, b->solve_iarena);
  return sdsge_classify(rc, s->stab);
}

static inline int sdsge_solve2_run(sdsge_obj_common *b, sdsge_solve1 *s,
                                   sdsge_solve2 *s2,
                                   const f64 *SDSGE_RESTRICT Q) {
  const sgu_klein_spec spec = {
      .first = sdsge_spec_from(b), .bc_residual = b->bc_residual, .Q = Q};
  const i64 rc =
      sdsge_sgu_klein_solve2(&spec, s, s2, b->solve_arena, b->solve_iarena);
  return sdsge_classify(rc, s->stab);
}

/* Fold the log-prior into a computed loglik. Non-finite loglik or logprior ->
 * the -inf sentinel; has_priors == 0 returns loglik as-is. */
static inline f64 sdsge_add_lp(const sdsge_obj_common *b,
                               const f64 *SDSGE_RESTRICT theta, f64 ll,
                               int has_priors) {
  if (!isfinite(ll)) {
    return -INFINITY;
  }
  if (!has_priors) {
    return ll;
  }
  const sdsge_prior_tables *pr = &b->prior;
  const f64 lp = sdsge_logprior_program(
      (f64 *)theta, (i64 *)pr->scalar_indices, (i64 *)pr->scalar_dist_codes,
      (i64 *)pr->scalar_transform_codes, (f64 *)pr->scalar_dist_params,
      (f64 *)pr->scalar_transform_params, pr->n_scalar,
      (i64 *)pr->matrix_offsets, (i64 *)pr->matrix_dims,
      (i64 *)pr->matrix_lengths, (f64 *)pr->matrix_etas,
      (f64 *)pr->matrix_log_constants, pr->n_blocks, pr->include_logjac);
  if (!isfinite(lp)) {
    return -INFINITY;
  }
  return ll + lp;
}

/* Public: the log-prior alone at a theta (e.g. x_best), from the packed tables.
 * No filter. 0 when the run carries no prior (MLE). */
f64 sdsge_logprior_at(const sdsge_obj_common *SDSGE_RESTRICT base,
                      const f64 *SDSGE_RESTRICT theta) {
  const sdsge_prior_tables *pr = &base->prior;
  if (!pr->has_prior) {
    return 0.0;
  }
  return sdsge_logprior_program(
      (f64 *)theta, (i64 *)pr->scalar_indices, (i64 *)pr->scalar_dist_codes,
      (i64 *)pr->scalar_transform_codes, (f64 *)pr->scalar_dist_params,
      (f64 *)pr->scalar_transform_params, pr->n_scalar,
      (i64 *)pr->matrix_offsets, (i64 *)pr->matrix_dims,
      (i64 *)pr->matrix_lengths, (f64 *)pr->matrix_etas,
      (f64 *)pr->matrix_log_constants, pr->n_blocks, pr->include_logjac);
}

/* Linear measurement (C, d) from the meas / jac cfuncs at the linearization
 * point. C is n_obs*n_var, d is n_obs. */
static inline void sdsge_build_measurement(sdsge_linear_ctx *ctx) {
  const sdsge_obj_common *b = &ctx->base;
  const sdsge_solve1 *s = &ctx->solve;

  b->meas(s->ss, b->params, ctx->d);
  b->jac(s->ss, b->params, ctx->C);
}

static inline i64 sdsge_resolve_stationary_p0(sdsge_obj_common *b,
                                              const f64 *SDSGE_RESTRICT A,
                                              const f64 *SDSGE_RESTRICT B,
                                              const f64 *SDSGE_RESTRICT Q,
                                              const i64 n, const i64 ld_out) {
  if (!b->derive_P0) {
    return KF_OK;
  }
  return kf_stationary_covariance(A, B, Q, 1e-12, 64, b->filter_arena, b->P0, n,
                                  b->dims.n_exog, ld_out);
}

f64 sdsge_obj_linear(sdsge_linear_ctx *ctx, const f64 *SDSGE_RESTRICT theta,
                     int has_priors) {
  sdsge_obj_common *b = &ctx->base;
  sdsge_solve1 *s = &ctx->solve;

  sdsge_fill_params(b, theta);
  const f64 *Q =
      sdsge_build_cov(&b->q_spec, theta, b->params, b->std_q, b->corr_q, b->Q);
  const f64 *R =
      sdsge_build_cov(&b->r_spec, theta, b->params, b->std_r, b->corr_r, b->R);

  const int solve_rc = sdsge_solve1_run(b, s);
  if (solve_rc == SDSGE_SOLVE_BK) {
    b->bk_violations++;
    return -INFINITY;
  }
  if (solve_rc != SDSGE_SOLVE_OK) {
    return -INFINITY;
  }
  sdsge_build_measurement(ctx);

  i64 p0_rc = sdsge_resolve_stationary_p0(b, s->A, s->B, Q, b->dims.n_var,
                                          b->dims.n_var);
  if (p0_rc != KF_OK) {
    return -INFINITY;
  }

  f64 ll = 0.0;
  kf_inputs in = {.n = b->dims.n_var,
                  .m = b->dims.n_obs,
                  .k = b->dims.n_exog,
                  .T = b->dims.T,
                  .A = s->A,
                  .B = s->B,
                  .C = ctx->C,
                  .d = ctx->d,
                  .Q = Q,
                  .R = R,
                  .y = b->y,
                  .x0 = b->x0,
                  .P0 = b->P0,
                  .symmetrize = b->symmetrize,
                  .joseph_cov = b->joseph_cov,
                  .jitter = b->jitter,
                  .return_shocks = 0,
                  .store_history = 0};
  kf_outputs out = {.loglik = &ll};
  if (kf_hot_loop(&in, b->filter_arena, &out) != KF_OK) {
    return -INFINITY;
  }
  return sdsge_add_lp(b, theta, ll, has_priors);
}

f64 sdsge_obj_extended(sdsge_extended_ctx *ctx, const f64 *SDSGE_RESTRICT theta,
                       int has_priors) {
  sdsge_obj_common *b = &ctx->base;
  sdsge_solve1 *s = &ctx->solve;

  sdsge_fill_params(b, theta);
  const f64 *Q =
      sdsge_build_cov(&b->q_spec, theta, b->params, b->std_q, b->corr_q, b->Q);
  const f64 *R =
      sdsge_build_cov(&b->r_spec, theta, b->params, b->std_r, b->corr_r, b->R);

  const int solve_rc = sdsge_solve1_run(b, s);
  if (solve_rc == SDSGE_SOLVE_BK) {
    b->bk_violations++;
    return -INFINITY;
  }
  if (solve_rc != SDSGE_SOLVE_OK) {
    return -INFINITY;
  }

  i64 p0_rc = sdsge_resolve_stationary_p0(b, s->A, s->B, Q, b->dims.n_var,
                                          b->dims.n_var);

  if (p0_rc != KF_OK) {
    return -INFINITY;
  }
  /* No precomputed (C, d): the EKF relinearizes each step via the meas / jac
   * cfuncs at the running state estimate. */
  f64 ll = 0.0;
  ekf_inputs in = {.meas = b->meas,
                   .jac = b->jac,
                   .A = s->A,
                   .B = s->B,
                   .calib_params = b->params,
                   .Q = Q,
                   .R = R,
                   .y = b->y,
                   .x0 = b->x0,
                   .P0 = b->P0,
                   .T = b->dims.T,
                   .n = b->dims.n_var,
                   .m = b->dims.n_obs,
                   .k = b->dims.n_exog,
                   .n_par = b->dims.n_par,
                   .jitter = b->jitter,
                   .symmetrize = b->symmetrize,
                   .joseph_cov = b->joseph_cov,
                   .compute_y_filt = 0,
                   .return_shocks = 0,
                   .store_history = 0};
  ekf_outputs out = {.loglik = &ll};
  if (ekf_hot_loop(&in, b->filter_arena, &out) != KF_OK) {
    return -INFINITY;
  }
  return sdsge_add_lp(b, theta, ll, has_priors);
}

f64 sdsge_obj_unscented(sdsge_unscented_ctx *ctx,
                        const f64 *SDSGE_RESTRICT theta, int has_priors) {
  sdsge_obj_common *b = &ctx->base;
  sdsge_solve1 *s = &ctx->solve;
  sdsge_solve2 *s2 = &ctx->solve2;

  sdsge_fill_params(b, theta);

  const f64 *Q =
      sdsge_build_cov(&b->q_spec, theta, b->params, b->std_q, b->corr_q, b->Q);
  const f64 *R =
      sdsge_build_cov(&b->r_spec, theta, b->params, b->std_r, b->corr_r, b->R);

  const int rc = sdsge_solve2_run(b, s, s2, Q);

  if (rc == SDSGE_SOLVE_BK) {
    b->bk_violations++;
    return -INFINITY;
  }
  if (rc != SDSGE_SOLVE_OK) {
    return -INFINITY;
  }
  i64 p0_rc = sdsge_resolve_stationary_p0(b, s->p, s->B, Q, b->dims.n_state,
                                          2 * b->dims.n_state);
  if (p0_rc != KF_OK) {
    return -INFINITY;
  }

  f64 ll = 0.0;
  ukf_inputs in = {.meas = b->meas,
                   .hx = s->p,
                   .gx = s->f,
                   .bu = s->B,
                   .hxx = s2->hxx,
                   .gxx = s2->gxx,
                   .hxu = s2->hxu,
                   .gxu = s2->gxu,
                   .huu = s2->huu,
                   .guu = s2->guu,
                   .hss = s2->hss,
                   .gss = s2->gss,
                   .steady_state = s->ss,
                   .params = b->params,
                   .Q = Q,
                   .R = R,
                   .obs = b->y,
                   .z0 = ctx->z0,
                   .P0 = b->P0,
                   .T = b->dims.T,
                   .n_state = b->dims.n_state,
                   .n_ctrl = b->dims.n_ctrl,
                   .n_exog = b->dims.n_exog,
                   .n_obs = b->dims.n_obs,
                   .n_params = b->dims.n_par,
                   .alpha = ctx->alpha,
                   .beta = ctx->beta,
                   .kappa = ctx->kappa,
                   .jitter = b->jitter,
                   .symmetrize = b->symmetrize,
                   .store_history = 0};

  ukf_outputs out = {.loglik = &ll};
  if (ukf_hot_loop(&in, b->filter_arena, &out) != KF_OK) {

    return -INFINITY;
  }
  return sdsge_add_lp(b, theta, ll, has_priors);
}

/* ---- Driver-facing closures (see estimation.h for the sign convention) ----
 */

f64 sdsge_min_linear_ll(const f64 *SDSGE_RESTRICT x, void *ctx) {
  return -sdsge_obj_linear((sdsge_linear_ctx *)ctx, x, 0);
}

f64 sdsge_min_linear_lp(const f64 *SDSGE_RESTRICT x, void *ctx) {
  return -sdsge_obj_linear((sdsge_linear_ctx *)ctx, x, 1);
}

f64 sdsge_min_extended_ll(const f64 *SDSGE_RESTRICT x, void *ctx) {
  return -sdsge_obj_extended((sdsge_extended_ctx *)ctx, x, 0);
}

f64 sdsge_min_extended_lp(const f64 *SDSGE_RESTRICT x, void *ctx) {
  return -sdsge_obj_extended((sdsge_extended_ctx *)ctx, x, 1);
}

f64 sdsge_min_unscented_ll(const f64 *SDSGE_RESTRICT x, void *ctx) {
  return -sdsge_obj_unscented((sdsge_unscented_ctx *)ctx, x, 0);
}

f64 sdsge_min_unscented_lp(const f64 *SDSGE_RESTRICT x, void *ctx) {
  return -sdsge_obj_unscented((sdsge_unscented_ctx *)ctx, x, 1);
}

f64 sdsge_post_linear(const f64 *SDSGE_RESTRICT x, void *ctx) {
  return sdsge_obj_linear((sdsge_linear_ctx *)ctx, x, 1);
}

f64 sdsge_post_extended(const f64 *SDSGE_RESTRICT x, void *ctx) {
  return sdsge_obj_extended((sdsge_extended_ctx *)ctx, x, 1);
}

f64 sdsge_post_unscented(const f64 *SDSGE_RESTRICT x, void *ctx) {
  return sdsge_obj_unscented((sdsge_unscented_ctx *)ctx, x, 1);
}

static const sdsge_objective_fn obj_table[2][3] = {
    {sdsge_min_linear_ll, sdsge_min_extended_ll, sdsge_min_unscented_ll},
    {sdsge_min_linear_lp, sdsge_min_extended_lp, sdsge_min_unscented_lp}};

/* The likelihood alone, unnegated. sdsge_post_* already covers the with-priors
 * row; these three complete the table the covariance needs, which is the +value
 * objective rather than the one the optimizer minimizes. */
static f64 sdsge_pos_linear_ll(const f64 *SDSGE_RESTRICT x, void *ctx) {
  return sdsge_obj_linear((sdsge_linear_ctx *)ctx, x, 0);
}

static f64 sdsge_pos_extended_ll(const f64 *SDSGE_RESTRICT x, void *ctx) {
  return sdsge_obj_extended((sdsge_extended_ctx *)ctx, x, 0);
}

static f64 sdsge_pos_unscented_ll(const f64 *SDSGE_RESTRICT x, void *ctx) {
  return sdsge_obj_unscented((sdsge_unscented_ctx *)ctx, x, 0);
}

static const sdsge_objective_fn pos_table[2][3] = {
    {sdsge_pos_linear_ll, sdsge_pos_extended_ll, sdsge_pos_unscented_ll},
    {sdsge_post_linear, sdsge_post_extended, sdsge_post_unscented}};

/* The tables as one lookup. `negate` picks the minimized form a driver wants
 * over the +value form a reported density wants. Callers outside this file go
 * through here rather than re-spelling which symbol belongs to which mode, and
 * it is what reaches the likelihood row, whose entries are file-static. */
sdsge_objective_fn sdsge_select_objective(int negate, int has_priors,
                                          int filter_mode) {
  return negate ? obj_table[has_priors][filter_mode]
                : pos_table[has_priors][filter_mode];
}

/* Covariance of an estimate at `theta`, in factored form. Separate from every
 * driver that wants it: the optimizer takes it as the asymptotic covariance of
 * the point it just found, and the sampler takes it as the proposal it starts
 * from, at a mode either found here or supplied. The objective returns
 * +logpost, so every finite-difference expression below is written directly
 * for H = -d^2 logpost.
 *
 * The off-diagonal stencil matches Dynare's hessian.m: it reuses the two
 * coordinate-direction evaluations and needs only the (++), (--) pair for
 * each i < j. After H = L L^T, solving L^T X = I gives X X^T = H^-1. */
i64 sdsge_estimation_cov_factor(sdsge_objective_fn logpost, void *obj_ctx,
                                const f64 *SDSGE_RESTRICT theta, i64 d,
                                f64 fd_step_scale, f64 fd_absolute_floor,
                                f64 *SDSGE_RESTRICT factor,
                                f64 *SDSGE_RESTRICT work) {
  if (d <= 0 || fd_step_scale <= 0.0 || fd_absolute_floor <= 0.0) {
    return SDSGE_ESTIMATION_EHESSIAN;
  }
  const f64 fd_relative_step = sqrt(cbrt(DBL_EPSILON)) * fd_step_scale;
  if (!isfinite(fd_relative_step) || fd_relative_step <= 0.0) {
    return SDSGE_ESTIMATION_EHESSIAN;
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
    return SDSGE_ESTIMATION_EHESSIAN;
  }

  for (i64 i = 0; i < d; ++i) {
    h[i] = fmax(fabs(theta[i]), fd_absolute_floor) * fd_relative_step;
    if (!isfinite(h[i]) || h[i] == 0.0) {
      return SDSGE_ESTIMATION_EHESSIAN;
    }

    x[i] = theta[i] + h[i];
    plus[i] = logpost(x, obj_ctx);
    x[i] = theta[i] - h[i];
    minus[i] = logpost(x, obj_ctx);
    x[i] = theta[i];
    if (!isfinite(plus[i]) || !isfinite(minus[i])) {
      return SDSGE_ESTIMATION_EHESSIAN;
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
        return SDSGE_ESTIMATION_EHESSIAN;
      }

      const f64 hij = (-lp_pp - lp_mm + plus[i] + minus[i] + plus[j] +
                       minus[j] - 2.0 * lp0) /
                      (2.0 * h[i] * h[j]);
      H[i * d + j] = hij;
      H[j * d + i] = hij;
    }
  }

  if (sdsge_chol(H, 0.0, L, d) != SDSGE_OK) {
    return SDSGE_ESTIMATION_ENOTSPD;
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

  return SDSGE_ESTIMATION_OK;
}

/* vcov = L L^T for the factor at `theta`. The buffer is filled with NaN first,
 * so every failure below is a report rather than a half-written answer. */
static void sdsge_fill_cov(void *ctx, i64 d, const f64 *SDSGE_RESTRICT theta,
                           const sdsge_estimation_options *opt,
                           sdsge_estimation_result *out) {
  const size_t nm = (size_t)d * (size_t)d;
  for (size_t i = 0; i < nm; ++i) {
    out->vcov[i] = NAN;
  }
  if (!out->base.success) {
    out->cov_status = SDSGE_ESTIMATION_EHESSIAN;
    return;
  }

  /* The factor, then sdsge_estimation_cov_factor's documented scratch. */
  f64 *scratch = (f64 *)malloc((3 * nm + 4 * (size_t)d) * sizeof(f64));
  if (scratch == NULL) {
    out->cov_status = SDSGE_ESTIMATION_EALLOC;
    return;
  }
  f64 *factor = scratch;
  f64 *work = scratch + nm;

  const i64 status = sdsge_estimation_cov_factor(
      pos_table[opt->has_priors][opt->filter_mode], ctx, theta, d,
      opt->cov_fd_step_scale, opt->cov_fd_absolute_floor, factor, work);
  if (status != SDSGE_ESTIMATION_OK) {
    out->cov_status = status;
    free(scratch);
    return;
  }

  sdsge_matmul_abt(factor, factor, out->vcov, d, d, d);
  out->cov_status = SDSGE_ESTIMATION_OK;
  free(scratch);
}

/* Standard errors in the caller's parameter space, from the theta-space
 * covariance. The jacobian of theta -> parameters is block diagonal: a scalar
 * depends on its own theta entry alone and a CPC block on its own run, so no
 * d*d jacobian is ever formed and V's cross terms never reach a diagonal
 * entry. A scalar's row is |dx/dz|, which the transform's own log-jacobian
 * already carries, exact and free; only a block is differenced, over the
 * entries kernel. Filled with NaN first, and a covariance that failed is NaN
 * already, so both it and a negative variance report themselves in place with
 * no status to consult. */
static void sdsge_fill_se(const sdsge_obj_common *SDSGE_RESTRICT b, i64 d,
                          const f64 *SDSGE_RESTRICT theta,
                          const f64 *SDSGE_RESTRICT vcov,
                          f64 *SDSGE_RESTRICT out_se) {
  for (i64 i = 0; i < d; ++i) {
    out_se[i] = NAN;
  }

  f64 x, logjac;
  for (i64 s = 0; s < b->pmap.n_scalars; ++s) {
    const sdsge_scalar_scatter *sc = &b->pmap.scalars[s];
    const i64 idx = sc->theta_idx;
    sdsge_transform_inverse_and_logjac(sc->transform_code, sc->transform_params,
                                       theta[idx], &x, &logjac);
    const f64 v = vcov[idx * d + idx];
    if (v >= 0.0) {
      out_se[idx] = exp(logjac) * sqrt(v);
    }
  }

  const sdsge_cov_spec *specs[2] = {&b->q_spec, &b->r_spec};
  i64 lmax = 0;
  i64 kmax = 0;
  for (int sp = 0; sp < 2; ++sp) {
    if (!sdsge_spec_has_block(specs[sp])) {
      continue;
    }
    lmax = max_i64(lmax, specs[sp]->block_theta_len);
    kmax = max_i64(kmax, specs[sp]->K);
  }
  if (lmax == 0) {
    return;
  }

  const size_t need =
      (size_t)(2 * lmax * lmax + 3 * lmax + kmax * kmax) * sizeof(f64);
  f64 *scratch = (f64 *)malloc(need);
  if (scratch == NULL) {
    return;
  }
  f64 *jac = scratch;                  /* L*L: d(corr entries) / dz */
  f64 *jv = jac + lmax * lmax;         /* L*L: jac * V_block */
  f64 *probe = jv + lmax * lmax;       /* L: the perturbed z */
  f64 *plus = probe + lmax;            /* L */
  f64 *minus = plus + lmax;            /* L */
  f64 *chol = minus + lmax;            /* K*K: the entries kernel's factor */

  /* Central difference of a closed-form algebraic map, so the step is the
   * first-derivative optimum and not opt->cov_fd_step_scale, which is tuned
   * for a second derivative of the filter. */
  const f64 step = cbrt(DBL_EPSILON);

  for (int sp = 0; sp < 2; ++sp) {
    const sdsge_cov_spec *spec = specs[sp];
    if (!sdsge_spec_has_block(spec)) {
      continue;
    }
    const i64 off = spec->block_theta_off;
    const i64 len = spec->block_theta_len;
    const f64 *z = theta + off;
    f64 unused_logjac;

    for (i64 j = 0; j < len; ++j) {
      const f64 h = step * max_f64(fabs(z[j]), 1.0);
      for (i64 c = 0; c < len; ++c) {
        probe[c] = z[c];
      }
      probe[j] = z[j] + h;
      sdsge_corr_entries_from_unconstrained(probe, spec->K, chol, plus,
                                            &unused_logjac);
      probe[j] = z[j] - h;
      sdsge_corr_entries_from_unconstrained(probe, spec->K, chol, minus,
                                            &unused_logjac);
      const f64 scale = 0.5 / h;
      for (i64 i = 0; i < len; ++i) {
        jac[i * len + j] = (plus[i] - minus[i]) * scale;
      }
    }

    for (i64 i = 0; i < len; ++i) {
      for (i64 j = 0; j < len; ++j) {
        f64 acc = 0.0;
        for (i64 k = 0; k < len; ++k) {
          acc += jac[i * len + k] * vcov[(off + k) * d + off + j];
        }
        jv[i * len + j] = acc;
      }
    }
    for (i64 i = 0; i < len; ++i) {
      f64 var = 0.0;
      for (i64 j = 0; j < len; ++j) {
        var += jv[i * len + j] * jac[i * len + j];
      }
      if (var >= 0.0) {
        out_se[off + i] = sqrt(var);
      }
    }
  }

  free(scratch);
}

void sdsge_run_estimation(void *ctx, i64 n_theta, f64 *SDSGE_RESTRICT theta,
                          const sdsge_estimation_options *opt,
                          sdsge_estimation_result *out) {

  const sdsge_objective_fn obj = obj_table[opt->has_priors][opt->filter_mode];
  switch (opt->method) {
  case ESTIMATION_LBFGSB:
    sdsge_lbfgsb(obj, ctx, n_theta, theta, opt->lo, opt->hi, opt->nbd,
                 &opt->optim, &out->base);
    break;
  case ESTIMATION_NELDER_MEAD:
    sdsge_neldermead(obj, ctx, n_theta, theta, opt->lo, opt->hi, opt->nbd,
                     &opt->optim, &out->base);
    break;
  default:
    out->base.status = SDSGE_OPTIM_EINVAL;
    out->base.success = 0;
    out->base.message = "ERROR: unknown estimation method";
    out->base.nfev = 0;
    out->base.nit = 0;
    out->base.fun = NAN;
  }

  out->cov_status = SDSGE_ESTIMATION_OK;
  if (opt->compute_cov && out->vcov != NULL) {
    sdsge_fill_cov(ctx, n_theta, theta, opt, out);
    if (out->se != NULL) {
      sdsge_fill_se((const sdsge_obj_common *)ctx, n_theta, theta, out->vcov,
                    out->se);
    }
  }

  /* Last, not before the covariance: its probes scatter perturbed params, and
   * callers read the ctx expecting it to sit at the returned theta. */
  sdsge_scatter_params((sdsge_obj_common *)ctx, theta);
}
