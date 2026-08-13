#include "estimation.h"
#include "../core/klein_solve.h"

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
                                   sdsge_solve2 *s2) {
  const sgu_klein_spec spec = {.first = sdsge_spec_from(b),
                               .bc_residual = b->bc_residual};
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
      (f64 *)pr->matrix_log_constants, pr->n_blocks);
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
      (f64 *)pr->matrix_log_constants, pr->n_blocks);
}

/* Linear measurement (C, d) from the meas / jac cfuncs at the linearization
 * point. C is n_obs*n_var, d is n_obs. */
static inline void sdsge_build_measurement(sdsge_linear_ctx *ctx) {
  const sdsge_obj_common *b = &ctx->base;
  const sdsge_solve1 *s = &ctx->solve;

  b->meas(s->ss, b->params, ctx->d);
  b->jac(s->ss, b->params, ctx->C);
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

  /* eta is chol(Q) in the leading n_exog rows. A constant Q is factored once at
   * compose time, so only a theta-driven Q refactors here. */
  if (!b->q_spec.is_constant) {
    if (sdsge_chol(Q, 0.0, s2->eta, b->q_spec.K) != SDSGE_OK) {
      return -INFINITY;
    }
  }

  const int rc = sdsge_solve2_run(b, s, s2);

  if (rc == SDSGE_SOLVE_BK) {
    b->bk_violations++;
    return -INFINITY;
  }
  if (rc != SDSGE_SOLVE_OK) {
    return -INFINITY;
  }

  f64 ll = 0.0;
  ukf_inputs in = {.meas = b->meas,
                   .hx = s->p,
                   .gx = s->f,
                   .bx = s2->bx,
                   .hxx = s2->hxx,
                   .gxx = s2->gxx,
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
