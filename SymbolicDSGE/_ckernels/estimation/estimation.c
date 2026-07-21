#include "estimation.h"
#include "../core/core.h"
#include "../core/klein_postproc.h"

/* theta -> params fill. Non-estimated slots (and their calib image) never move
 * across evals, so they are seeded once here; the per-eval fill touches only
 * the estimated entries. */

void sdsge_init_params(f64 *SDSGE_RESTRICT params,
                       const f64 *SDSGE_RESTRICT base_params, i64 n_params) {
  for (i64 i = 0; i < n_params; ++i) {
    params[i] = base_params[i];
  }
}

void sdsge_init_calib(f64 *SDSGE_RESTRICT calib_vec,
                      const f64 *SDSGE_RESTRICT params,
                      const i64 *SDSGE_RESTRICT calib_gather, i64 n_par) {
  for (i64 i = 0; i < n_par; ++i) {
    calib_vec[i] = params[calib_gather[i]];
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
  for (i64 u = 0; u < base->pmap.n_calib_upd; ++u) {
    const i64 pos = base->pmap.calib_upd[u];
    base->calib_vec[pos] = base->params[base->pmap.calib_gather[pos]];
  }
}

/* Real pencil (row-major) -> complex Schur input (column-major), widened. */
static inline void sdsge_to_complex_colmajor(const f64 *SDSGE_RESTRICT a,
                                             c128 *SDSGE_RESTRICT s,
                                             const i64 n) {
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < n; ++j) {
      s[j * n + i] = c128_from_real(a[i * n + j]);
    }
  }
}

/* Real part of a contiguous complex buffer. */
static inline void sdsge_real_part(const c128 *SDSGE_RESTRICT src,
                                   f64 *SDSGE_RESTRICT dst, const i64 len) {
  for (i64 k = 0; k < len; ++k) {
    dst[k] = c128_real(src[k]);
  }
}

/* In-place square transpose (column-major <-> row-major). */
static inline void sdsge_transpose_sq(c128 *SDSGE_RESTRICT m, const i64 n) {
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = i + 1; j < n; ++j) {
      const c128 tmp = m[i * n + j];
      m[i * n + j] = m[j * n + i];
      m[j * n + i] = tmp;
    }
  }
}

static inline void sdsge_bx_from_B(const f64 *SDSGE_RESTRICT B,
                                   const i64 n_state, const i64 n_exog,
                                   f64 *SDSGE_RESTRICT out) {
  /* out = B[:n_state, :] */
  for (i64 k = 0; k < n_state * n_exog; ++k) {
    out[k] = B[k];
  }
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

static inline int sdsge_solve1_run(sdsge_obj_common *b, sdsge_solve1 *s) {
  const i64 n = b->dims.n_var;
  klein_preproc(b->residual, b->steady_state, b->calib_vec, n, b->dims.n_par, n,
                b->log_linear, s->a_real, s->b_real);
  sdsge_to_complex_colmajor(s->a_real, s->s, n);
  sdsge_to_complex_colmajor(s->b_real, s->t, n);
  if (klein_qz(b->zgges, n, s->s, s->t, s->z) != KLEIN_QZ_OK) {
    return 1;
  }
  sdsge_transpose_sq(s->s, n);
  sdsge_transpose_sq(s->t, n);
  sdsge_transpose_sq(s->z, n);
  klein_postproc(s->s, s->t, s->z, b->dims.n_state, b->dims.n_ctrl, s->f, s->p,
                 &s->stab, s->eig);
  if (s->stab != 0) {
    return 1;
  }
  sdsge_assemble_state_space(s->p, s->f, b->dims.n_state, b->dims.n_ctrl,
                             b->dims.n_exog, s->A, s->B);
  return 0;
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

/* Linear measurement (C, d) from the meas / jac cfuncs at the linearization
 * point. C is n_obs*n_var, d is n_obs. */
static inline void sdsge_build_measurement(sdsge_linear_ctx *ctx) {
  const sdsge_obj_common *b = &ctx->base;
  b->meas(ctx->zero_state, b->calib_vec, ctx->d);
  b->jac(ctx->zero_state, b->calib_vec, ctx->C);
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

  if (sdsge_solve1_run(b, s)) {
    b->bk_violations++;
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
  if (kf_hot_loop(&in, &out) != KF_OK) {
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

  if (sdsge_solve1_run(b, s)) {
    b->bk_violations++;
    return -INFINITY;
  }

  /* No precomputed (C, d): the EKF relinearizes each step via the meas / jac
   * cfuncs at the running state estimate. */
  f64 ll = 0.0;
  ekf_inputs in = {.meas = b->meas,
                   .jac = b->jac,
                   .A = s->A,
                   .B = s->B,
                   .calib_params = b->calib_vec,
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
  if (ekf_hot_loop(&in, &out) != KF_OK) {
    return -INFINITY;
  }
  return sdsge_add_lp(b, theta, ll, has_priors);
}
