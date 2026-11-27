#include "unscented.h"
#include "../_common/sdsge_perturbation.h"
#include "kalman.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Per period, conditioned on Y_t, with a = [z_{t-1}; eps_t]:
 *   C  = Cov(a_t, z_t)                        (T, na, nz)
 *   Wa = Cov(z_{t-1}, y_t) L^{-T}             (T, nz, n_obs)
 *   Cv = Cov(v_t, z_t)                        (T, nv, nz) */
typedef struct {
  f64 *C;
  f64 *Wa;
  f64 *Cv;
} ukf_cache;

static arena_size ukf_cache_size(const i64 n_state, const i64 n_ctrl,
                                 const i64 n_exog, const i64 n_obs,
                                 const i64 T) {
  const i64 nz = 2 * n_state;
  const i64 na = nz + n_exog;
  const i64 nv = n_state + n_ctrl;
  return make_sizer(T * (na * nz + nz * n_obs + nv * nz), 0);
}

/* out(n,m) -= A(n,p) @ B(m,p)^T */
static void ukf_sub_abt(const f64 *SDSGE_RESTRICT A,
                        const f64 *SDSGE_RESTRICT B, f64 *SDSGE_RESTRICT out,
                        i64 n, i64 p, i64 m) {
  for (i64 i = 0; i < n; ++i) {
    const f64 *Ai = A + i * p;
    for (i64 j = 0; j < m; ++j) {
      const f64 *Bj = B + j * p;
      f64 s = 0.0;
      for (i64 k = 0; k < p; ++k)
        s += Ai[k] * Bj[k];
      out[i * m + j] -= s;
    }
  }
}

static void ukf_build_sigma_points(const f64 *SDSGE_RESTRICT mean,
                                   const f64 *SDSGE_RESTRICT chol, f64 gamma,
                                   f64 *SDSGE_RESTRICT sigma, i64 n) {
  memcpy(sigma, mean, (size_t)n * sizeof(f64));
  for (i64 col = 0; col < n; ++col) {
    f64 *plus = sigma + (1 + col) * n;
    f64 *minus = sigma + (1 + n + col) * n;
    for (i64 row = 0; row < n; ++row) {
      f64 delta = gamma * chol[row * n + col];
      plus[row] = mean[row] + delta;
      minus[row] = mean[row] - delta;
    }
  }
}

static void ukf_weighted_mean(const f64 *SDSGE_RESTRICT sigma, f64 w0, f64 wi,
                              i64 n_sig, i64 n, f64 *SDSGE_RESTRICT out) {
  for (i64 j = 0; j < n; ++j)
    out[j] = w0 * sigma[j];
  for (i64 r = 1; r < n_sig; ++r) {
    const f64 *row = sigma + r * n;
    for (i64 j = 0; j < n; ++j)
      out[j] += wi * row[j];
  }
}

static void ukf_weighted_cov(const f64 *SDSGE_RESTRICT sigma,
                             const f64 *SDSGE_RESTRICT mean, f64 w0, f64 wi,
                             i64 n_sig, i64 n, f64 *SDSGE_RESTRICT out) {
  sdsge_zero_mat(out, n, n);
  for (i64 r = 0; r < n_sig; ++r) {
    const f64 *row = sigma + r * n;
    const f64 w = (r == 0) ? w0 : wi;
    for (i64 i = 0; i < n; ++i) {
      const f64 di = row[i] - mean[i];
      for (i64 j = 0; j < n; ++j)
        out[i * n + j] += w * di * (row[j] - mean[j]);
    }
  }
}

static void ukf_project_vars(const ukf_inputs *in, const f64 *SDSGE_RESTRICT z,
                             f64 *SDSGE_RESTRICT vars) {
  const i64 ns = in->n_state;
  const i64 nc = in->n_ctrl;
  const f64 *x1 = z;
  const f64 *x2 = z + ns;

  for (i64 i = 0; i < ns; ++i)
    vars[i] = in->steady_state[i] + x1[i] + x2[i];

  for (i64 i = 0; i < nc; ++i) {
    f64 s = 0.5 * in->gss[i];
    const f64 *gx_i = in->gx + i * ns;
    for (i64 j = 0; j < ns; ++j)
      s += gx_i[j] * (x1[j] + x2[j]);
    const f64 *gxx_i = in->gxx + i * ns * ns;
    for (i64 j = 0; j < ns; ++j)
      for (i64 k = 0; k < ns; ++k)
        s += 0.5 * gxx_i[j * ns + k] * x1[j] * x1[k];
    vars[ns + i] = in->steady_state[ns + i] + s;
  }
}

static void ukf_weighted_cross(const f64 *SDSGE_RESTRICT sigma_a,
                               const f64 *SDSGE_RESTRICT a_mean,
                               const f64 *SDSGE_RESTRICT sigma_y,
                               const f64 *SDSGE_RESTRICT y_mean, f64 w0, f64 wi,
                               i64 n_sig, i64 na, i64 no,
                               f64 *SDSGE_RESTRICT out) {
  sdsge_zero_mat(out, na, no);
  for (i64 r = 0; r < n_sig; ++r) {
    const f64 *ar = sigma_a + r * na;
    const f64 *yr = sigma_y + r * no;
    const f64 w = (r == 0) ? w0 : wi;
    for (i64 i = 0; i < na; ++i) {
      const f64 dai = ar[i] - a_mean[i];
      for (i64 j = 0; j < no; ++j)
        out[i * no + j] += w * dai * (yr[j] - y_mean[j]);
    }
  }
}

static void ukf_weighted_meas_cov_cross(const f64 *SDSGE_RESTRICT sigma_z,
                                        const f64 *SDSGE_RESTRICT z_mean,
                                        const f64 *SDSGE_RESTRICT sigma_y,
                                        const f64 *SDSGE_RESTRICT y_mean,
                                        f64 w0, f64 wi, i64 n_sig, i64 nz,
                                        i64 no, f64 *SDSGE_RESTRICT S,
                                        f64 *SDSGE_RESTRICT Pzy) {
  sdsge_zero_mat(S, no, no);
  sdsge_zero_mat(Pzy, nz, no);

  for (i64 r = 0; r < n_sig; ++r) {
    const f64 *zr = sigma_z + r * nz;
    const f64 *yr = sigma_y + r * no;
    const f64 w = (r == 0) ? w0 : wi;
    for (i64 i = 0; i < no; ++i) {
      const f64 dyi = yr[i] - y_mean[i];
      for (i64 j = 0; j < no; ++j)
        S[i * no + j] += w * dyi * (yr[j] - y_mean[j]);
    }
    for (i64 i = 0; i < nz; ++i) {
      const f64 dzi = zr[i] - z_mean[i];
      for (i64 j = 0; j < no; ++j)
        Pzy[i * no + j] += w * dzi * (yr[j] - y_mean[j]);
    }
  }
}

static void ukf_store_history(const ukf_inputs *in, const f64 *SDSGE_RESTRICT z,
                              const f64 *SDSGE_RESTRICT vars,
                              f64 *SDSGE_RESTRICT x1, f64 *SDSGE_RESTRICT x2,
                              f64 *SDSGE_RESTRICT x, i64 t) {
  const i64 ns = in->n_state;
  const i64 nv = ns + in->n_ctrl;

  memcpy(x1 + t * ns, z, (size_t)ns * sizeof(f64));
  memcpy(x2 + t * ns, z + ns, (size_t)ns * sizeof(f64));
  memcpy(x + t * nv, vars, (size_t)nv * sizeof(f64));
}

/* Sigma-point Cholesky with an on-failure floor. Factor at the caller's jitter
 * first, so a well-conditioned covariance is unperturbed (parity with the plain
 * path); only when that fails add a scale-relative floor to lift a
 * rank-deficient covariance (e.g. a zero-risk-correction second order, whose
 * augmented block is degenerate) above the pivot threshold. A genuinely
 * unfilterable matrix still returns SDSGE_NOT_PD. */
#define UKF_CHOL_FLOOR_REL 1e-10
static int ukf_chol_auto(const f64 *SDSGE_RESTRICT P, f64 jitter,
                         f64 *SDSGE_RESTRICT L, i64 n) {
  if (sdsge_chol(P, jitter, L, n) == SDSGE_OK)
    return SDSGE_OK;
  f64 scale = 0.0;
  for (i64 i = 0; i < n; ++i)
    if (P[i * n + i] > scale)
      scale = P[i * n + i];
  return sdsge_chol(P, jitter + scale * UKF_CHOL_FLOOR_REL, L, n);
}

arena_size ukf_arena_size(const i64 n_state, const i64 n_ctrl, const i64 n_exog,
                          const i64 n_obs) {
  const i64 nz = 2 * n_state;
  const i64 na = nz + n_exog;
  const i64 n_sig = 2 * na + 1;
  const i64 nv = n_state + n_ctrl;

  return make_sizer(3 * nz + 4 * nz * nz + na * na + n_exog * n_exog +
                        n_sig * na + n_sig * nz + n_sig * n_obs + 4 * n_obs +
                        2 * n_obs * n_obs + nz * n_obs + n_sig * nv + 2 * nv +
                        nv * n_obs + na +
                        sdsge_second_order_step_scratch(n_state, n_exog),
                    0);
}
static int ukf_hot_loop(const ukf_inputs *in, f64 *SDSGE_RESTRICT arena,
                        ukf_outputs *out, ukf_cache *cache) {
  const i64 ns = in->n_state;
  const i64 nc = in->n_ctrl;
  const i64 ne = in->n_exog;
  const i64 no = in->n_obs;
  const i64 T = in->T;
  const i64 nz = 2 * ns;
  /* The innovation is a sigma-point coordinate, not an additive covariance:
   * ghxu and ghuu need it pointwise, and a control responds to it within the
   * period. Dynare augments the same way (gaussian_filter_bank.m). */
  const i64 na = nz + ne;
  const i64 n_sig = 2 * na + 1;
  const i64 nv = ns + nc;

  if (in->meas == NULL || ns <= 0 || no <= 0 || nz <= 0)
    return KF_ERR_SHAPE_MISMATCH;

  const f64 lambda = in->alpha * in->alpha * ((f64)na + in->kappa) - (f64)na;
  const f64 scale = (f64)na + lambda;
  if (!(scale > 0.0) || !isfinite(scale))
    return KF_ERR_BAD_HYPERPARAMS;
  const f64 gamma = sqrt(scale);
  const f64 w0_m = lambda / scale;
  const f64 w0_c = w0_m + (1.0 - in->alpha * in->alpha + in->beta);
  const f64 wi = 1.0 / (2.0 * scale);

  f64 *z_prev = arena;
  f64 *z_pred = z_prev + nz;
  f64 *z_filt = z_pred + nz;
  f64 *P_prev = z_filt + nz;
  f64 *P_pred = P_prev + nz * nz;
  f64 *P_filt = P_pred + nz * nz;
  f64 *P_chol = P_filt + nz * nz;
  f64 *A_chol = P_chol + nz * nz;
  f64 *Q_chol = A_chol + na * na;
  f64 *sigma_a = Q_chol + ne * ne;
  f64 *sigma_z = sigma_a + n_sig * na;
  f64 *sigma_y = sigma_z + n_sig * nz;
  f64 *step = sigma_y + n_sig * no;
  f64 *y_pred = step + sdsge_second_order_step_scratch(ns, ne);
  f64 *innov = y_pred + no;
  f64 *std_innov = innov + no;
  f64 *S_inv_v = std_innov + no;
  f64 *S = S_inv_v + no;
  f64 *L = S + no * no;
  f64 *Pzy = L + no * no;
  f64 *sigma_v = Pzy + nz * no;
  f64 *vars_pred = sigma_v + n_sig * nv;
  f64 *vars_filt = vars_pred + nv;
  f64 *Pvy = vars_filt + nv;
  f64 *z_aug = Pvy + nv * no;
  /* Whitened in place once S is factored. */
  f64 *W = Pzy;

  memcpy(z_prev, in->z0, (size_t)nz * sizeof(f64));
  memcpy(P_prev, in->P0, (size_t)(nz * nz) * sizeof(f64));

  /* Q is fixed for the run. The augmented root is block diagonal and only its
   * state block moves, so the innovation block is written here and never again.
   */
  if (ne > 0 && ukf_chol_auto(in->Q, in->jitter, Q_chol, ne) != SDSGE_OK)
    return KF_ERR_Q_FACTORIZATION;
  sdsge_zero_mat(A_chol, na, na);
  for (i64 i = 0; i < ne; ++i)
    for (i64 j = 0; j < ne; ++j)
      A_chol[(nz + i) * na + nz + j] = Q_chol[i * ne + j];

  const f64 const_term = (f64)no * log(TWO_PI);
  f64 loglik = 0.0;
  int status = KF_OK;

  for (i64 t = 0; t < T; ++t) {
    if (ukf_chol_auto(P_prev, in->jitter, P_chol, nz) != SDSGE_OK) {
      status = KF_ERR_P_FACTORIZATION;
      break;
    }
    for (i64 i = 0; i < nz; ++i)
      for (i64 j = 0; j < nz; ++j)
        A_chol[i * na + j] = P_chol[i * nz + j];

    for (i64 i = 0; i < nz; ++i)
      z_aug[i] = z_prev[i];
    for (i64 i = nz; i < na; ++i)
      z_aug[i] = 0.0;
    ukf_build_sigma_points(z_aug, A_chol, gamma, sigma_a, na);

    for (i64 r = 0; r < n_sig; ++r) {
      const f64 *SDSGE_RESTRICT a = sigma_a + r * na;
      f64 *SDSGE_RESTRICT zn = sigma_z + r * nz;
      sdsge_second_order_step(in->hx, in->gx, in->bu, in->hxx, in->gxx, in->hxu,
                              in->gxu, in->huu, in->guu, in->hss, in->gss, a,
                              a + ns, ne > 0 ? a + nz : NULL, in->steady_state,
                              zn, zn + ns, sigma_v + r * nv, step, ns, nc, ne);
      in->meas(sigma_v + r * nv, in->params, sigma_y + r * no);
    }

    ukf_weighted_mean(sigma_z, w0_m, wi, n_sig, nz, z_pred);
    ukf_weighted_cov(sigma_z, z_pred, w0_c, wi, n_sig, nz, P_pred);
    if (in->symmetrize)
      sdsge_sym_inplace(P_pred, nz);

    ukf_weighted_mean(sigma_y, w0_m, wi, n_sig, no, y_pred);
    ukf_weighted_meas_cov_cross(sigma_z, z_pred, sigma_y, y_pred, w0_c, wi,
                                n_sig, nz, no, S, Pzy);
    for (i64 i = 0; i < no * no; ++i)
      S[i] += in->R[i];
    if (in->symmetrize)
      sdsge_sym_inplace(S, no);

    kf_row_minus_vec(in->obs, t, y_pred, innov, no);
    if (ukf_chol_auto(S, in->jitter, L, no) != SDSGE_OK) {
      status = KF_ERR_S_FACTORIZATION;
      break;
    }
    sdsge_forward_subst(L, innov, std_innov, no);

    kf_whiten_pc_t(L, Pzy, nz, no);
    kf_state_update(z_pred, W, std_innov, z_filt, nz, no);
    kf_cov_downdate(P_pred, W, P_filt, nz, no);
    if (in->symmetrize)
      sdsge_sym_inplace(P_filt, nz);

    loglik += -0.5 * (const_term + sdsge_logdet_from_chol(L, no) +
                      sdsge_dot(std_innov, std_innov, no));

    if (in->store_history) {
      ukf_weighted_mean(sigma_v, w0_m, wi, n_sig, nv, vars_pred);
      ukf_weighted_cross(sigma_v, vars_pred, sigma_y, y_pred, w0_c, wi, n_sig,
                         nv, no, Pvy);
      kf_whiten_pc_t(L, Pvy, nv, no);
      kf_state_update(vars_pred, Pvy, std_innov, vars_filt, nv, no);

      if (cache) {
        f64 *Cv = cache->Cv + t * nv * nz;
        ukf_weighted_cross(sigma_v, vars_pred, sigma_z, z_pred, w0_c, wi, n_sig,
                           nv, nz, Cv);
        ukf_sub_abt(Pvy, W, Cv, nv, no, nz);
      }

      /* Every cross with a is w_i * gamma * A_chol @ (x_{+j} - x_{-j}): the
       * augmented root is block diagonal and the paired weights are equal.
       * Plus points are rows 1..na, minus points rows na+1..2na. Shocks only
       * need the shock pairs; the cache needs all of them. */
      const i64 d0 = cache ? 0 : nz;
      f64 *dy = sigma_y + (1 + d0) * no;
      sdsge_vsub(dy, sigma_y + (1 + na + d0) * no, dy, (na - d0) * no);
      if (ne > 0) {
        sdsge_backward_subst_chol_t(L, std_innov, S_inv_v, no);
        sdsge_matvec(sigma_y + (1 + nz) * no, S_inv_v, out->eps_hat + t * ne,
                     ne, no);
        sdsge_trmv_lower_inplace(Q_chol, out->eps_hat + t * ne, wi * gamma, ne);
      }

      if (cache) {
        f64 *dz = sigma_z + nz;
        sdsge_vsub(dz, sigma_z + (1 + na) * nz, dz, na * nz);
        f64 *dyw = sigma_y + no;
        kf_whiten_pc_t(L, dyw, na, no);

        f64 *C = cache->C + t * na * nz;
        memcpy(C, dz, (size_t)(na * nz) * sizeof(f64));
        ukf_sub_abt(dyw, W, C, na, no, nz);
        sdsge_trmm_lower_inplace(P_chol, C, wi * gamma, nz, nz);
        if (ne > 0)
          sdsge_trmm_lower_inplace(Q_chol, C + nz * nz, wi * gamma, ne, nz);

        f64 *Wa = cache->Wa + t * nz * no;
        memcpy(Wa, dyw, (size_t)(nz * no) * sizeof(f64));
        sdsge_trmm_lower_inplace(P_chol, Wa, wi * gamma, nz, no);
      }

      ukf_store_history(in, z_pred, vars_pred, out->x1_pred, out->x2_pred,
                        out->x_pred, t);
      ukf_store_history(in, z_filt, vars_filt, out->x1_filt, out->x2_filt,
                        out->x_filt, t);
      memcpy(out->P_pred + t * nz * nz, P_pred,
             (size_t)(nz * nz) * sizeof(f64));
      memcpy(out->P_filt + t * nz * nz, P_filt,
             (size_t)(nz * nz) * sizeof(f64));
      memcpy(out->y_pred + t * no, y_pred, (size_t)no * sizeof(f64));
      in->meas(out->x_filt + t * nv, in->params, out->y_filt + t * no);
      memcpy(out->innov + t * no, innov, (size_t)no * sizeof(f64));
      memcpy(out->std_innov + t * no, std_innov, (size_t)no * sizeof(f64));
      memcpy(out->S + t * no * no, S, (size_t)(no * no) * sizeof(f64));
    }

    f64 *z_tmp = z_prev;
    z_prev = z_filt;
    z_filt = z_tmp;
    f64 *P_tmp = P_prev;
    P_prev = P_filt;
    P_filt = P_tmp;
  }

  *out->loglik = loglik;
  return status;
}

int ukf_forward(const ukf_inputs *in, f64 *SDSGE_RESTRICT arena,
                ukf_outputs *out) {
  return ukf_hot_loop(in, arena, out, NULL);
}

static arena_size ukf_smooth_loop_arena(const i64 n_state, const i64 n_ctrl,
                                        const i64 n_exog, const i64 n_obs) {
  const i64 nz = 2 * n_state;
  const arena_size fwd_arena = ukf_arena_size(n_state, n_ctrl, n_exog, n_obs);
  const arena_size smooth_arena = make_sizer(3 * nz + 5 * nz * nz, 0);
  return sdsge_max_arena(fwd_arena, smooth_arena);
}
arena_size ukf_smooth_arena_size(const i64 n_state, const i64 n_ctrl,
                                 const i64 n_exog, const i64 n_obs,
                                 const i64 T) {
  return add_arena(ukf_smooth_loop_arena(n_state, n_ctrl, n_exog, n_obs),
                   ukf_cache_size(n_state, n_ctrl, n_exog, n_obs, T));
}

int ukf_smooth(const ukf_inputs *in, f64 *SDSGE_RESTRICT arena,
               ukf_smooth_outputs *out) {

  int rc;
  ukf_cache cache;
  ukf_outputs *fwd = out->fwd_out;

  const i64 ns = in->n_state;
  const i64 ne = in->n_exog;
  const i64 no = in->n_obs;
  const i64 T = in->T;
  const i64 nz = 2 * ns;
  const i64 na = nz + ne;
  const i64 nv = ns + in->n_ctrl;

  const i64 akf = ukf_smooth_loop_arena(ns, in->n_ctrl, ne, no).n_float;
  cache.C = arena + akf;
  cache.Wa = cache.C + T * na * nz;
  cache.Cv = cache.Wa + T * nz * no;

  rc = ukf_hot_loop(in, arena, fwd, &cache);
  if (rc != KF_OK)
    return rc;

  f64 *zlag = arena;
  f64 *dz = zlag + nz;
  f64 *q = dz + nz;
  f64 *Plag = q + nz;
  f64 *dP = Plag + nz * nz;
  f64 *Lp = dP + nz * nz;
  f64 *G = Lp + nz * nz;
  f64 *tmp = G + nz * nz;

  memcpy(out->x1_smooth + (T - 1) * ns, fwd->x1_filt + (T - 1) * ns,
         (size_t)ns * sizeof(f64));
  memcpy(out->x2_smooth + (T - 1) * ns, fwd->x2_filt + (T - 1) * ns,
         (size_t)ns * sizeof(f64));
  memcpy(out->P_smooth + (T - 1) * nz * nz, fwd->P_filt + (T - 1) * nz * nz,
         (size_t)(nz * nz) * sizeof(f64));

  for (i64 t = T - 1; t >= 0; --t) {
    f64 *Ct = cache.C + t * na * nz;
    const f64 *Pf = fwd->P_filt + t * nz * nz;

    sdsge_vsub(out->x1_smooth + t * ns, fwd->x1_filt + t * ns, dz, ns);
    sdsge_vsub(out->x2_smooth + t * ns, fwd->x2_filt + t * ns, dz + ns, ns);
    sdsge_vsub(out->P_smooth + t * nz * nz, Pf, dP, nz * nz);

    if (ukf_chol_auto(Pf, in->jitter, Lp, nz) != SDSGE_OK) {
      return KF_ERR_P_FACTORIZATION;
    }

    sdsge_forward_subst(Lp, dz, q, nz);
    sdsge_backward_subst_chol_t(Lp, q, q, nz);

    if (ne > 0) {
      sdsge_matvec_plus_vec(Ct + nz * nz, q, fwd->eps_hat + t * ne,
                            out->eps_smooth + t * ne, ne, nz);
    }
    sdsge_matvec_plus_vec(cache.Cv + t * nv * nz, q, fwd->x_filt + t * nv,
                          out->x_smooth + t * nv, nv, nz);

    if (t == 0) {
      break;
    }

    /* Condition z_{t-1} on y_t, then on the smoothed z_t. */
    const f64 *Wa = cache.Wa + t * nz * no;
    const f64 *u = fwd->std_innov + t * no;
    kf_state_update(fwd->x1_filt + (t - 1) * ns, Wa, u, zlag, ns, no);
    kf_state_update(fwd->x2_filt + (t - 1) * ns, Wa + ns * no, u, zlag + ns, ns,
                    no);
    kf_cov_downdate(fwd->P_filt + (t - 1) * nz * nz, Wa, Plag, nz, no);

    sdsge_matvec_plus_vec(Ct, q, zlag, out->x1_smooth + (t - 1) * ns, ns, nz);
    sdsge_matvec_plus_vec(Ct + ns * nz, q, zlag + ns,
                          out->x2_smooth + (t - 1) * ns, ns, nz);

    /* Each period's cache entry is read once, so the gain whitens it in
     * place. */
    kf_whiten_pc_t(Lp, Ct, nz, nz);
    kf_gain_from_whitened(Lp, Ct, G, nz, nz);
    f64 *Ps_prev = out->P_smooth + (t - 1) * nz * nz;
    sdsge_matmul(G, dP, tmp, nz, nz, nz);
    sdsge_matmul_abt_plus_c(tmp, G, Plag, Ps_prev, nz, nz, nz);

    if (in->symmetrize) {
      sdsge_sym_inplace(Ps_prev, nz);
    }
  }

  return KF_OK;
}
