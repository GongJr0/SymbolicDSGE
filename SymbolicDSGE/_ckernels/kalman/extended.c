#include "extended.h"
#include "kalman.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  f64 *W; /* (T, n, m) */
  f64 *G; /* (T, m, n) */
} ekf_cache;

static arena_size ekf_cache_size(const i64 n, const i64 m, const i64 T) {
  return make_sizer(2 * T * n * m, 0);
}

arena_size ekf_arena_size(const i64 n, const i64 m, const i64 k) {
  return make_sizer(
      2 * n + 6 * m   /* vectors + triangular-solve scratch */
          + 6 * n * n /* P_pred, P_filt, KC, I_minus_KC, temp_nn, BQBT */
          + 2 * m * m /* S_buf, L */
          + 4 * n * m /* PCt, K, temp_nm, H_buf */
          + m * n     /* temp_mn */
          + n * k     /* temp_nk */
          + 2 * k * m /* M, temp_km */
          + n,        /* x_lvl holds x in levels for the measurement */
      0);
}

static int ekf_hot_loop(const ekf_inputs *in, f64 *SDSGE_RESTRICT arena,
                        ekf_outputs *out, ekf_cache *cache) {
  const i64 n = in->n, m = in->m, k = in->k, T = in->T;

  f64 *x_pred_buf = arena;
  f64 *x_filt_buf = x_pred_buf + n;
  f64 *y_pred_buf = x_filt_buf + n;
  f64 *v_buf = y_pred_buf + m;
  f64 *u_buf = v_buf + m;
  f64 *S_inv_v = u_buf + m;
  f64 *P_pred_buf = S_inv_v + m;
  f64 *P_filt_buf = P_pred_buf + n * n;
  f64 *KC = P_filt_buf + n * n;
  f64 *I_minus_KC = KC + n * n;
  f64 *temp_nn = I_minus_KC + n * n;
  f64 *BQBT = temp_nn + n * n;
  f64 *S_buf = BQBT + n * n;
  f64 *L = S_buf + m * m;
  f64 *PCt = L + m * m;
  f64 *K = PCt + n * m;
  f64 *temp_nm = K + n * m;
  f64 *H_buf = temp_nm + n * m;
  f64 *temp_mn = H_buf + m * n;
  f64 *temp_nk = temp_mn + m * n;
  f64 *M = temp_nk + n * k;
  f64 *temp_km = M + k * m;
  f64 *x_lvl = temp_km + k * m;
  /* Whitened in place once S is factored. */
  f64 *W = PCt;

  kf_build_bqbt(in->B, in->Q, temp_nk, BQBT, n, k);

  const f64 const_term = (f64)m * log(TWO_PI);
  f64 loglik = 0.0;

  /* x0/P0 are the prior for the FIRST OBSERVED state, so a period opens on the
   * update and closes on the propagation. Dynare's kalman_filter.m is written
   * the same way, which is what lets the same (x0, P0) mean the same thing on
   * both sides rather than differing by one application of the transition. */
  memcpy(x_pred_buf, in->x0, (size_t)n * sizeof(f64));
  memcpy(P_pred_buf, in->P0, (size_t)(n * n) * sizeof(f64));
  if (in->symmetrize)
    sdsge_sym_inplace(P_pred_buf, n);
  int status = KF_OK;

  for (i64 t = 0; t < T; ++t) {
    /* Nonlinear measurement + relinearization at the predicted state:
     * y_pred := h(x_pred, params);  H_buf := dh/dx(x_pred, params), (m, n). */
    for (i64 i = 0; i < n; ++i) {
      x_lvl[i] = x_pred_buf[i] + in->steady_state[i];
    }

    in->meas(x_lvl, in->calib_params, y_pred_buf);
    in->jac(x_lvl, in->calib_params, H_buf);

    kf_row_minus_vec(in->y, t, y_pred_buf, v_buf, m);
    kf_pc_t(P_pred_buf, H_buf, PCt, n, m);
    kf_measurement_cov(H_buf, PCt, in->R, S_buf, n, m);
    if (in->symmetrize)
      sdsge_sym_inplace(S_buf, m);

    if (sdsge_chol(S_buf, in->jitter, L, m) != SDSGE_OK) {
      status = KF_ERR_S_FACTORIZATION;
      break;
    }

    sdsge_forward_subst(L, v_buf, u_buf, m);
    kf_whiten_pc_t(L, PCt, n, m);

    kf_state_update(x_pred_buf, W, u_buf, x_filt_buf, n, m);

    if (in->joseph_cov) {
      kf_gain_from_whitened(L, W, K, n, m);
      kf_joseph_cov(K, H_buf, P_pred_buf, in->R, KC, I_minus_KC, temp_nn,
                    temp_nm, P_filt_buf, n, m);
    } else {
      kf_cov_downdate(P_pred_buf, W, P_filt_buf, n, m);
    }

    if (in->symmetrize)
      sdsge_sym_inplace(P_filt_buf, n);

    loglik += -0.5 * (const_term + sdsge_logdet_from_chol(L, m) +
                      sdsge_dot(u_buf, u_buf, m));

    if (in->store_history) {
      sdsge_backward_subst_chol_t(L, u_buf, S_inv_v, m);
      /* H_buf changes each step, so rebuild the shock projection per step. */
      kf_build_shock_projection(in->B, H_buf, in->Q, temp_km, M, n, k, m);
      sdsge_matvec(M, S_inv_v, out->eps_hat + t * k, k, m);

      memcpy(out->x_pred + t * n, x_pred_buf, (size_t)n * sizeof(f64));
      memcpy(out->x_filt + t * n, x_filt_buf, (size_t)n * sizeof(f64));

      for (i64 i = 0; i < n; ++i) {
        out->x_pred[n * t + i] += in->steady_state[i];
        out->x_filt[n * t + i] += in->steady_state[i];
      }

      memcpy(out->P_pred + t * n * n, P_pred_buf,
             (size_t)(n * n) * sizeof(f64));
      memcpy(out->P_filt + t * n * n, P_filt_buf,
             (size_t)(n * n) * sizeof(f64));
      memcpy(out->y_pred + t * m, y_pred_buf, (size_t)m * sizeof(f64));

      if (in->steady_state != NULL) {
        for (i64 i = 0; i < n; ++i) {
          x_lvl[i] = x_filt_buf[i] + in->steady_state[i];
        }
        in->meas(x_lvl, in->calib_params, out->y_filt + t * m);
      } else {
        in->meas(x_filt_buf, in->calib_params, out->y_filt + t * m);
      }

      memcpy(out->innov + t * m, v_buf, (size_t)m * sizeof(f64));
      memcpy(out->std_innov + t * m, u_buf, (size_t)m * sizeof(f64));
      memcpy(out->S + t * m * m, S_buf, (size_t)(m * m) * sizeof(f64));
    }

    if (cache) {
      memcpy(cache->W + t * n * m, W, (size_t)(n * m) * sizeof(f64));
      kf_whiten_meas(L, H_buf, cache->G + t * m * n, m, n);
    }

    /* Carry the posterior forward: the next period opens on this prediction.
     */
    sdsge_matvec(in->A, x_filt_buf, x_pred_buf, n, n);
    kf_predict_cov(in->A, P_filt_buf, BQBT, temp_nn, P_pred_buf, n);
    if (in->symmetrize)
      sdsge_sym_inplace(P_pred_buf, n);
  }

  *out->loglik = loglik;
  return status;
}

int ekf_forward(const ekf_inputs *in, f64 *SDSGE_RESTRICT arena,
                ekf_outputs *out) {
  return ekf_hot_loop(in, arena, out, NULL);
}

static arena_size ekf_smooth_loop_arena(const i64 n, const i64 m, const i64 k) {
  const arena_size fwd_arena = ekf_arena_size(n, m, k);
  const arena_size smooth_arena = make_sizer(5 * n * n + 2 * n + m + n * k, 0);
  return sdsge_max_arena(fwd_arena, smooth_arena);
}
arena_size ekf_smooth_arena_size(const i64 n, const i64 m, const i64 k,
                                 const i64 T) {
  return add_arena(ekf_smooth_loop_arena(n, m, k), ekf_cache_size(n, m, T));
}

int ekf_smooth(const ekf_inputs *in, f64 *SDSGE_RESTRICT arena,
               ekf_smooth_outputs *out) {
  int rc;
  ekf_cache cache;
  const i64 akf = ekf_smooth_loop_arena(in->n, in->m, in->k).n_float;

  cache.W = arena + akf;
  cache.G = cache.W + in->T * in->n * in->m;

  rc = ekf_hot_loop(in, arena, out->fwd_out, &cache);
  if (rc != KF_OK) {
    return rc;
  }

  const i64 n = in->n, m = in->m, k = in->k, T = in->T;

  f64 *r = arena;
  f64 *w = r + n;
  f64 *g = w + n;
  f64 *QBt = g + m;
  f64 *N_cur = QBt + k * n;
  f64 *N_next = N_cur + n * n;
  f64 *KC_Lt = N_next + n * n;
  f64 *I_minus_KC = KC_Lt + n * n;
  f64 *tmp_nn = I_minus_KC + n * n;

  memset(r, 0, (size_t)n * sizeof(f64));
  memset(N_cur, 0, (size_t)(n * n) * sizeof(f64));

  sdsge_matmul_abt(in->Q, in->B, QBt, k, k, n);

  for (i64 t = T - 1; t >= 0; --t) {
    const f64 *Wt = cache.W + t * n * m;
    const f64 *Gt = cache.G + t * m * n;
    /* Means + shocks */
    sdsge_matvec_atb(in->A, r, w, n, n);
    sdsge_matvec_atb(Wt, w, g, n, m);
    sdsge_vsub(out->fwd_out->std_innov + t * m, g, g, m);
    sdsge_matvec_atb_plus_vec(Gt, g, w, r, m, n);
    sdsge_matvec(QBt, r, out->eps_smooth + t * k, k, n);

    /* Covariances */
    const f64 *Pt = out->fwd_out->P_pred + t * n * n;
    f64 *Vt = out->P_smooth + t * n * n;

    sdsge_matmul(Wt, Gt, KC_Lt, n, m, n);
    kf_identity_minus(KC_Lt, I_minus_KC, n);
    sdsge_matmul(in->A, I_minus_KC, KC_Lt, n, n, n);

    sdsge_matmul(N_cur, KC_Lt, tmp_nn, n, n, n);
    sdsge_matmul_atb(KC_Lt, tmp_nn, N_next, n, n, n);
    sdsge_matmul_atb(Gt, Gt, tmp_nn, m, n, n);
    for (i64 i = 0; i < n * n; ++i) {
      N_next[i] += tmp_nn[i];
    }

    sdsge_matmul(Pt, N_next, tmp_nn, n, n, n);
    sdsge_matmul(tmp_nn, Pt, Vt, n, n, n);
    sdsge_vsub(Pt, Vt, Vt, n * n);
    if (in->symmetrize) {
      sdsge_sym_inplace(N_next, n);
      sdsge_sym_inplace(Vt, n);
    }

    f64 *tmp = N_cur;
    N_cur = N_next;
    N_next = tmp;
  }

  sdsge_matvec_plus_vec(in->P0, r, in->x0, w, n, n); /* x_s0 = x0 + P0 * r0 */
  for (i64 t = 0; t < T; ++t) {
    if (t > 0) {
      sdsge_matvec(in->A, w, r, n, n);
      sdsge_matvec_plus_vec(in->B, out->eps_smooth + t * k, r, w, n, k);
    }
    for (i64 i = 0; i < n; ++i) {
      out->x_smooth[t * n + i] = w[i] + in->steady_state[i];
    }
  }

  return KF_OK;
}
