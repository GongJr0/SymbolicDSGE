#include "extended.h"
#include "kalman.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  f64 *S_inv_v; /* (T, m) */
  f64 *K;       /* (T, n, m) */
  f64 *H;       /* (T, m, n) */
} ekf_cache;

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
  f64 *solve_f = S_inv_v + m;
  f64 *solve_b = solve_f + m;
  f64 *P_pred_buf = solve_b + m;
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
    kf_measurement_cov(H_buf, P_pred_buf, in->R, temp_mn, S_buf, n, m);
    if (in->symmetrize)
      sdsge_sym_inplace(S_buf, m);

    if (sdsge_chol(S_buf, in->jitter, L, m) != SDSGE_OK) {
      status = KF_ERR_S_FACTORIZATION;
      break;
    }

    sdsge_forward_subst(L, v_buf, u_buf, m);
    sdsge_backward_subst_chol_t(L, u_buf, S_inv_v, m);

    kf_pc_t(P_pred_buf, H_buf, PCt, n, m);
    kf_gain_from_pc_t(L, PCt, solve_f, solve_b, K, n, m);

    kf_state_update(x_pred_buf, K, v_buf, x_filt_buf, n, m);

    if (in->joseph_cov) {
      kf_joseph_cov(K, H_buf, P_pred_buf, in->R, KC, I_minus_KC, temp_nn,
                    temp_nm, P_filt_buf, n, m);
    } else {
      kf_simple_cov(K, PCt, P_pred_buf, P_filt_buf, n, m);
    }

    if (in->symmetrize)
      sdsge_sym_inplace(P_filt_buf, n);

    loglik += -0.5 * (const_term + sdsge_logdet_from_chol(L, m) +
                      sdsge_dot(v_buf, S_inv_v, m));

    if (in->return_shocks && in->store_history) {
      /* H_buf changes each step, so rebuild the shock projection per step. */
      kf_build_shock_projection(in->B, H_buf, in->Q, temp_km, M, n, k, m);
      sdsge_matvec(M, S_inv_v, out->eps_hat + t * k, k, m);
    }

    if (in->store_history) {
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

      if (in->compute_y_filt) {
        if (in->steady_state != NULL) {
          for (i64 i = 0; i < n; ++i) {
            x_lvl[i] = x_filt_buf[i] + in->steady_state[i];
          }
          in->meas(x_lvl, in->calib_params, out->y_filt + t * m);
        } else {
          in->meas(x_filt_buf, in->calib_params, out->y_filt + t * m);
        }
      }

      memcpy(out->innov + t * m, v_buf, (size_t)m * sizeof(f64));
      memcpy(out->std_innov + t * m, u_buf, (size_t)m * sizeof(f64));
      memcpy(out->S + t * m * m, S_buf, (size_t)(m * m) * sizeof(f64));
    }

    if (cache) {
      memcpy(cache->S_inv_v + t * m, S_inv_v, (size_t)m * sizeof(f64));
      memcpy(cache->K + t * n * m, K, (size_t)(n * m) * sizeof(f64));
      memcpy(cache->H + t * m * n, H_buf, (size_t)(m * n) * sizeof(f64));
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
