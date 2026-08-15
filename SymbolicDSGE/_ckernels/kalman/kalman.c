#include "kalman.h"
#include "../_common/sdsge_perturbation.h" /* sdsge_second_order_step */
#include <math.h>
#include <stdlib.h>
#include <string.h>

void kf_row_minus_vec(const f64 *SDSGE_RESTRICT A, i64 row,
                      const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT out,
                      i64 m) {
  const f64 *Arow = A + row * m;
  for (i64 j = 0; j < m; ++j)
    out[j] = Arow[j] - x[j];
}

void kf_chol_solve_row(const f64 *SDSGE_RESTRICT L, const f64 *SDSGE_RESTRICT B,
                       i64 row, f64 *SDSGE_RESTRICT fbuf,
                       f64 *SDSGE_RESTRICT bbuf, f64 *SDSGE_RESTRICT out,
                       i64 n) {
  const f64 *Brow = B + row * n;
  for (i64 i = 0; i < n; ++i) {
    f64 s = 0.0;
    for (i64 j = 0; j < i; ++j)
      s += L[i * n + j] * fbuf[j];
    fbuf[i] = (Brow[i] - s) / L[i * n + i];
  }
  for (i64 i = n - 1; i >= 0; --i) {
    f64 s = 0.0;
    for (i64 j = i + 1; j < n; ++j)
      s += L[j * n + i] * bbuf[j];
    bbuf[i] = (fbuf[i] - s) / L[i * n + i];
  }
  f64 *outrow = out + row * n;
  for (i64 i = 0; i < n; ++i)
    outrow[i] = bbuf[i];
}

void kf_predict_cov(const f64 *SDSGE_RESTRICT A,
                    const f64 *SDSGE_RESTRICT P_prev,
                    const f64 *SDSGE_RESTRICT BQBT, f64 *SDSGE_RESTRICT temp_nn,
                    f64 *SDSGE_RESTRICT out, i64 n) {
  sdsge_matmul(A, P_prev, temp_nn, n, n, n);
  sdsge_matmul_abt_plus_c(temp_nn, A, BQBT, out, n, n, n);
}

void kf_measurement_cov(const f64 *SDSGE_RESTRICT C,
                        const f64 *SDSGE_RESTRICT P_pred,
                        const f64 *SDSGE_RESTRICT R,
                        f64 *SDSGE_RESTRICT temp_mn, f64 *SDSGE_RESTRICT out,
                        i64 n, i64 m) {
  sdsge_matmul(C, P_pred, temp_mn, m, n, n);
  sdsge_matmul_abt_plus_c(temp_mn, C, R, out, m, n, m);
}

void kf_pc_t(const f64 *SDSGE_RESTRICT P_pred, const f64 *SDSGE_RESTRICT C,
             f64 *SDSGE_RESTRICT out, i64 n, i64 m) {
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < m; ++j) {
      f64 s = 0.0;
      for (i64 k = 0; k < n; ++k)
        s += P_pred[i * n + k] * C[j * n + k];
      out[i * m + j] = s;
    }
  }
}

void kf_gain_from_pc_t(const f64 *SDSGE_RESTRICT L,
                       const f64 *SDSGE_RESTRICT PCt, f64 *SDSGE_RESTRICT fbuf,
                       f64 *SDSGE_RESTRICT bbuf, f64 *SDSGE_RESTRICT out, i64 n,
                       i64 m) {
  for (i64 row = 0; row < n; ++row)
    kf_chol_solve_row(L, PCt, row, fbuf, bbuf, out, m);
}

void kf_state_update(const f64 *SDSGE_RESTRICT x_pred,
                     const f64 *SDSGE_RESTRICT K, const f64 *SDSGE_RESTRICT v,
                     f64 *SDSGE_RESTRICT out, i64 n, i64 m) {
  for (i64 i = 0; i < n; ++i) {
    f64 s = x_pred[i];
    for (i64 j = 0; j < m; ++j)
      s += K[i * m + j] * v[j];
    out[i] = s;
  }
}

void kf_identity_minus(const f64 *SDSGE_RESTRICT A, f64 *SDSGE_RESTRICT out,
                       i64 n) {
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < n; ++j)
      out[i * n + j] = -A[i * n + j];
    out[i * n + i] += 1.0;
  }
}

void kf_joseph_cov(const f64 *SDSGE_RESTRICT K, const f64 *SDSGE_RESTRICT C,
                   const f64 *SDSGE_RESTRICT P_pred,
                   const f64 *SDSGE_RESTRICT R, f64 *SDSGE_RESTRICT KC,
                   f64 *SDSGE_RESTRICT I_minus_KC, f64 *SDSGE_RESTRICT temp_nn,
                   f64 *SDSGE_RESTRICT temp_nm, f64 *SDSGE_RESTRICT out, i64 n,
                   i64 m) {
  sdsge_matmul(K, C, KC, n, m, n);
  kf_identity_minus(KC, I_minus_KC, n);
  sdsge_matmul(I_minus_KC, P_pred, temp_nn, n, n, n);
  sdsge_matmul_abt(temp_nn, I_minus_KC, out, n, n, n);
  sdsge_matmul(K, R, temp_nm, n, m, m);
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < n; ++j) {
      f64 s = 0.0;
      for (i64 k = 0; k < m; ++k)
        s += temp_nm[i * m + k] * K[j * m + k];
      out[i * n + j] += s;
    }
  }
}

void kf_build_bqbt(const f64 *SDSGE_RESTRICT B, const f64 *SDSGE_RESTRICT Q,
                   f64 *SDSGE_RESTRICT temp_nk, f64 *SDSGE_RESTRICT out, i64 n,
                   i64 k) {
  sdsge_matmul(B, Q, temp_nk, n, k, k);
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < n; ++j) {
      f64 s = 0.0;
      for (i64 l = 0; l < k; ++l)
        s += temp_nk[i * k + l] * B[j * k + l];
      out[i * n + j] = s;
    }
  }
  sdsge_sym_inplace(out, n);
}

void kf_build_shock_projection(const f64 *SDSGE_RESTRICT B,
                               const f64 *SDSGE_RESTRICT C,
                               const f64 *SDSGE_RESTRICT Q,
                               f64 *SDSGE_RESTRICT temp_km,
                               f64 *SDSGE_RESTRICT out, i64 n, i64 k, i64 m) {
  for (i64 i = 0; i < k; ++i) {
    for (i64 j = 0; j < m; ++j) {
      f64 s = 0.0;
      for (i64 l = 0; l < n; ++l)
        s += B[l * k + i] * C[j * n + l];
      temp_km[i * m + j] = s;
    }
  }
  sdsge_matmul(Q, temp_km, out, k, k, m);
}

arena_size kf_arena_size(const i64 n, const i64 m, const i64 k) {
  return make_sizer(
      2 * n + 7 * m    /* vectors + triangular-solve scratch */
          + 6 * n * n  /* P_pred, P_filt, KC, I_minus_KC, temp_nn, BQBT */
          + 2 * m * m  /* S_buf, L */
          + 3 * n * m  /* PCt, K, temp_nm */
          + m * n      /* temp_mn */
          + n * k      /* temp_nk */
          + 2 * k * m, /* M, temp_km */
      0);
}
int kf_hot_loop(const kf_inputs *in, f64 *SDSGE_RESTRICT arena,
                kf_outputs *out) {
  const i64 n = in->n, m = in->m, k = in->k, T = in->T;

  f64 *x_pred_buf = arena;
  f64 *x_filt_buf = x_pred_buf + n;
  f64 *y_pred_buf = x_filt_buf + n;
  f64 *y_filt_buf = y_pred_buf + m;
  f64 *v_buf = y_filt_buf + m;
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
  f64 *temp_mn = temp_nm + n * m;
  f64 *temp_nk = temp_mn + m * n;
  f64 *M = temp_nk + n * k;
  f64 *temp_km = M + k * m;

  kf_build_bqbt(in->B, in->Q, temp_nk, BQBT, n, k);
  if (in->return_shocks && in->store_history)
    kf_build_shock_projection(in->B, in->C, in->Q, temp_km, M, n, k, m);

  const f64 const_term = (f64)m * log(TWO_PI); /* m * log(2*pi) */
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
    sdsge_matvec_plus_vec(in->C, x_pred_buf, in->d, y_pred_buf, m, n);
    kf_row_minus_vec(in->y, t, y_pred_buf, v_buf, m);
    kf_measurement_cov(in->C, P_pred_buf, in->R, temp_mn, S_buf, n, m);
    if (in->symmetrize)
      sdsge_sym_inplace(S_buf, m);

    if (sdsge_chol(S_buf, in->jitter, L, m) != SDSGE_OK) {
      status = KF_ERR_MATRIX_CONDITION;
      break;
    }

    sdsge_forward_subst(L, v_buf, u_buf, m);
    sdsge_backward_subst_chol_t(L, u_buf, S_inv_v, m);

    kf_pc_t(P_pred_buf, in->C, PCt, n, m);
    kf_gain_from_pc_t(L, PCt, solve_f, solve_b, K, n, m);

    kf_state_update(x_pred_buf, K, v_buf, x_filt_buf, n, m);
    kf_joseph_cov(K, in->C, P_pred_buf, in->R, KC, I_minus_KC, temp_nn, temp_nm,
                  P_filt_buf, n, m);
    if (in->symmetrize)
      sdsge_sym_inplace(P_filt_buf, n);

    loglik += -0.5 * (const_term + sdsge_logdet_from_chol(L, m) +
                      sdsge_dot(v_buf, S_inv_v, m));

    if (in->return_shocks && in->store_history)
      sdsge_matvec(M, S_inv_v, out->eps_hat + t * k, k, m);

    if (in->store_history) {
      sdsge_matvec_plus_vec(in->C, x_filt_buf, in->d, y_filt_buf, m, n);
      memcpy(out->x_pred + t * n, x_pred_buf, (size_t)n * sizeof(f64));
      memcpy(out->x_filt + t * n, x_filt_buf, (size_t)n * sizeof(f64));
      memcpy(out->P_pred + t * n * n, P_pred_buf,
             (size_t)(n * n) * sizeof(f64));
      memcpy(out->P_filt + t * n * n, P_filt_buf,
             (size_t)(n * n) * sizeof(f64));
      memcpy(out->y_pred + t * m, y_pred_buf, (size_t)m * sizeof(f64));
      memcpy(out->y_filt + t * m, y_filt_buf, (size_t)m * sizeof(f64));
      memcpy(out->innov + t * m, v_buf, (size_t)m * sizeof(f64));
      memcpy(out->std_innov + t * m, u_buf, (size_t)m * sizeof(f64));
      memcpy(out->S + t * m * m, S_buf, (size_t)(m * m) * sizeof(f64));
    }

    /* Carry the posterior forward: the next period opens on this prediction. */
    sdsge_matvec(in->A, x_filt_buf, x_pred_buf, n, n);
    kf_predict_cov(in->A, P_filt_buf, BQBT, temp_nn, P_pred_buf, n);
    if (in->symmetrize)
      sdsge_sym_inplace(P_pred_buf, n);
  }

  *out->loglik = loglik;
  return status;
}

arena_size ekf_arena_size(const i64 n, const i64 m, const i64 k) {
  return make_sizer(
      2 * n + 6 * m    /* vectors + triangular-solve scratch */
          + 6 * n * n  /* P_pred, P_filt, KC, I_minus_KC, temp_nn, BQBT */
          + 2 * m * m  /* S_buf, L */
          + 4 * n * m  /* PCt, K, temp_nm, H_buf */
          + m * n      /* temp_mn */
          + n * k      /* temp_nk */
          + 2 * k * m, /* M, temp_km */
      0);
}

int ekf_hot_loop(const ekf_inputs *in, f64 *SDSGE_RESTRICT arena,
                 ekf_outputs *out) {
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
    in->meas(x_pred_buf, in->calib_params, y_pred_buf);
    in->jac(x_pred_buf, in->calib_params, H_buf);

    kf_row_minus_vec(in->y, t, y_pred_buf, v_buf, m);
    kf_measurement_cov(H_buf, P_pred_buf, in->R, temp_mn, S_buf, n, m);
    if (in->symmetrize)
      sdsge_sym_inplace(S_buf, m);

    if (sdsge_chol(S_buf, in->jitter, L, m) != SDSGE_OK) {
      status = KF_ERR_MATRIX_CONDITION;
      break;
    }

    sdsge_forward_subst(L, v_buf, u_buf, m);
    sdsge_backward_subst_chol_t(L, u_buf, S_inv_v, m);

    kf_pc_t(P_pred_buf, H_buf, PCt, n, m);
    kf_gain_from_pc_t(L, PCt, solve_f, solve_b, K, n, m);

    kf_state_update(x_pred_buf, K, v_buf, x_filt_buf, n, m);
    kf_joseph_cov(K, H_buf, P_pred_buf, in->R, KC, I_minus_KC, temp_nn, temp_nm,
                  P_filt_buf, n, m);
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
      memcpy(out->P_pred + t * n * n, P_pred_buf,
             (size_t)(n * n) * sizeof(f64));
      memcpy(out->P_filt + t * n * n, P_filt_buf,
             (size_t)(n * n) * sizeof(f64));
      memcpy(out->y_pred + t * m, y_pred_buf, (size_t)m * sizeof(f64));
      if (in->compute_y_filt)
        in->meas(x_filt_buf, in->calib_params, out->y_filt + t * m);
      memcpy(out->innov + t * m, v_buf, (size_t)m * sizeof(f64));
      memcpy(out->std_innov + t * m, u_buf, (size_t)m * sizeof(f64));
      memcpy(out->S + t * m * m, S_buf, (size_t)(m * m) * sizeof(f64));
    }

    /* Carry the posterior forward: the next period opens on this prediction. */
    sdsge_matvec(in->A, x_filt_buf, x_pred_buf, n, n);
    kf_predict_cov(in->A, P_filt_buf, BQBT, temp_nn, P_pred_buf, n);
    if (in->symmetrize)
      sdsge_sym_inplace(P_pred_buf, n);
  }

  *out->loglik = loglik;
  return status;
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

static void ukf_cov_update(const f64 *SDSGE_RESTRICT P_pred,
                           const f64 *SDSGE_RESTRICT K,
                           const f64 *SDSGE_RESTRICT Pzy,
                           f64 *SDSGE_RESTRICT P_filt, i64 nz, i64 no) {
  for (i64 i = 0; i < nz; ++i) {
    for (i64 j = 0; j < nz; ++j) {
      f64 s = P_pred[i * nz + j];
      for (i64 l = 0; l < no; ++l)
        s -= K[i * no + l] * Pzy[j * no + l];
      P_filt[i * nz + j] = s;
    }
  }
}

static void ukf_store_history(const ukf_inputs *in,
                              const f64 *SDSGE_RESTRICT z,
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

arena_size ukf_arena_size(const i64 n_state, const i64 n_ctrl,
                           const i64 n_exog, const i64 n_obs) {
  const i64 nz = 2 * n_state;
  const i64 na = nz + n_exog;
  const i64 n_sig = 2 * na + 1;
  const i64 nv = n_state + n_ctrl;

  return make_sizer(3 * nz + 4 * nz * nz + na * na + n_exog * n_exog +
                        n_sig * na + n_sig * nz + n_sig * n_obs + 6 * n_obs +
                        2 * n_obs * n_obs + 2 * nz * n_obs +
                        n_sig * nv + 2 * nv + 2 * nv * n_obs + na +
                        sdsge_second_order_step_scratch(n_state, n_exog),
                    0);
}
i64 ukf_hot_loop(const ukf_inputs *in, f64 *SDSGE_RESTRICT arena,
                 ukf_outputs *out) {
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
    return KF_ERR_MATRIX_CONDITION;
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
  f64 *solve_f = S_inv_v + no;
  f64 *solve_b = solve_f + no;
  f64 *S = solve_b + no;
  f64 *L = S + no * no;
  f64 *Pzy = L + no * no;
  f64 *K = Pzy + nz * no;
  f64 *sigma_v = K + nz * no;
  f64 *vars_pred = sigma_v + n_sig * nv;
  f64 *vars_filt = vars_pred + nv;
  f64 *Pvy = vars_filt + nv;
  f64 *Kv = Pvy + nv * no;
  f64 *z_aug = Kv + nv * no;

  memcpy(z_prev, in->z0, (size_t)nz * sizeof(f64));
  memcpy(P_prev, in->P0, (size_t)(nz * nz) * sizeof(f64));

  /* Q is fixed for the run, so its factor is taken once. The augmented root is
   * block diagonal and only its state block moves, so the innovation block is
   * written here and never again. */
  if (ne > 0 && ukf_chol_auto(in->Q, in->jitter, Q_chol, ne) != SDSGE_OK)
    return KF_ERR_MATRIX_CONDITION;
  sdsge_zero_mat(A_chol, na, na);
  for (i64 i = 0; i < ne; ++i)
    for (i64 j = 0; j < ne; ++j)
      A_chol[(nz + i) * na + nz + j] = Q_chol[i * ne + j];

  const f64 const_term = (f64)no * log(TWO_PI);
  f64 loglik = 0.0;
  i64 status = KF_OK;

  for (i64 t = 0; t < T; ++t) {
    if (ukf_chol_auto(P_prev, in->jitter, P_chol, nz) != SDSGE_OK) {
      status = KF_ERR_MATRIX_CONDITION;
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

    /* One sweep: the same point gives the state it carries forward and the
     * observation of the period it lands in, which is the only way the
     * measurement sees this period's innovation. */
    for (i64 r = 0; r < n_sig; ++r) {
      const f64 *SDSGE_RESTRICT a = sigma_a + r * na;
      f64 *SDSGE_RESTRICT zn = sigma_z + r * nz;
      sdsge_second_order_step(in->hx, in->gx, in->bu, in->hxx, in->gxx,
                              in->hxu, in->gxu, in->huu, in->guu, in->hss,
                              in->gss, a, a + ns, ne > 0 ? a + nz : NULL,
                              in->steady_state, zn, zn + ns, sigma_v + r * nv,
                              step, ns, nc, ne);
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
      status = KF_ERR_MATRIX_CONDITION;
      break;
    }
    sdsge_forward_subst(L, innov, std_innov, no);
    sdsge_backward_subst_chol_t(L, std_innov, S_inv_v, no);

    kf_gain_from_pc_t(L, Pzy, solve_f, solve_b, K, nz, no);
    kf_state_update(z_pred, K, innov, z_filt, nz, no);
    ukf_cov_update(P_pred, K, Pzy, P_filt, nz, no);
    if (in->symmetrize)
      sdsge_sym_inplace(P_filt, nz);

    loglik += -0.5 * (const_term + sdsge_logdet_from_chol(L, no) +
                      sdsge_dot(innov, S_inv_v, no));

    if (in->store_history) {
      /* The variable vector is an output, not a state: it is never carried and
       * never initialized, so it is filtered by its own gain against the same
       * innovation rather than re-derived from z_filt. Deriving it there would
       * drop this period's shock, which no longer exists once the sigma set has
       * been marginalized down to the pruned state. */
      ukf_weighted_mean(sigma_v, w0_m, wi, n_sig, nv, vars_pred);
      ukf_weighted_cross(sigma_v, vars_pred, sigma_y, y_pred, w0_c, wi, n_sig,
                         nv, no, Pvy);
      kf_gain_from_pc_t(L, Pvy, solve_f, solve_b, Kv, nv, no);
      kf_state_update(vars_pred, Kv, innov, vars_filt, nv, no);

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
