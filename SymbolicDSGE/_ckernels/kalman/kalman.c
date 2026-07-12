#include "kalman.h"
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

int kf_hot_loop(const kf_inputs *in, kf_outputs *out) {
  const i64 n = in->n, m = in->m, k = in->k, T = in->T;

  /* One scratch arena carved into every per-step buffer (allocated once). */
  const i64 total =
      2 * n + 7 * m /* vectors + triangular-solve scratch */
      + 6 * n * n   /* P_pred, P_filt, KC, I_minus_KC, temp_nn, BQBT */
      + 2 * m * m   /* S_buf, L */
      + 3 * n * m   /* PCt, K, temp_nm */
      + m * n       /* temp_mn */
      + n * k       /* temp_nk */
      + 2 * k * m;  /* M, temp_km */
  f64 *arena = (f64 *)malloc((size_t)total * sizeof(f64));
  if (arena == NULL)
    return KF_ERR_ALLOC;

  f64 *p = arena;
  f64 *x_pred_buf = p;
  p += n;
  f64 *x_filt_buf = p;
  p += n;
  f64 *y_pred_buf = p;
  p += m;
  f64 *y_filt_buf = p;
  p += m;
  f64 *v_buf = p;
  p += m;
  f64 *u_buf = p;
  p += m;
  f64 *S_inv_v = p;
  p += m;
  f64 *solve_f = p;
  p += m;
  f64 *solve_b = p;
  p += m;
  f64 *P_pred_buf = p;
  p += n * n;
  f64 *P_filt_buf = p;
  p += n * n;
  f64 *KC = p;
  p += n * n;
  f64 *I_minus_KC = p;
  p += n * n;
  f64 *temp_nn = p;
  p += n * n;
  f64 *BQBT = p;
  p += n * n;
  f64 *S_buf = p;
  p += m * m;
  f64 *L = p;
  p += m * m;
  f64 *PCt = p;
  p += n * m;
  f64 *K = p;
  p += n * m;
  f64 *temp_nm = p;
  p += n * m;
  f64 *temp_mn = p;
  p += m * n;
  f64 *temp_nk = p;
  p += n * k;
  f64 *M = p;
  p += k * m;
  f64 *temp_km = p;
  p += k * m;

  kf_build_bqbt(in->B, in->Q, temp_nk, BQBT, n, k);
  if (in->return_shocks && in->store_history)
    kf_build_shock_projection(in->B, in->C, in->Q, temp_km, M, n, k, m);

  const f64 const_term = (f64)m * log(TWO_PI); /* m * log(2*pi) */
  f64 loglik = 0.0;

  /* x_prev/P_prev start at the inputs, then alias the filtered scratch; each
   * is read (into x_pred_buf / P_pred_buf) before being overwritten. */
  const f64 *x_prev = in->x0;
  const f64 *P_prev = in->P0;
  int status = KF_OK;

  for (i64 t = 0; t < T; ++t) {
    sdsge_matvec(in->A, x_prev, x_pred_buf, n, n);
    kf_predict_cov(in->A, P_prev, BQBT, temp_nn, P_pred_buf, n);
    if (in->symmetrize)
      sdsge_sym_inplace(P_pred_buf, n);

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

    x_prev = x_filt_buf;
    P_prev = P_filt_buf;
  }

  *out->loglik = loglik;
  free(arena);
  return status;
}

int ekf_hot_loop(const ekf_inputs *in, ekf_outputs *out) {
  const i64 n = in->n, m = in->m, k = in->k, T = in->T;

  /* One scratch arena carved into every per-step buffer (allocated once). Same
   * layout as the linear filter, plus a per-step measurement jacobian H_buf(m,n)
   * since the EKF relinearizes each step. y_filt is written straight into the
   * output (only when compute_y_filt), so it needs no scratch vector. */
  const i64 total =
      2 * n + 6 * m /* vectors + triangular-solve scratch */
      + 6 * n * n   /* P_pred, P_filt, KC, I_minus_KC, temp_nn, BQBT */
      + 2 * m * m   /* S_buf, L */
      + 4 * n * m   /* PCt, K, temp_nm, H_buf */
      + m * n       /* temp_mn */
      + n * k       /* temp_nk */
      + 2 * k * m;  /* M, temp_km */
  f64 *arena = (f64 *)malloc((size_t)total * sizeof(f64));
  if (arena == NULL)
    return KF_ERR_ALLOC;

  f64 *p = arena;
  f64 *x_pred_buf = p;
  p += n;
  f64 *x_filt_buf = p;
  p += n;
  f64 *y_pred_buf = p;
  p += m;
  f64 *v_buf = p;
  p += m;
  f64 *u_buf = p;
  p += m;
  f64 *S_inv_v = p;
  p += m;
  f64 *solve_f = p;
  p += m;
  f64 *solve_b = p;
  p += m;
  f64 *P_pred_buf = p;
  p += n * n;
  f64 *P_filt_buf = p;
  p += n * n;
  f64 *KC = p;
  p += n * n;
  f64 *I_minus_KC = p;
  p += n * n;
  f64 *temp_nn = p;
  p += n * n;
  f64 *BQBT = p;
  p += n * n;
  f64 *S_buf = p;
  p += m * m;
  f64 *L = p;
  p += m * m;
  f64 *PCt = p;
  p += n * m;
  f64 *K = p;
  p += n * m;
  f64 *temp_nm = p;
  p += n * m;
  f64 *H_buf = p;
  p += m * n;
  f64 *temp_mn = p;
  p += m * n;
  f64 *temp_nk = p;
  p += n * k;
  f64 *M = p;
  p += k * m;
  f64 *temp_km = p;
  p += k * m;

  kf_build_bqbt(in->B, in->Q, temp_nk, BQBT, n, k);

  const f64 const_term = (f64)m * log(TWO_PI);
  f64 loglik = 0.0;

  const f64 *x_prev = in->x0;
  const f64 *P_prev = in->P0;
  int status = KF_OK;

  for (i64 t = 0; t < T; ++t) {
    sdsge_matvec(in->A, x_prev, x_pred_buf, n, n);
    kf_predict_cov(in->A, P_prev, BQBT, temp_nn, P_pred_buf, n);
    if (in->symmetrize)
      sdsge_sym_inplace(P_pred_buf, n);

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

    x_prev = x_filt_buf;
    P_prev = P_filt_buf;
  }

  *out->loglik = loglik;
  free(arena);
  return status;
}

static void ukf_build_sigma_points(const f64 *SDSGE_RESTRICT mean,
                                   const f64 *SDSGE_RESTRICT chol,
                                   f64 gamma, f64 *SDSGE_RESTRICT sigma,
                                   i64 n) {
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

static void ukf_pruned_transition(const ukf_inputs *in,
                                  const f64 *SDSGE_RESTRICT z,
                                  f64 *SDSGE_RESTRICT out) {
  const i64 ns = in->n_state;
  const f64 *x1 = z;
  const f64 *x2 = z + ns;
  f64 *x1_next = out;
  f64 *x2_next = out + ns;

  for (i64 i = 0; i < ns; ++i) {
    f64 s1 = 0.0;
    f64 s2 = 0.5 * in->hss[i];
    for (i64 j = 0; j < ns; ++j) {
      const f64 hxij = in->hx[i * ns + j];
      s1 += hxij * x1[j];
      s2 += hxij * x2[j];
    }
    const f64 *hxx_i = in->hxx + i * ns * ns;
    for (i64 j = 0; j < ns; ++j)
      for (i64 k = 0; k < ns; ++k)
        s2 += 0.5 * hxx_i[j * ns + k] * x1[j] * x1[k];
    x1_next[i] = s1;
    x2_next[i] = s2;
  }
}

static void ukf_project_vars(const ukf_inputs *in,
                             const f64 *SDSGE_RESTRICT z,
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

static void ukf_eval_measurement_sigma(const ukf_inputs *in,
                                       const f64 *SDSGE_RESTRICT sigma_z,
                                       i64 n_sig,
                                       f64 *SDSGE_RESTRICT vars_buf,
                                       f64 *SDSGE_RESTRICT sigma_y) {
  const i64 nz = 2 * in->n_state;
  const i64 no = in->n_obs;
  for (i64 r = 0; r < n_sig; ++r) {
    ukf_project_vars(in, sigma_z + r * nz, vars_buf);
    in->meas(vars_buf, in->params, sigma_y + r * no);
  }
}

static void ukf_weighted_meas_cov_cross(
    const f64 *SDSGE_RESTRICT sigma_z, const f64 *SDSGE_RESTRICT z_mean,
    const f64 *SDSGE_RESTRICT sigma_y, const f64 *SDSGE_RESTRICT y_mean, f64 w0,
    f64 wi, i64 n_sig, i64 nz, i64 no, f64 *SDSGE_RESTRICT S,
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

static void ukf_add_process_cov(const f64 *SDSGE_RESTRICT BQBT,
                                f64 *SDSGE_RESTRICT P, i64 ns, i64 nz) {
  for (i64 i = 0; i < ns; ++i)
    for (i64 j = 0; j < ns; ++j)
      P[i * nz + j] += BQBT[i * ns + j];
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
                              const f64 *SDSGE_RESTRICT P,
                              f64 *SDSGE_RESTRICT x1,
                              f64 *SDSGE_RESTRICT x2,
                              f64 *SDSGE_RESTRICT x, i64 t) {
  const i64 ns = in->n_state;
  const i64 nc = in->n_ctrl;
  const i64 nz = 2 * ns;
  const i64 nv = ns + nc;
  const f64 *z1 = z;
  const f64 *z2 = z + ns;
  f64 *x1_row = x1 + t * ns;
  f64 *x2_row = x2 + t * ns;
  f64 *x_row = x + t * nv;

  memcpy(x1_row, z1, (size_t)ns * sizeof(f64));
  memcpy(x2_row, z2, (size_t)ns * sizeof(f64));

  for (i64 i = 0; i < ns; ++i)
    x_row[i] = in->steady_state[i] + z1[i] + z2[i];

  for (i64 i = 0; i < nc; ++i) {
    f64 s = 0.5 * in->gss[i];
    const f64 *gx_i = in->gx + i * ns;
    for (i64 j = 0; j < ns; ++j)
      s += gx_i[j] * (z1[j] + z2[j]);
    const f64 *gxx_i = in->gxx + i * ns * ns;
    for (i64 j = 0; j < ns; ++j) {
      for (i64 k = 0; k < ns; ++k) {
        const f64 m2 = P[j * nz + k] + z1[j] * z1[k];
        s += 0.5 * gxx_i[j * ns + k] * m2;
      }
    }
    x_row[ns + i] = in->steady_state[ns + i] + s;
  }
}

i64 ukf_hot_loop(const ukf_inputs *in, ukf_outputs *out) {
  const i64 ns = in->n_state;
  const i64 nc = in->n_ctrl;
  const i64 ne = in->n_exog;
  const i64 no = in->n_obs;
  const i64 T = in->T;
  const i64 nz = 2 * ns;
  const i64 n_sig = 2 * nz + 1;
  const i64 nv = ns + nc;

  if (in->meas == NULL || ns <= 0 || no <= 0 || nz <= 0)
    return KF_ERR_SHAPE_MISMATCH;

  const f64 lambda = in->alpha * in->alpha * ((f64)nz + in->kappa) - (f64)nz;
  const f64 scale = (f64)nz + lambda;
  if (!(scale > 0.0) || !isfinite(scale))
    return KF_ERR_MATRIX_CONDITION;
  const f64 gamma = sqrt(scale);
  const f64 w0_m = lambda / scale;
  const f64 w0_c = w0_m + (1.0 - in->alpha * in->alpha + in->beta);
  const f64 wi = 1.0 / (2.0 * scale);

  const i64 total =
      3 * nz + 4 * nz * nz + 2 * n_sig * nz + n_sig * no + ns * ns +
      ns * ne + 6 * no + 2 * no * no + 2 * nz * no + nv;

  f64 *arena = (f64 *)malloc((size_t)total * sizeof(f64));
  if (arena == NULL)
    return KF_ERR_ALLOC;

  f64 *p = arena;
  f64 *z_prev = p;
  p += nz;
  f64 *z_pred = p;
  p += nz;
  f64 *z_filt = p;
  p += nz;
  f64 *P_prev = p;
  p += nz * nz;
  f64 *P_pred = p;
  p += nz * nz;
  f64 *P_filt = p;
  p += nz * nz;
  f64 *P_chol = p;
  p += nz * nz;
  f64 *sigma_z = p;
  p += n_sig * nz;
  f64 *sigma_z_next = p;
  p += n_sig * nz;
  f64 *sigma_y = p;
  p += n_sig * no;
  f64 *BQBT = p;
  p += ns * ns;
  f64 *temp_nk = p;
  p += ns * ne;
  f64 *y_pred = p;
  p += no;
  f64 *innov = p;
  p += no;
  f64 *std_innov = p;
  p += no;
  f64 *S_inv_v = p;
  p += no;
  f64 *solve_f = p;
  p += no;
  f64 *solve_b = p;
  p += no;
  f64 *S = p;
  p += no * no;
  f64 *L = p;
  p += no * no;
  f64 *Pzy = p;
  p += nz * no;
  f64 *K = p;
  p += nz * no;
  f64 *vars_buf = p;

  memcpy(z_prev, in->z0, (size_t)nz * sizeof(f64));
  memcpy(P_prev, in->P0, (size_t)(nz * nz) * sizeof(f64));
  kf_build_bqbt(in->bx, in->Q, temp_nk, BQBT, ns, ne);

  const f64 const_term = (f64)no * log(TWO_PI);
  f64 loglik = 0.0;
  i64 status = KF_OK;

  for (i64 t = 0; t < T; ++t) {
    if (sdsge_chol(P_prev, in->jitter, P_chol, nz) != SDSGE_OK) {
      status = KF_ERR_MATRIX_CONDITION;
      break;
    }
    ukf_build_sigma_points(z_prev, P_chol, gamma, sigma_z, nz);

    for (i64 r = 0; r < n_sig; ++r)
      ukf_pruned_transition(in, sigma_z + r * nz, sigma_z_next + r * nz);

    ukf_weighted_mean(sigma_z_next, w0_m, wi, n_sig, nz, z_pred);
    ukf_weighted_cov(sigma_z_next, z_pred, w0_c, wi, n_sig, nz, P_pred);
    ukf_add_process_cov(BQBT, P_pred, ns, nz);
    if (in->symmetrize)
      sdsge_sym_inplace(P_pred, nz);

    if (sdsge_chol(P_pred, in->jitter, P_chol, nz) != SDSGE_OK) {
      status = KF_ERR_MATRIX_CONDITION;
      break;
    }
    ukf_build_sigma_points(z_pred, P_chol, gamma, sigma_z, nz);
    ukf_eval_measurement_sigma(in, sigma_z, n_sig, vars_buf, sigma_y);
    ukf_weighted_mean(sigma_y, w0_m, wi, n_sig, no, y_pred);
    ukf_weighted_meas_cov_cross(sigma_z, z_pred, sigma_y, y_pred, w0_c, wi,
                                n_sig, nz, no, S, Pzy);
    for (i64 i = 0; i < no * no; ++i)
      S[i] += in->R[i];
    if (in->symmetrize)
      sdsge_sym_inplace(S, no);

    kf_row_minus_vec(in->obs, t, y_pred, innov, no);
    if (sdsge_chol(S, in->jitter, L, no) != SDSGE_OK) {
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
      ukf_store_history(in, z_pred, P_pred, out->x1_pred, out->x2_pred,
                        out->x_pred, t);
      ukf_store_history(in, z_filt, P_filt, out->x1_filt, out->x2_filt,
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
  free(arena);
  return status;
}
