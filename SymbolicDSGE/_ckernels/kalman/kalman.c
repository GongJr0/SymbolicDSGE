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

void kf_predict_cov(const f64 *SDSGE_RESTRICT A,
                    const f64 *SDSGE_RESTRICT P_prev,
                    const f64 *SDSGE_RESTRICT BQBT, f64 *SDSGE_RESTRICT temp_nn,
                    f64 *SDSGE_RESTRICT out, i64 n) {
  sdsge_matmul(A, P_prev, temp_nn, n, n, n);
  sdsge_matmul_abt_plus_c(temp_nn, A, BQBT, out, n, n, n);
}

void kf_measurement_cov(const f64 *SDSGE_RESTRICT C,
                        const f64 *SDSGE_RESTRICT PCt,
                        const f64 *SDSGE_RESTRICT R, f64 *SDSGE_RESTRICT out,
                        i64 n, i64 m) {
  sdsge_matmul_plus_c(C, PCt, R, out, m, n, m);
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

void kf_whiten_pc_t(const f64 *SDSGE_RESTRICT L, f64 *SDSGE_RESTRICT PCt, i64 n,
                    i64 m) {
  for (i64 i = 0; i < n; ++i) {
    sdsge_forward_subst(L, PCt + i * m, PCt + i * m, m);
  }
}

void kf_gain_from_whitened(const f64 *SDSGE_RESTRICT L,
                           const f64 *SDSGE_RESTRICT W, f64 *SDSGE_RESTRICT out,
                           i64 n, i64 m) {
  for (i64 i = 0; i < n; ++i) {
    sdsge_backward_subst_chol_t(L, W + i * m, out + i * m, m);
  }
}

void kf_whiten_meas(const f64 *SDSGE_RESTRICT L, const f64 *SDSGE_RESTRICT C,
                    f64 *SDSGE_RESTRICT out, i64 m, i64 n) {
  for (i64 i = 0; i < m; ++i) {
    f64 *Gi = out + i * n;
    const f64 *Ci = C + i * n;
    for (i64 c = 0; c < n; ++c) {
      Gi[c] = Ci[c];
    }

    for (i64 j = 0; j < i; ++j) {
      const f64 l = L[i * m + j];
      const f64 *Gj = out + j * n;
      for (i64 c = 0; c < n; ++c) {
        Gi[c] -= l * Gj[c];
      }
    }

    const f64 d = L[i * m + i];
    for (i64 c = 0; c < n; ++c) {
      Gi[c] /= d;
    }
  }
}

void kf_cov_downdate(const f64 *SDSGE_RESTRICT P_pred,
                     const f64 *SDSGE_RESTRICT W, f64 *SDSGE_RESTRICT out,
                     i64 n, i64 m) {
  for (i64 i = 0; i < n; ++i) {
    const f64 *Wi = W + i * m;
    for (i64 j = 0; j <= i; ++j) {
      const f64 *Wj = W + j * m;

      f64 s = 0.0;
      for (i64 k = 0; k < m; ++k) {
        s += Wi[k] * Wj[k];
      }

      out[i * n + j] = P_pred[i * n + j] - s;
      out[j * n + i] = P_pred[j * n + i] - s;
    }
  }
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

arena_size kf_stationary_covariance_arena_size(const i64 n, const i64 k) {
  return make_sizer(4 * n * n + n * k, 0);
}

int kf_stationary_covariance(const f64 *SDSGE_RESTRICT A,
                             const f64 *SDSGE_RESTRICT B,
                             const f64 *SDSGE_RESTRICT Q, f64 tol, i64 max_iter,
                             f64 *SDSGE_RESTRICT arena, f64 *SDSGE_RESTRICT out,
                             i64 n, i64 k, i64 ld_out) {
  f64 *BQBT = arena;
  f64 *A_power = BQBT + n * n;
  f64 *temp_nn = A_power + n * n;
  f64 *residual = temp_nn + n * n;
  f64 *temp_nk = residual + n * n;
  /* The iteration runs packed in the caller's own buffer: an n*n block always
   * fits inside n rows of stride ld_out >= n. It is expanded in place at the
   * end rather than staged through scratch. */
  f64 *P = out;

  kf_build_bqbt(B, Q, temp_nk, BQBT, n, k);
  memcpy(P, BQBT, (size_t)(n * n) * sizeof(f64));
  memcpy(A_power, A, (size_t)(n * n) * sizeof(f64));

  for (i64 iter = 0; iter < max_iter; ++iter) {
    sdsge_matmul(A_power, P, temp_nn, n, n, n);
    sdsge_matmul_abt(temp_nn, A_power, residual, n, n, n);

    f64 correction_norm = 0.0;
    f64 covariance_norm = 0.0;
    for (i64 i = 0; i < n * n; ++i) {
      if (!isfinite(residual[i]))
        return KF_ERR_LYAPUNOV_CONVERGENCE;
      const f64 correction = fabs(residual[i]);
      if (correction > correction_norm)
        correction_norm = correction;
      P[i] += residual[i];
      if (!isfinite(P[i]))
        return KF_ERR_LYAPUNOV_CONVERGENCE;
      const f64 covariance = fabs(P[i]);
      if (covariance > covariance_norm)
        covariance_norm = covariance;
    }
    sdsge_sym_inplace(P, n);

    if (!isfinite(correction_norm) || !isfinite(covariance_norm))
      return KF_ERR_LYAPUNOV_CONVERGENCE;

    if (correction_norm <= tol * fmax(1.0, covariance_norm)) {
      sdsge_matmul(A, P, temp_nn, n, n, n);
      sdsge_matmul_abt(temp_nn, A, residual, n, n, n);

      f64 residual_norm = 0.0;
      for (i64 i = 0; i < n * n; ++i) {
        const f64 difference = P[i] - residual[i] - BQBT[i];
        if (!isfinite(difference))
          return KF_ERR_LYAPUNOV_CONVERGENCE;
        const f64 entry = fabs(difference);
        if (entry > residual_norm)
          residual_norm = entry;
      }
      if (isfinite(residual_norm) &&
          residual_norm <= tol * fmax(1.0, covariance_norm)) {
        /* Expand the packed result into the stride, last row first: row r
         * lands at r*ld_out, at or past the r*n it came from, so descending
         * order never overwrites a row still waiting to move. Each row's tail
         * is zeroed behind it, which is what makes the unscented P0's
         * second-order quadrant zero rather than iteration debris. */
        for (i64 r = n - 1; r >= 0; --r) {
          memmove(out + r * ld_out, P + r * n, (size_t)n * sizeof(f64));
          for (i64 c = n; c < ld_out; ++c) {
            out[r * ld_out + c] = 0.0;
          }
        }
        return KF_OK;
      }
    }

    sdsge_matmul(A_power, A_power, temp_nn, n, n, n);
    memcpy(A_power, temp_nn, (size_t)(n * n) * sizeof(f64));
  }

  return KF_ERR_LYAPUNOV_CONVERGENCE;
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
