#include "kalman.h"
#include <math.h>

void kf_zero_mat(f64 *SDSGE_RESTRICT out, i64 r, i64 c) {
    const i64 total = r * c;
    for (i64 i = 0; i < total; ++i)
        out[i] = 0.0;
}

void kf_sym_inplace(f64 *SDSGE_RESTRICT P, i64 n) {
    for (i64 i = 0; i < n; ++i) {
        for (i64 j = i + 1; j < n; ++j) {
            f64 avg = 0.5 * (P[i * n + j] + P[j * n + i]);
            P[i * n + j] = avg;
            P[j * n + i] = avg;
        }
    }
}

void kf_matmul(const f64 *SDSGE_RESTRICT A, const f64 *SDSGE_RESTRICT B,
               f64 *SDSGE_RESTRICT out, i64 n, i64 p, i64 m) {
    for (i64 i = 0; i < n; ++i) {
        for (i64 j = 0; j < m; ++j) {
            f64 s = 0.0;
            for (i64 k = 0; k < p; ++k)
                s += A[i * p + k] * B[k * m + j];
            out[i * m + j] = s;
        }
    }
}

void kf_matmul_abt(const f64 *SDSGE_RESTRICT A, const f64 *SDSGE_RESTRICT B,
                   f64 *SDSGE_RESTRICT out, i64 n, i64 p, i64 m) {
    for (i64 i = 0; i < n; ++i) {
        for (i64 j = 0; j < m; ++j) {
            f64 s = 0.0;
            for (i64 k = 0; k < p; ++k)
                s += A[i * p + k] * B[j * p + k];
            out[i * m + j] = s;
        }
    }
}

void kf_matmul_abt_plus_c(const f64 *SDSGE_RESTRICT A, const f64 *SDSGE_RESTRICT B,
                          const f64 *SDSGE_RESTRICT C, f64 *SDSGE_RESTRICT out,
                          i64 n, i64 p, i64 m) {
    for (i64 i = 0; i < n; ++i) {
        for (i64 j = 0; j < m; ++j) {
            f64 s = 0.0;
            for (i64 k = 0; k < p; ++k)
                s += A[i * p + k] * B[j * p + k];
            out[i * m + j] = s + C[i * m + j];
        }
    }
}

void kf_matvec(const f64 *SDSGE_RESTRICT A, const f64 *SDSGE_RESTRICT x,
               f64 *SDSGE_RESTRICT out, i64 n, i64 m) {
    for (i64 i = 0; i < n; ++i) {
        f64 s = 0.0;
        for (i64 j = 0; j < m; ++j)
            s += A[i * m + j] * x[j];
        out[i] = s;
    }
}

void kf_matvec_plus_vec(const f64 *SDSGE_RESTRICT A, const f64 *SDSGE_RESTRICT x,
                        const f64 *SDSGE_RESTRICT b, f64 *SDSGE_RESTRICT out,
                        i64 n, i64 m) {
    for (i64 i = 0; i < n; ++i) {
        f64 s = b[i];
        for (i64 j = 0; j < m; ++j)
            s += A[i * m + j] * x[j];
        out[i] = s;
    }
}

void kf_row_minus_vec(const f64 *SDSGE_RESTRICT A, i64 row,
                      const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT out, i64 m) {
    const f64 *Arow = A + row * m;
    for (i64 j = 0; j < m; ++j)
        out[j] = Arow[j] - x[j];
}

f64 kf_dot(const f64 *SDSGE_RESTRICT a, const f64 *SDSGE_RESTRICT b, i64 n) {
    f64 s = 0.0;
    for (i64 i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

f64 kf_logdet_from_chol(const f64 *SDSGE_RESTRICT L, i64 n) {
    f64 s = 0.0;
    for (i64 i = 0; i < n; ++i)
        s += log(L[i * n + i]);
    return 2.0 * s;
}

int kf_chol_shifted(const f64 *SDSGE_RESTRICT S, f64 jitter,
                    f64 *SDSGE_RESTRICT L, i64 n) {
    kf_zero_mat(L, n, n);
    for (i64 i = 0; i < n; ++i) {
        for (i64 j = 0; j <= i; ++j) {
            f64 s = S[i * n + j];
            if (i == j && jitter > 0.0)
                s += jitter;
            for (i64 k = 0; k < j; ++k)
                s -= L[i * n + k] * L[j * n + k];
            if (i == j) {
                if (s <= 0.0)
                    return KF_ERR_MATRIX_CONDITION;
                L[i * n + j] = sqrt(s);
            } else {
                L[i * n + j] = s / L[j * n + j];
            }
        }
    }
    return KF_OK;
}

void kf_forward_subst(const f64 *SDSGE_RESTRICT L, const f64 *SDSGE_RESTRICT b,
                      f64 *SDSGE_RESTRICT out, i64 n) {
    for (i64 i = 0; i < n; ++i) {
        f64 s = 0.0;
        for (i64 j = 0; j < i; ++j)
            s += L[i * n + j] * out[j];
        out[i] = (b[i] - s) / L[i * n + i];
    }
}

void kf_backward_subst_chol_t(const f64 *SDSGE_RESTRICT L, const f64 *SDSGE_RESTRICT b,
                              f64 *SDSGE_RESTRICT out, i64 n) {
    for (i64 i = n - 1; i >= 0; --i) {
        f64 s = 0.0;
        for (i64 j = i + 1; j < n; ++j)
            s += L[j * n + i] * out[j];
        out[i] = (b[i] - s) / L[i * n + i];
    }
}

void kf_chol_solve_row(const f64 *SDSGE_RESTRICT L, const f64 *SDSGE_RESTRICT B,
                       i64 row, f64 *SDSGE_RESTRICT fbuf, f64 *SDSGE_RESTRICT bbuf,
                       f64 *SDSGE_RESTRICT out, i64 n) {
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

void kf_predict_cov(const f64 *SDSGE_RESTRICT A, const f64 *SDSGE_RESTRICT P_prev,
                    const f64 *SDSGE_RESTRICT BQBT, f64 *SDSGE_RESTRICT temp_nn,
                    f64 *SDSGE_RESTRICT out, i64 n) {
    kf_matmul(A, P_prev, temp_nn, n, n, n);
    kf_matmul_abt_plus_c(temp_nn, A, BQBT, out, n, n, n);
}

void kf_measurement_cov(const f64 *SDSGE_RESTRICT C, const f64 *SDSGE_RESTRICT P_pred,
                        const f64 *SDSGE_RESTRICT R, f64 *SDSGE_RESTRICT temp_mn,
                        f64 *SDSGE_RESTRICT out, i64 n, i64 m) {
    kf_matmul(C, P_pred, temp_mn, m, n, n);
    kf_matmul_abt_plus_c(temp_mn, C, R, out, m, n, m);
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

void kf_gain_from_pc_t(const f64 *SDSGE_RESTRICT L, const f64 *SDSGE_RESTRICT PCt,
                       f64 *SDSGE_RESTRICT fbuf, f64 *SDSGE_RESTRICT bbuf,
                       f64 *SDSGE_RESTRICT out, i64 n, i64 m) {
    for (i64 row = 0; row < n; ++row)
        kf_chol_solve_row(L, PCt, row, fbuf, bbuf, out, m);
}

void kf_state_update(const f64 *SDSGE_RESTRICT x_pred, const f64 *SDSGE_RESTRICT K,
                     const f64 *SDSGE_RESTRICT v, f64 *SDSGE_RESTRICT out,
                     i64 n, i64 m) {
    for (i64 i = 0; i < n; ++i) {
        f64 s = x_pred[i];
        for (i64 j = 0; j < m; ++j)
            s += K[i * m + j] * v[j];
        out[i] = s;
    }
}

void kf_identity_minus(const f64 *SDSGE_RESTRICT A, f64 *SDSGE_RESTRICT out, i64 n) {
    for (i64 i = 0; i < n; ++i) {
        for (i64 j = 0; j < n; ++j)
            out[i * n + j] = -A[i * n + j];
        out[i * n + i] += 1.0;
    }
}

void kf_joseph_cov(const f64 *SDSGE_RESTRICT K, const f64 *SDSGE_RESTRICT C,
                   const f64 *SDSGE_RESTRICT P_pred, const f64 *SDSGE_RESTRICT R,
                   f64 *SDSGE_RESTRICT KC, f64 *SDSGE_RESTRICT I_minus_KC,
                   f64 *SDSGE_RESTRICT temp_nn, f64 *SDSGE_RESTRICT temp_nm,
                   f64 *SDSGE_RESTRICT out, i64 n, i64 m) {
    kf_matmul(K, C, KC, n, m, n);
    kf_identity_minus(KC, I_minus_KC, n);
    kf_matmul(I_minus_KC, P_pred, temp_nn, n, n, n);
    kf_matmul_abt(temp_nn, I_minus_KC, out, n, n, n);
    kf_matmul(K, R, temp_nm, n, m, m);
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
                   f64 *SDSGE_RESTRICT temp_nk, f64 *SDSGE_RESTRICT out,
                   i64 n, i64 k) {
    kf_matmul(B, Q, temp_nk, n, k, k);
    for (i64 i = 0; i < n; ++i) {
        for (i64 j = 0; j < n; ++j) {
            f64 s = 0.0;
            for (i64 l = 0; l < k; ++l)
                s += temp_nk[i * k + l] * B[j * k + l];
            out[i * n + j] = s;
        }
    }
    kf_sym_inplace(out, n);
}

void kf_build_shock_projection(const f64 *SDSGE_RESTRICT B, const f64 *SDSGE_RESTRICT C,
                               const f64 *SDSGE_RESTRICT Q, f64 *SDSGE_RESTRICT temp_km,
                               f64 *SDSGE_RESTRICT out, i64 n, i64 k, i64 m) {
    for (i64 i = 0; i < k; ++i) {
        for (i64 j = 0; j < m; ++j) {
            f64 s = 0.0;
            for (i64 l = 0; l < n; ++l)
                s += B[l * k + i] * C[j * n + l];
            temp_km[i * m + j] = s;
        }
    }
    kf_matmul(Q, temp_km, out, k, k, m);
}
