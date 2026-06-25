#include "kalman.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void kf_row_minus_vec(const f64 *SDSGE_RESTRICT A, i64 row,
                      const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT out, i64 m) {
    const f64 *Arow = A + row * m;
    for (i64 j = 0; j < m; ++j)
        out[j] = Arow[j] - x[j];
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
    sdsge_matmul(A, P_prev, temp_nn, n, n, n);
    sdsge_matmul_abt_plus_c(temp_nn, A, BQBT, out, n, n, n);
}

void kf_measurement_cov(const f64 *SDSGE_RESTRICT C, const f64 *SDSGE_RESTRICT P_pred,
                        const f64 *SDSGE_RESTRICT R, f64 *SDSGE_RESTRICT temp_mn,
                        f64 *SDSGE_RESTRICT out, i64 n, i64 m) {
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
                   f64 *SDSGE_RESTRICT temp_nk, f64 *SDSGE_RESTRICT out,
                   i64 n, i64 k) {
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
    sdsge_matmul(Q, temp_km, out, k, k, m);
}

int kf_hot_loop(const kf_inputs *in, kf_outputs *out) {
    const i64 n = in->n, m = in->m, k = in->k, T = in->T;

    /* One scratch arena carved into every per-step buffer (allocated once). */
    const i64 total = 2 * n + 7 * m   /* vectors + triangular-solve scratch */
                      + 6 * n * n     /* P_pred, P_filt, KC, I_minus_KC, temp_nn, BQBT */
                      + 2 * m * m     /* S_buf, L */
                      + 3 * n * m     /* PCt, K, temp_nm */
                      + m * n         /* temp_mn */
                      + n * k         /* temp_nk */
                      + 2 * k * m;    /* M, temp_km */
    f64 *arena = (f64 *)malloc((size_t)total * sizeof(f64));
    if (arena == NULL)
        return KF_ERR_ALLOC;

    f64 *p = arena;
    f64 *x_pred_buf = p; p += n;
    f64 *x_filt_buf = p; p += n;
    f64 *y_pred_buf = p; p += m;
    f64 *y_filt_buf = p; p += m;
    f64 *v_buf = p; p += m;
    f64 *u_buf = p; p += m;
    f64 *S_inv_v = p; p += m;
    f64 *solve_f = p; p += m;
    f64 *solve_b = p; p += m;
    f64 *P_pred_buf = p; p += n * n;
    f64 *P_filt_buf = p; p += n * n;
    f64 *KC = p; p += n * n;
    f64 *I_minus_KC = p; p += n * n;
    f64 *temp_nn = p; p += n * n;
    f64 *BQBT = p; p += n * n;
    f64 *S_buf = p; p += m * m;
    f64 *L = p; p += m * m;
    f64 *PCt = p; p += n * m;
    f64 *K = p; p += n * m;
    f64 *temp_nm = p; p += n * m;
    f64 *temp_mn = p; p += m * n;
    f64 *temp_nk = p; p += n * k;
    f64 *M = p; p += k * m;
    f64 *temp_km = p; p += k * m;

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
        kf_joseph_cov(K, in->C, P_pred_buf, in->R, KC, I_minus_KC, temp_nn,
                      temp_nm, P_filt_buf, n, m);
        if (in->symmetrize)
            sdsge_sym_inplace(P_filt_buf, n);

        loglik += -0.5 * (const_term + sdsge_logdet_from_chol(L, m)
                          + sdsge_dot(v_buf, S_inv_v, m));

        if (in->return_shocks && in->store_history)
            sdsge_matvec(M, S_inv_v, out->eps_hat + t * k, k, m);

        if (in->store_history) {
            sdsge_matvec_plus_vec(in->C, x_filt_buf, in->d, y_filt_buf, m, n);
            memcpy(out->x_pred + t * n, x_pred_buf, (size_t)n * sizeof(f64));
            memcpy(out->x_filt + t * n, x_filt_buf, (size_t)n * sizeof(f64));
            memcpy(out->P_pred + t * n * n, P_pred_buf, (size_t)(n * n) * sizeof(f64));
            memcpy(out->P_filt + t * n * n, P_filt_buf, (size_t)(n * n) * sizeof(f64));
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
