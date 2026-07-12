#include "prior_program.h"
#include <math.h>
#include <stdlib.h>

f64 sdsge_softplus_scalar(f64 x) {
  if (x > 0.0) {
    return x + log1p(exp(-x));
  } else {
    return log1p(exp(x));
  }
}

f64 sdsge_log_sigmoid_scalar(f64 x) {
  if (x > 0.0) {
    return -log1p(exp(-x));
  } else {
    return x - log1p(exp(x));
  }
}

f64 sdsge_sigmoid_scalar(f64 x) {
  if (x >= 0.0) {
    return 1.0 / (1.0 + exp(-x));
  } else {
    f64 exp_x = exp(x);
    return exp_x / (1.0 + exp_x);
  }
}

f64 sdsge_std_norm_cdf(f64 x) { return 0.5 * (1.0 + erf(x / SQRT2)); }

f64 sdsge_std_norm_logpdf(f64 x) { return -0.5 * x * x - 0.5 * log(TWO_PI); }

void sdsge_transform_inverse_and_logjac(i64 code, f64 *SDSGE_RESTRICT params,
                                        f64 z, f64 *SDSGE_RESTRICT out_x,
                                        f64 *SDSGE_RESTRICT out_logjac) {

  switch (code) {
  case SDSGE_TRANSFORM_IDENTITY:
    *out_x = z;
    *out_logjac = 0.0;
    break;
  case SDSGE_TRANSFORM_LOG:
    *out_x = exp(z);
    *out_logjac = z;
    break;
  case SDSGE_TRANSFORM_SOFTPLUS:
    *out_x = sdsge_softplus_scalar(z);
    *out_logjac = sdsge_log_sigmoid_scalar(z);
    break;
  case SDSGE_TRANSFORM_LOGIT:
    *out_x = sdsge_sigmoid_scalar(z);
    *out_logjac = sdsge_log_sigmoid_scalar(z) + sdsge_log_sigmoid_scalar(-z);
    break;
  case SDSGE_TRANSFORM_PROBIT:
    *out_x = sdsge_std_norm_cdf(z);
    *out_logjac = sdsge_std_norm_logpdf(z);
    break;
  case SDSGE_TRANSFORM_AFFINE_LOGIT: {
    f64 sig = sdsge_sigmoid_scalar(z);
    *out_x = params[0] + (params[2] * sig);
    *out_logjac = log(params[2]) + sdsge_log_sigmoid_scalar(z) +
                  sdsge_log_sigmoid_scalar(-z);
    break;
  }
  case SDSGE_TRANSFORM_AFFINE_PROBIT:
    *out_x = params[0] + (params[2] * sdsge_std_norm_cdf(z));
    *out_logjac = log(params[2]) + sdsge_std_norm_logpdf(z);
    break;
  case SDSGE_TRANSFORM_LOWER_BOUNDED:
    *out_x = params[0] + exp(z);
    *out_logjac = z;
    break;
  case SDSGE_TRANSFORM_UPPER_BOUNDED:
    *out_x = params[0] - exp(z);
    *out_logjac = z;
    break;
  default:
    *out_x = NAN;
    *out_logjac = NAN;
    break;
  }
}

void sdsge_dist_logpdf(i64 code, f64 *SDSGE_RESTRICT params, f64 x,
                       f64 *SDSGE_RESTRICT out_logpdf) {
  switch (code) {
  case SDSGE_DIST_NORMAL:
    *out_logpdf = -0.5 * log(TWO_PI * params[1]) -
                  0.5 * ((x - params[0]) * (x - params[0])) / params[1];
    break;
  case SDSGE_DIST_LOG_NORMAL:
    if (x <= 0.0) {
      *out_logpdf = NAN;
    } else {
      f64 log_x = log(x);
      *out_logpdf = -log(params[1]) - log_x - 0.5 * log(TWO_PI) -
                    0.5 * ((log_x - params[0]) / params[1]) *
                        ((log_x - params[0]) / params[1]);
    }
    break;
  case SDSGE_DIST_HALF_NORMAL:
    if (x < 0.0) {
      *out_logpdf = NAN;
    } else {
      *out_logpdf = 0.5 * log(2.0 / PI) - log(params[0]) -
                    0.5 * (x / params[0]) * (x / params[0]);
    }
    break;
  case SDSGE_DIST_TRUNC_NORMAL:
    if (x < params[2] || x > params[3]) {
      *out_logpdf = NAN;
    } else {
      f64 z = (x - params[0]) / params[1];
      *out_logpdf = -0.5 * z * z - params[4];
    }
    break;
  case SDSGE_DIST_HALF_CAUCHY:
    if (x < 0.0) {
      *out_logpdf = NAN;
    } else {
      f64 centered = x / params[0];
      *out_logpdf = log(2.0 / PI) - log(params[0]) - log1p(centered * centered);
    }
    break;
  case SDSGE_DIST_BETA:
    if (x < 0.0 || x > 1.0) {
      *out_logpdf = NAN;
    } else {
      *out_logpdf = 0.0;
      if (params[0] != 1.0) {
        *out_logpdf += (params[0] - 1.0) * log(x);
      }
      if (params[1] != 1.0) {
        *out_logpdf += (params[1] - 1.0) * log1p(-x);
      }
      *out_logpdf -= params[2];
    }
    break;
  case SDSGE_DIST_GAMMA:
    if (x < 0.0) {
      *out_logpdf = NAN;
    } else {
      f64 tmp = 0.0;
      if (params[0] != 1.0) {
        tmp += (params[0] - 1.0) * log(x);
      }
      *out_logpdf = tmp - x / params[1] - params[2];
    }
    break;
  case SDSGE_DIST_INV_GAMMA:
    if (x <= 0.0) {
      *out_logpdf = NAN;
    } else {
      *out_logpdf = params[2] - (params[0] + 1.0) * log(x) - params[1] / x;
    }
    break;
  case SDSGE_DIST_UNIFORM:
    if (x < params[0] || x > params[1]) {
      *out_logpdf = NAN;
    } else {
      *out_logpdf = -log(params[2]);
    }
    break;
  default:
    *out_logpdf = NAN;
    break;
  }
}

f64 sdsge_lkj_chol_logjac_return(f64 *SDSGE_RESTRICT z, i64 dim, i64 len) {
  f64 logjac = 0.0;
  i64 idx = 0;
  for (i64 k = 1; k < dim; ++k) {
    f64 rem = 1.0;
    for (i64 j = 0; j < k; ++j) {
      if (idx >= len) {
        return NAN; // Out of bounds
      }
      f64 cpc_i = tanh(z[idx]);
      logjac += 0.5 * log(max_f64(rem, 1e-300)); // Avoid log(0)
      logjac += log1p(-(cpc_i * cpc_i));
      rem *= (1.0 - cpc_i * cpc_i);
      idx++;
    }
  }
  return logjac;
}

void sdsge_lkj_chol_logjac(f64 *SDSGE_RESTRICT z, i64 dim, i64 len,
                           f64 *SDSGE_RESTRICT out_logjac) {
  *out_logjac = sdsge_lkj_chol_logjac_return(z, dim, len);
}

void sdsge_lkj_chol_logpdf_from_z(f64 *SDSGE_RESTRICT z, i64 dim, i64 len,
                                  f64 eta, f64 log_const,
                                  f64 *SDSGE_RESTRICT out_logpdf) {
  f64 log_kernel = 0.0;
  i64 idx = 0;
  for (i64 i = 1; i < dim; ++i) {
    f64 rem = 1.0;
    for (i64 j = 0; j < i; ++j) {
      if (idx >= len) {
        *out_logpdf = NAN; // Out of bounds
        return;
      }
      f64 cpc_i = tanh(z[idx]);
      rem *= (1.0 - cpc_i * cpc_i);
      idx++;
    }
    log_kernel +=
        ((f64)dim - i + 2.0 * eta - 3.0) * log(sqrt(max_f64(rem, 1e-14)));
  }
  *out_logpdf =
      log_const + log_kernel + sdsge_lkj_chol_logjac_return(z, dim, len);
}

/* Packed log-prior driver: the per-replication hot path. Mirrors the numba
 * _evaluate_logprior_program -- sums the scalar terms (inverse-transform z -> x,
 * then dist logpdf + transform log-jacobian) and the LKJ matrix blocks, and
 * short-circuits to NaN the moment any term is NaN. */
f64 sdsge_logprior_program(
    f64 *SDSGE_RESTRICT theta, i64 *SDSGE_RESTRICT scalar_indices,
    i64 *SDSGE_RESTRICT scalar_dist_codes,
    i64 *SDSGE_RESTRICT scalar_transform_codes,
    f64 *SDSGE_RESTRICT scalar_dist_params,
    f64 *SDSGE_RESTRICT scalar_transform_params, i64 n_scalar,
    i64 *SDSGE_RESTRICT matrix_indices, i64 *SDSGE_RESTRICT matrix_dims,
    i64 *SDSGE_RESTRICT matrix_lengths, f64 *SDSGE_RESTRICT matrix_etas,
    f64 *SDSGE_RESTRICT matrix_log_constants, i64 n_blocks, i64 max_matrix_len) {
  f64 lp = 0.0;

  for (i64 i = 0; i < n_scalar; ++i) {
    f64 z = theta[scalar_indices[i]];
    f64 x, logjac;
    sdsge_transform_inverse_and_logjac(
        scalar_transform_codes[i],
        scalar_transform_params + i * SDSGE_N_TRANSFORM_PARAMS, z, &x, &logjac);
    if (isnan(x) || isnan(logjac)) {
      return NAN;
    }
    f64 logp;
    sdsge_dist_logpdf(scalar_dist_codes[i],
                      scalar_dist_params + i * SDSGE_N_DIST_PARAMS, x, &logp);
    if (isnan(logp)) {
      return NAN;
    }
    lp += logp + logjac;
  }

  if (n_blocks > 0) {
    f64 *z_block = (f64 *)malloc((size_t)max_matrix_len * sizeof(f64));
    if (z_block == NULL) {
      return NAN;
    }
    for (i64 b = 0; b < n_blocks; ++b) {
      i64 length = matrix_lengths[b];
      for (i64 j = 0; j < length; ++j) {
        z_block[j] = theta[matrix_indices[b * max_matrix_len + j]];
      }
      f64 block_lp;
      sdsge_lkj_chol_logpdf_from_z(z_block, matrix_dims[b], length,
                                   matrix_etas[b], matrix_log_constants[b],
                                   &block_lp);
      if (isnan(block_lp)) {
        free(z_block);
        return NAN;
      }
      lp += block_lp;
    }
    free(z_block);
  }

  return lp;
}

/* Unconstrained (z, std) -> full covariance. Builds the correlation Cholesky
 * factor row by row from the unconstrained CPC values z (tanh + stick-breaking
 * remainder), then forms cov = diag(std) (L L^T) diag(std). ``scratch_M`` holds
 * L (K*K, row-major); ``out`` receives the K*K covariance (row-major). */
void sdsge_cov_from_unconstrained(const f64 *SDSGE_RESTRICT z,
                                  const f64 *SDSGE_RESTRICT std, const i64 K,
                                  f64 *SDSGE_RESTRICT L,
                                  f64 *SDSGE_RESTRICT out) {
  i64 idx = 0;
  for (i64 i = 0; i < K; ++i) {
    const i64 ri = i * K;
    const f64 si = std[i];

    f64 rem = 1.0;
    for (i64 j = 0; j < i; ++j) {
      const f64 v = sqrt(max_f64(1e-14, rem)) * tanh(z[idx++]);
      L[ri + j] = v;
      rem -= v * v;
    }
    L[ri + i] = sqrt(max_f64(1e-14, rem));

    for (i64 j = 0; j < i; ++j) {
      const i64 rj = j * K;
      f64 s = 0.0;
      for (i64 c = 0; c <= j; ++c)
        s += L[ri + c] * L[rj + c];
      const f64 v = si * std[j] * s;
      out[ri + j] = v;
      out[rj + i] = v;
    }
    out[ri + i] = si * si;
  }
}

/* Inverse of the Cholesky stage of sdsge_cov_from_unconstrained: correlation
 * Cholesky factor L (K*K, row-major) -> unconstrained CPC values out_z
 * (length K(K-1)/2). Recovers each partial correlation as L[k,j] / sqrt(rem),
 * clamps to the open unit interval, and applies atanh. */
void sdsge_unconstrained_from_corr_chol(const f64 *SDSGE_RESTRICT L, const i64 K,
                                        f64 *SDSGE_RESTRICT out_z) {
  i64 idx = 0;
  for (i64 k = 1; k < K; ++k) {
    const i64 rk = k * K;
    f64 rem = 1.0;
    for (i64 j = 0; j < k; ++j) {
      const f64 v = sqrt(max_f64(1e-14, rem));
      f64 cpc = L[rk + j] / v;
      if (cpc < -1.0 + 1e-14)
        cpc = -1.0 + 1e-14;
      else if (cpc > 1.0 - 1e-14)
        cpc = 1.0 - 1e-14;
      out_z[idx++] = atanh(cpc);
      rem -= L[rk + j] * L[rk + j];
    }
  }
}
