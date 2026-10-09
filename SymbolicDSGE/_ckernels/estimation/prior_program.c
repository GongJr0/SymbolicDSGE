#include "prior_program.h"
#include <math.h>

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

/* log(sech^2(y)) = log(1 - tanh^2(y)); mirrors sdsge_log_sech2 in transforms.c
 * so the TANH scatter is bit-identical to the low-level tanh kernel the Python
 * path uses. Avoids the 1 - tanh^2 cancellation and the cosh overflow. */
static inline f64 sdsge_log_sech2(f64 y) {
  f64 ay = fabs(y);
  return 2.0 * (0.6931471805599453 - ay - log1p(exp(-2.0 * ay)));
}

f64 sdsge_std_norm_cdf(f64 x) { return 0.5 * (1.0 + erf(x / SQRT2)); }

f64 sdsge_std_norm_logpdf(f64 x) { return -0.5 * x * x - 0.5 * log(TWO_PI); }

void sdsge_transform_inverse_and_logjac(i64 code,
                                        const f64 *SDSGE_RESTRICT params, f64 z,
                                        f64 *SDSGE_RESTRICT out_x,
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
  case SDSGE_TRANSFORM_TANH:
    *out_x = tanh(z);
    *out_logjac = sdsge_log_sech2(z);
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
 * _evaluate_logprior_program -- sums the scalar terms (inverse-transform z ->
 * x, then dist logpdf + transform log-jacobian) and the LKJ matrix blocks, and
 * short-circuits to NaN the moment any term is NaN. Each block's unconstrained
 * z occupies a contiguous run theta[offset .. offset+length), so the block is
 * read straight off theta by base-pointer offset (no gather, no scratch). */
f64 sdsge_logprior_program(f64 *SDSGE_RESTRICT theta,
                           const sdsge_prior_tables *pr) {
  f64 lp = 0.0;
  f64 z, x, logp, logjac;

  for (i64 i = 0; i < pr->n_theta; ++i) {

    /* A run's density is evaluated once, at the slot its row starts on. */
    const i64 len = sdsge_block_run_len(pr, i);
    if (len) {
      const f64 *row = pr->dist_params + i * SDSGE_N_DIST_PARAMS;
      const i64 K = (i64)row[0];
      const f64 eta = row[1];
      const f64 log_const = row[2];
      sdsge_lkj_chol_logpdf_from_z(theta + i, K, len, eta, log_const, &logp);
      if (isnan(logp)) {
        return NAN;
      }
      lp += pr->include_logjac
                ? logp
                : logp - sdsge_lkj_chol_logjac_return(theta + i, K, len);

      i += len - 1;
      continue;
    }

    z = theta[i];
    sdsge_transform_inverse_and_logjac(
        pr->transform_codes[i],
        pr->transform_params + i * SDSGE_N_TRANSFORM_PARAMS, z, &x, &logjac);
    if (isnan(x) || isnan(logjac)) {
      return NAN;
    }
    sdsge_dist_logpdf(pr->dist_codes[i],
                      (f64 *)pr->dist_params + i * SDSGE_N_DIST_PARAMS, x,
                      &logp);
    if (isnan(logp)) {
      return NAN;
    }
    lp += pr->include_logjac ? logp + logjac : logp;
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

/* Unconstrained -> the block's packed correlation entries, with the LKJ
 * log-jacobian taken from the same sweep. The Cholesky recursion is
 * sdsge_cov_from_unconstrained's at std == 1, where the covariance it would
 * form is the correlation itself, so only the K(K-1)/2 lower-triangle entries
 * are written and no K*K output is needed. The jacobian rides the
 * multiplicative remainder sdsge_lkj_chol_logjac_return carries, separate from
 * the subtractive one the factor needs, so the two kernels answer alike.
 * ``L`` is K*K scratch; ``out`` receives the entries in the (row, col) order
 * the block's theta run carries. */
void sdsge_corr_entries_from_unconstrained(const f64 *SDSGE_RESTRICT z,
                                           const i64 K, f64 *SDSGE_RESTRICT L,
                                           f64 *SDSGE_RESTRICT out,
                                           f64 *SDSGE_RESTRICT out_logjac) {
  f64 logjac = 0.0;
  i64 idx = 0;
  i64 k = 0;
  for (i64 i = 0; i < K; ++i) {
    const i64 ri = i * K;

    f64 rem = 1.0;
    f64 remj = 1.0;
    for (i64 j = 0; j < i; ++j) {
      const f64 cpc = tanh(z[idx++]);
      const f64 v = sqrt(max_f64(1e-14, rem)) * cpc;
      L[ri + j] = v;
      rem -= v * v;
      logjac += 0.5 * log(max_f64(remj, 1e-300));
      logjac += log1p(-(cpc * cpc));
      remj *= (1.0 - cpc * cpc);
    }
    L[ri + i] = sqrt(max_f64(1e-14, rem));

    for (i64 j = 0; j < i; ++j) {
      const i64 rj = j * K;
      f64 s = 0.0;
      for (i64 c = 0; c <= j; ++c)
        s += L[ri + c] * L[rj + c];
      out[k++] = s;
    }
  }
  *out_logjac = logjac;
}

/* Inverse of the Cholesky stage of sdsge_cov_from_unconstrained: correlation
 * Cholesky factor L (K*K, row-major) -> unconstrained CPC values out_z
 * (length K(K-1)/2). Recovers each partial correlation as L[k,j] / sqrt(rem),
 * clamps to the open unit interval, and applies atanh. */
void sdsge_unconstrained_from_corr_chol(const f64 *SDSGE_RESTRICT L,
                                        const i64 K,
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
