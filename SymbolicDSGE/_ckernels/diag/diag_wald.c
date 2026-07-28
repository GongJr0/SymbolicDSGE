#include "diag_wald.h"
#include "diag.h"
#include <math.h>
#include <string.h>

/* Column mean/var of a row-major (n, p) buffer (not on the Python side). */
static f64 col_mean(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 p,
                    const i64 col) {
  f64 mean = 0.0;
  for (i64 i = 0; i < n; ++i) {
    mean += x[i * p + col];
  }

  return mean / (f64)n;
}

static f64 col_var(const f64 *SDSGE_RESTRICT x, f64 mean, const i64 n,
                   const i64 p, const i64 col) {
  f64 var = 0.0;
  for (i64 i = 0; i < n; ++i) {
    f64 diff = x[i * p + col] - mean;
    var += diff * diff;
  }
  return var / (f64)n;
}

// --- moment_calculation_utils ---

void sdsge_fill_mean_ax0(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 p,
                         f64 *SDSGE_RESTRICT mean) {

  for (i64 i = 0; i < p; ++i) {
    mean[i] = 0.0;
  }

  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < p; ++j) {
      mean[j] += x[i * p + j];
    }
  }
  for (i64 i = 0; i < p; ++i) {
    mean[i] /= (f64)n;
  }
}

void sdsge_fill_centered_ax0(const f64 *SDSGE_RESTRICT x,
                             const f64 *SDSGE_RESTRICT mean, const i64 n,
                             const i64 p, f64 *SDSGE_RESTRICT centered) {
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < p; ++j) {
      centered[i * p + j] = x[i * p + j] - mean[j];
    }
  }
}

// -------

// --- hac_covariance ---

i64 wooldridge_bandwidth(const f64 *SDSGE_RESTRICT
                             x, /* kept for signature parity with Python */
                         const i64 n) {
  (void)x;
  return (i64)floor(4.0 * pow((f64)n / 100.0, 2.0 / 9.0));
}

/* Andrews (1991) AR(1)-plug-in bandwidth for the strided series y[0],
 * y[stride],
 * ..., y[(n-1)*stride]. The stride lets the matrix variant walk a column of a
 * row-major (n, p) buffer in place -- no per-column copy. Returns 1 for any
 * degenerate case (n < 2, ~zero variance, non-finite moments, non-positive
 * Rhat), matching the numba reference's guards. */
static i64 andrews_bw_strided(const f64 *SDSGE_RESTRICT y, const i64 n,
                              const i64 stride, const f64 c, const f64 q) {
  if (n < 2)
    return 1;

  f64 mean = 0.0;
  for (i64 i = 0; i < n; ++i)
    mean += y[i * stride];
  mean /= (f64)n;

  f64 var = 0.0;
  for (i64 i = 0; i < n; ++i) {
    f64 d = y[i * stride] - mean;
    var += d * d;
  }
  var /= (f64)n;
  if (var <= 1e-14)
    return 1;

  /* Uncentered AR(1) coefficient: beta = dot(y_lag, y_cur) / dot(y_lag, y_lag),
   * with y_lag = y[:-1] and y_cur = y[1:] -- a pure index offset, no buffers.
   */
  f64 denom = 0.0, numer = 0.0;
  for (i64 i = 1; i < n; ++i) {
    f64 prev = y[(i - 1) * stride];
    denom += prev * prev;
    numer += prev * y[i * stride];
  }
  if (!isfinite(denom) || !isfinite(numer) || denom == 0.0)
    return 1;

  f64 beta = numer / denom;
  beta = max_f64(-0.999, min_f64(0.999, beta)); /* clip to avoid Rhat blowup */

  f64 rhat = 2.0 * beta * (1.0 + beta) / ((1.0 - beta) * (1.0 - beta));
  if (rhat <= 0.0 || !isfinite(rhat))
    return 1;

  const f64 expo = 1.0 / (2.0 * q + 1.0);
  return max_i64(1, (i64)floor(c * pow(rhat, expo) * pow((f64)n, expo)));
}

i64 andrews_bandwidth(const f64 *SDSGE_RESTRICT y, KernelID kernel_id,
                      const i64 n) {
  return andrews_bw_strided(y, n, 1, KERNEL_SPECS[kernel_id].c,
                            KERNEL_SPECS[kernel_id].q);
}

/* Median of the per-column Andrews bandwidths, excluding near-constant columns
 * (var <= 1e-14) from the set -- the numba reference drops them before taking
 * the median. `ls` is caller-owned scratch of length >= p; it is overwritten
 * and reordered (no allocation here -- slice it from the entry-point arena). */
i64 andrews_bandwidth_matrix(const f64 *SDSGE_RESTRICT r, KernelID kernel_id,
                             const i64 n, const i64 p, f64 *SDSGE_RESTRICT ls) {
  if (p == 1)
    return andrews_bandwidth(r, kernel_id, n);

  const f64 c = KERNEL_SPECS[kernel_id].c;
  const f64 q = KERNEL_SPECS[kernel_id].q;

  i64 m = 0;
  for (i64 j = 0; j < p; ++j) {
    f64 mean = col_mean(r, n, p, j);
    if (col_var(r, mean, n, p, j) > 1e-14)
      ls[m++] = (f64)andrews_bw_strided(r + j, n, p, c, q);
  }
  if (m == 0)
    return 1;
  return (i64)floor(sdsge_median_f64(ls, m));
}

static int wald_resolve_bandwidth(const f64 *SDSGE_RESTRICT r,
                                  const KernelID kernel_id,
                                  const WaldBandwidthMode bandwidth_mode,
                                  const i64 manual_bandwidth, const i64 n,
                                  const i64 p,
                                  f64 *SDSGE_RESTRICT bandwidth_scratch,
                                  i64 *SDSGE_RESTRICT out) {
  if (kernel_id < BARTLETT || kernel_id >= KERNEL_COUNT || n < 2)
    return (kernel_id < BARTLETT || kernel_id >= KERNEL_COUNT)
               ? DIAG_BAD_PARAMETER
               : DIAG_INSUFFICIENT_SAMPLES;

  i64 bandwidth = 0;
  switch (bandwidth_mode) {
  case WALD_BW_MANUAL:
    if (manual_bandwidth < 0)
      return DIAG_BAD_PARAMETER;
    bandwidth = manual_bandwidth;
    break;
  case WALD_BW_WOOLDRIDGE:
    bandwidth = wooldridge_bandwidth(r, n);
    break;
  case WALD_BW_ANDREWS:
    bandwidth = andrews_bandwidth_matrix(r, kernel_id, n, p, bandwidth_scratch);
    break;
  case WALD_BW_AUTO:
    bandwidth =
        (kernel_id == BARTLETT)
            ? wooldridge_bandwidth(r, n)
            : andrews_bandwidth_matrix(r, kernel_id, n, p, bandwidth_scratch);
    break;
  default:
    return DIAG_BAD_PARAMETER;
  }
  *out = min_i64(bandwidth, n - 1);
  return DIAG_OK;
}

static void center_inplace(f64 *SDSGE_RESTRICT x,
                           const f64 *SDSGE_RESTRICT mean, const i64 n,
                           const i64 p) {
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < p; ++j)
      x[i * p + j] -= mean[j];
  }
}

// kernel weight function

f64 kernel_weight(i64 j, i64 L, KernelID kernel_id) {
  f64 x = (f64)j / (f64)(L + 1);

  switch (kernel_id) {
  case BARTLETT:
    return (j <= L) ? (1.0 - x) : 0.0;
  case PARZEN:
    if (x > 1.0)
      return 0.0;
    if (x <= 0.5)
      return 1.0 - 6.0 * (x * x) + 6.0 * (x * x * x);
    return 2.0 * (1.0 - x) * (1.0 - x) * (1.0 - x);
  case QS:
    if (fabs(x) <= 1e-8)
      return 1.0; // Handle the case when x is very close to 0

    f64 outer = 25.0 / (12.0 * PI * PI * x * x);
    f64 arg = 6.0 * PI * x / 5.0;
    return outer * (sin(arg) / arg - cos(arg));
  default:
    return 0.0; // Unknown kernel ID
  }
}

void sdsge_hac_estimator_matmul(f64 *SDSGE_RESTRICT r, KernelID kernel_id,
                                i64 L, i64 n, i64 p,
                                f64 *SDSGE_RESTRICT gamma_scratch,
                                f64 *SDSGE_RESTRICT out) {
  /* Gamma_0 = r^T r (full symmetric); the lag terms accumulate on top. */
  sdsge_gram(r, out, n, p);

  L = min_i64(L, n - 1);
  for (i64 j = 1; j <= L; ++j) {
    f64 w_j = kernel_weight(j, L, kernel_id);

    if (w_j == 0.0) {
      continue;
    }

    /* Gamma_j = r[:-j]^T @ r[j:] -- lagged views of r, no copy. */
    sdsge_matmul_atb(r, r + j * p, gamma_scratch, n - j, p, p);

    // out += w_j * (Gamma_j + Gamma_j')
    for (i64 k = 0; k < p; ++k) {
      for (i64 l = 0; l < p; ++l) {
        out[k * p + l] +=
            w_j * (gamma_scratch[k * p + l] + gamma_scratch[l * p + k]);
      }
    }
  }

  /* The numba reference divides every autocovariance by n; do it once over the
   * assembled sum -- identical up to rounding, well within parity tolerance. */
  for (i64 i = 0; i < p * p; ++i) {
    out[i] /= (f64)n;
  }
}

// ------
// --- wald_test ---

int sdsge_wald_stat_from_mean_and_cov(
    const f64 *SDSGE_RESTRICT mean, const f64 *SDSGE_RESTRICT target,
    const f64 *SDSGE_RESTRICT omega, const i64 n, const i64 p,
    f64 *SDSGE_RESTRICT dev_scratch, f64 *SDSGE_RESTRICT factor_scratch,
    i64 *SDSGE_RESTRICT pivot_scratch, f64 *SDSGE_RESTRICT solved_scratch,
    f64 *SDSGE_RESTRICT stat_out) {
  /* Compute the Wald statistic: *
   * dev = mean - target;
   * stat = n * (dev^T @ omega^-1 @ dev); */
  *stat_out = NAN;
  for (i64 i = 0; i < p; ++i) {
    dev_scratch[i] = mean[i] - target[i];
  }
  int code = sdsge_chol(omega, 0.0, factor_scratch, p);
  if (code == SDSGE_OK) {
    sdsge_forward_subst(factor_scratch, dev_scratch, solved_scratch, p);
    sdsge_backward_subst_chol_t(factor_scratch, solved_scratch, solved_scratch,
                                p);
  } else {
    memcpy(factor_scratch, omega, sizeof(f64) * p * p);
    if (sdsge_lu_factor_inplace(factor_scratch, pivot_scratch, p) !=
        SDSGE_LU_SUCCESS) {
      return DIAG_LINALG;
    }
    sdsge_lu_solve(factor_scratch, pivot_scratch, dev_scratch, solved_scratch,
                   p, 1);
  }

  /* Stat = n * dev^T omega^-1 dev. */
  f64 stat = 0.0;
  for (i64 i = 0; i < p; ++i) {
    stat += dev_scratch[i] * solved_scratch[i];
  }
  stat *= (f64)n;
  if (stat < 0.0 && stat > -1e-12) {
    stat = 0.0;
  }
  *stat_out = stat;
  return DIAG_OK;
}

static int wald_hac_from_moments(f64 *SDSGE_RESTRICT moments,
                                 const f64 *SDSGE_RESTRICT target, const i64 n,
                                 const i64 p, const KernelID kernel_id,
                                 const WaldBandwidthMode bandwidth_mode,
                                 const i64 manual_bandwidth,
                                 f64 *SDSGE_RESTRICT scratch,
                                 i64 *SDSGE_RESTRICT pivot_scratch,
                                 f64 *SDSGE_RESTRICT stat_out) {
  f64 *mean = scratch;
  f64 *gamma = mean + p;
  f64 *omega = gamma + p * p;
  f64 *dev = omega + p * p;
  f64 *factor = dev + p;
  f64 *solved = factor + p * p;
  f64 *bandwidth_scratch = solved + p;
  i64 bandwidth = 0;

  sdsge_fill_mean_ax0(moments, n, p, mean);
  center_inplace(moments, mean, n, p);
  int status = wald_resolve_bandwidth(moments, kernel_id, bandwidth_mode,
                                      manual_bandwidth, n, p, bandwidth_scratch,
                                      &bandwidth);
  if (status != DIAG_OK)
    return status;
  sdsge_hac_estimator_matmul(moments, kernel_id, bandwidth, n, p, gamma, omega);
  return sdsge_wald_stat_from_mean_and_cov(
      mean, target, omega, n, p, dev, factor, pivot_scratch, solved, stat_out);
}

i64 sdsge_wald_mean_hac_arena_size(const i64 n, const i64 q) {
  return n * q + 3 * q * q + 4 * q;
}

i64 sdsge_wald_covariance_hac_arena_size(const i64 n, const i64 q) {
  const i64 v = q * (q + 1) / 2;
  return n * q + n * v + 3 * v * v + 5 * v;
}

i64 sdsge_wald_second_moment_hac_arena_size(const i64 n, const i64 q) {
  const i64 v = q * (q + 1) / 2;
  return n * v + 3 * v * v + 5 * v;
}

int sdsge_wald_mean_hac(const f64 *SDSGE_RESTRICT g,
                        const f64 *SDSGE_RESTRICT target, const i64 n,
                        const i64 q, const KernelID kernel_id,
                        const WaldBandwidthMode bandwidth_mode,
                        const i64 manual_bandwidth, f64 *SDSGE_RESTRICT arena,
                        i64 *SDSGE_RESTRICT pivot_scratch,
                        f64 *SDSGE_RESTRICT stat_out) {
  if (n < 2)
    return DIAG_INSUFFICIENT_SAMPLES;
  f64 *moments = arena;
  f64 *scratch = moments + n * q;
  memcpy(moments, g, sizeof(f64) * n * q);
  return wald_hac_from_moments(moments, target, n, q, kernel_id, bandwidth_mode,
                               manual_bandwidth, scratch, pivot_scratch,
                               stat_out);
}

static int wald_matrix_moment_hac(const f64 *SDSGE_RESTRICT g,
                                  const f64 *SDSGE_RESTRICT target, const i64 n,
                                  const i64 q, const KernelID kernel_id,
                                  const WaldBandwidthMode bandwidth_mode,
                                  const i64 manual_bandwidth,
                                  f64 *SDSGE_RESTRICT arena,
                                  i64 *SDSGE_RESTRICT pivot_scratch,
                                  f64 *SDSGE_RESTRICT stat_out) {
  if (n < 2)
    return DIAG_INSUFFICIENT_SAMPLES;
  const i64 v = q * (q + 1) / 2;
  f64 *target_vec = arena;
  f64 *moments = target_vec + v;
  f64 *scratch = moments + n * v;
  int status =
      sdsge_fill_symmetric_target_vec(target, 1e-8, 1e-5, q, target_vec);
  if (status != DIAG_OK)
    return status;
  sdsge_symmetric_outer_prod_2dim(g, n, q, v, moments);
  return wald_hac_from_moments(moments, target_vec, n, v, kernel_id,
                               bandwidth_mode, manual_bandwidth, scratch,
                               pivot_scratch, stat_out);
}

int sdsge_wald_covariance_hac(const f64 *SDSGE_RESTRICT g,
                              const f64 *SDSGE_RESTRICT target, const i64 n,
                              const i64 q, const KernelID kernel_id,
                              const WaldBandwidthMode bandwidth_mode,
                              const i64 manual_bandwidth,
                              f64 *SDSGE_RESTRICT arena,
                              i64 *SDSGE_RESTRICT pivot_scratch,
                              f64 *SDSGE_RESTRICT stat_out) {
  /* Covariance moments are vech((g_t - mean(g))(g_t - mean(g))'). */
  const i64 v = q * (q + 1) / 2;
  f64 *target_vec = arena;
  f64 *centered = target_vec + v;
  f64 *mean = centered + n * q;
  f64 *moments = mean;
  f64 *scratch = moments + n * v;
  if (n < 2)
    return DIAG_INSUFFICIENT_SAMPLES;
  sdsge_fill_mean_ax0(g, n, q, mean);
  sdsge_fill_centered_ax0(g, mean, n, q, centered);
  int status =
      sdsge_fill_symmetric_target_vec(target, 1e-8, 1e-5, q, target_vec);
  if (status != DIAG_OK)
    return status;
  sdsge_symmetric_outer_prod_2dim(centered, n, q, v, moments);
  return wald_hac_from_moments(moments, target_vec, n, v, kernel_id,
                               bandwidth_mode, manual_bandwidth, scratch,
                               pivot_scratch, stat_out);
}

int sdsge_wald_second_moment_hac(const f64 *SDSGE_RESTRICT g,
                                 const f64 *SDSGE_RESTRICT target, const i64 n,
                                 const i64 q, const KernelID kernel_id,
                                 const WaldBandwidthMode bandwidth_mode,
                                 const i64 manual_bandwidth,
                                 f64 *SDSGE_RESTRICT arena,
                                 i64 *SDSGE_RESTRICT pivot_scratch,
                                 f64 *SDSGE_RESTRICT stat_out) {
  return wald_matrix_moment_hac(g, target, n, q, kernel_id, bandwidth_mode,
                                manual_bandwidth, arena, pivot_scratch,
                                stat_out);
}

int sdsge_symmetric_outer_prod_2dim(const f64 *SDSGE_RESTRICT x, const i64 n,
                                    const i64 p, const i64 q,
                                    f64 *SDSGE_RESTRICT out) {
  /* out is (n, q) with q = floor(p * (p + 1) / 2); the python side computes q
   * for shape checks, so don't recompute it here. x is (n, p): its row stride
   * is p, only out's row stride is q. */
  i64 k = 0;
  f64 x_i = 0.0;
  for (i64 t = 0; t < n; ++t) {
    k = 0;
    for (i64 i = 0; i < p; ++i) {
      x_i = x[t * p + i];
      for (i64 j = i; j < p; ++j) {
        out[t * q + k] = x_i * x[t * p + j];
        k += 1;
      }
    }
  }
  return DIAG_OK;
}

int sdsge_fill_symmetric_target_vec(const f64 *SDSGE_RESTRICT target,
                                    const f64 atol, const f64 rtol, const i64 p,
                                    f64 *SDSGE_RESTRICT out) {

  i64 k = 0;
  f64 a = 0.0;
  f64 b = 0.0;
  f64 diff = 0.0;

  for (i64 i = 0; i < p; ++i) {
    for (i64 j = i; j < p; ++j) {
      a = target[i * p + j];
      b = target[j * p + i];
      if (a != b) {
        diff = fabs(a - b);

        if (!isfinite(diff) || diff > atol + rtol * fabs(b)) {
          return DIAG_BAD_SHAPE;
        }
      }
      out[k] = a;
      k += 1;
    }
  }
  return DIAG_OK;
}
