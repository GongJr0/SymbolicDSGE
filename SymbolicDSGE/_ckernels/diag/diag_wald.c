#include "diag_wald.h"
#include <math.h>

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

static void fill_mean_ax0(const f64 *SDSGE_RESTRICT x, f64 *SDSGE_RESTRICT mean,
                          const i64 n, const i64 p) {

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

static void fill_centered_ax0(const f64 *SDSGE_RESTRICT x,
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

void hac_estimator_matmul(f64 *SDSGE_RESTRICT r, KernelID kernel_id, i64 L,
                          i64 n, i64 p, f64 *SDSGE_RESTRICT gamma_scratch,
                          f64 *SDSGE_RESTRICT out) {
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
}
