#include "jb_lookup.h"

#include <math.h>

/* Small-N Jarque-Bera critical-value tables. Source: Wuertz and Keller (2004),
 * "Precise finite-sample quantiles of the Jarque-Bera adjusted Lagrange
 * multiplier test". These mirror JB_N_GRID / JB_PVAL_GRID /
 * JB_SMALL_N_CRITICAL_VALUES in SymbolicDSGE/_diag_tests/jb_lookup.py exactly;
 * the parity tests pin the two copies together. */

#define JB_NN 15 /* sample-size grid nodes */
#define JB_NP 23 /* p-value grid nodes / critical-value rows */

/* Sample-size grid (ascending). Stored f64 so the ascending bracket search runs
 * on a single primitive; the magnitudes are exact in double. */
static const f64 JB_N_GRID[JB_NN] = {
    10.0,   20.0,   35.0,   50.0,   75.0,   100.0,  150.0, 200.0,
    300.0,  500.0,  800.0,  1000.0, 1600.0, 2400.0, 10000.0};

/* Upper-tail probability grid (ascending). */
static const f64 JB_PVAL_GRID[JB_NP] = {
    0.0001, 0.0005, 0.0010, 0.0050, 0.0100, 0.0500, 0.1000, 0.1500,
    0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.8500,
    0.9000, 0.9500, 0.9900, 0.9950, 0.9990, 0.9995, 0.9999};

/* Critical values, row-major CV[p-value][sample-size]. Each column is strictly
 * descending in the p-value index. */
static const f64 JB_CV[JB_NP][JB_NN] = {
    {15.345, 46.996, 66.612, 71.734, 69.910, 68.032, 60.632, 54.736, 47.572,
     38.847, 33.247, 31.213, 26.956, 24.249, 19.940},
    {12.444, 31.159, 40.759, 43.256, 41.909, 40.430, 37.229, 34.330, 30.561,
     26.270, 23.045, 21.979, 19.760, 18.366, 16.052},
    {10.995, 24.970, 31.969, 33.753, 32.738, 31.840, 29.547, 27.551, 24.830,
     21.812, 19.521, 18.736, 17.150, 16.083, 14.397},
    {7.304, 13.471, 16.414, 17.281, 17.305, 16.959, 16.257, 15.638, 14.669,
     13.583, 12.726, 12.366, 11.762, 11.384, 10.792},
    {5.7029, 9.7182, 11.736, 12.392, 12.586, 12.491, 12.185, 11.882, 11.3580,
     10.778, 10.299, 10.117, 9.8095, 9.6084, 9.3128},
    {2.5247, 3.7954, 4.5929, 4.9757, 5.2777, 5.4300, 5.5984, 5.6758, 5.7732,
     5.8551, 5.9103, 5.9242, 5.9569, 5.9671, 5.9857},
    {1.6232, 2.3470, 2.8814, 3.1834, 3.4862, 3.6734, 3.9041, 4.0327, 4.1891,
     4.3317, 4.4274, 4.4568, 4.5132, 4.5424, 4.5888},
    {1.2826, 1.8230, 2.2533, 2.5094, 2.7713, 2.9390, 3.1416, 3.2580, 3.4003,
     3.5312, 3.6198, 3.6507, 3.7016, 3.7309, 3.7778},
    {1.1236, 1.5623, 1.9162, 2.1278, 2.3463, 2.4865, 2.6558, 2.7559, 2.8764,
     2.9882, 3.0645, 3.0909, 3.1360, 3.1611, 3.2036},
    {0.9389, 1.2516, 1.4997, 1.6466, 1.7975, 1.8944, 2.0112, 2.0807, 2.1639,
     2.2427, 2.2962, 2.3153, 2.3460, 2.3650, 2.3968},
    {0.8077, 1.0360, 1.2115, 1.3128, 1.4165, 1.4828, 1.5619, 1.6087, 1.6649,
     1.7175, 1.7547, 1.7679, 1.7889, 1.8024, 1.8248},
    {0.6950, 0.8574, 0.9771, 1.0447, 1.1126, 1.1563, 1.2076, 1.2385, 1.2752,
     1.3101, 1.3338, 1.3420, 1.3568, 1.3655, 1.3808},
    {0.5885, 0.6948, 0.7699, 0.8114, 0.8529, 0.8800, 0.9105, 0.9292, 0.9518,
     0.9732, 0.9882, 0.9931, 1.0024, 1.0085, 1.0181},
    {0.4801, 0.5378, 0.5769, 0.5985, 0.6202, 0.6348, 0.6508, 0.6610, 0.6730,
     0.6851, 0.6940, 0.6965, 0.7018, 0.7056, 0.7108},
    {0.3618, 0.3777, 0.3896, 0.3969, 0.4046, 0.4105, 0.4168, 0.4213, 0.4267,
     0.4325, 0.4368, 0.4376, 0.4402, 0.4421, 0.4451},
    {0.2950, 0.2938, 0.2958, 0.2982, 0.3010, 0.3044, 0.3071, 0.3096, 0.3130,
     0.3163, 0.3189, 0.3194, 0.3209, 0.3221, 0.3245},
    {0.2192, 0.2047, 0.2002, 0.1997, 0.1997, 0.2006, 0.2016, 0.2024, 0.2040,
     0.2060, 0.2071, 0.2074, 0.2081, 0.2089, 0.2106},
    {0.1272, 0.1084, 0.1022, 0.1005, 0.0995, 0.0996, 0.0992, 0.0995, 0.1000,
     0.1005, 0.1010, 0.1012, 0.1013, 0.1019, 0.1024},
    {0.0304, 0.0230, 0.0208, 0.0203, 0.0198, 0.0197, 0.0196, 0.0196, 0.0197,
     0.0198, 0.0197, 0.0199, 0.0200, 0.0200, 0.0200},
    {0.0156, 0.0116, 0.0104, 0.0101, 0.0099, 0.0098, 0.0098, 0.0098, 0.0099,
     0.0099, 0.0098, 0.0099, 0.0099, 0.0099, 0.0100},
    {0.0032, 0.0023, 0.0021, 0.0020, 0.0020, 0.0019, 0.0019, 0.0019, 0.0020,
     0.0019, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020},
    {0.0016, 0.0012, 0.0010, 0.0010, 0.0010, 0.0009, 0.0010, 0.0010, 0.0010,
     0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010},
    {0.0003, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002,
     0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002}};

/* searchsorted(arr, val, side='left'): first index i with arr[i] >= val. */
static inline i64 searchsorted_left(f64 val, const f64 *SDSGE_RESTRICT arr,
                                    i64 n) {
  i64 lo = 0, hi = n;
  while (lo < hi) {
    i64 mid = (lo + hi) / 2;
    if (arr[mid] < val)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

void sdsge_jb_find_hilo_ascending(f64 val, const f64 *SDSGE_RESTRICT arr, i64 n,
                                  i64 *SDSGE_RESTRICT lo,
                                  i64 *SDSGE_RESTRICT hi) {
  i64 idx = searchsorted_left(val, arr, n);
  if (idx == 0) {
    *lo = 0;
    *hi = 0;
  } else if (idx == n) {
    *lo = n - 1;
    *hi = n - 1;
  } else if (arr[idx] == val) {
    *lo = idx;
    *hi = idx;
  } else {
    *lo = idx - 1;
    *hi = idx;
  }
}

void sdsge_jb_find_hilo_descending(f64 val, const f64 *SDSGE_RESTRICT arr,
                                   i64 n, i64 *SDSGE_RESTRICT lo,
                                   i64 *SDSGE_RESTRICT hi) {
  /* searchsorted(-arr, -val, side='left'): first index i with arr[i] <= val. */
  i64 idx = 0, upper = n;
  while (idx < upper) {
    i64 mid = (idx + upper) / 2;
    if (arr[mid] > val)
      idx = mid + 1;
    else
      upper = mid;
  }
  if (idx == 0) {
    *lo = 0;
    *hi = 0;
  } else if (idx == n) {
    *lo = n - 1;
    *hi = n - 1;
  } else if (arr[idx] == val) {
    *lo = idx;
    *hi = idx;
  } else {
    *lo = idx - 1;
    *hi = idx;
  }
}

f64 sdsge_jb_isf_interp(i64 n, f64 p) {
  if (isnan(p))
    return NAN;
  if (p <= 0.0)
    return INFINITY;
  if (p >= 1.0)
    return 0.0;

  i64 n_lo, n_hi, p_lo, p_hi;
  sdsge_jb_find_hilo_ascending((f64)n, JB_N_GRID, JB_NN, &n_lo, &n_hi);
  sdsge_jb_find_hilo_ascending(p, JB_PVAL_GRID, JB_NP, &p_lo, &p_hi);

  if (n_lo == n_hi && p_lo == p_hi) {
    return JB_CV[p_lo][n_lo];
  } else if (n_lo == n_hi) {
    f64 w_p =
        (p - JB_PVAL_GRID[p_lo]) / (JB_PVAL_GRID[p_hi] - JB_PVAL_GRID[p_lo]);
    return (1.0 - w_p) * JB_CV[p_lo][n_lo] + w_p * JB_CV[p_hi][n_lo];
  } else if (p_lo == p_hi) {
    f64 w_n = ((f64)n - JB_N_GRID[n_lo]) / (JB_N_GRID[n_hi] - JB_N_GRID[n_lo]);
    return (1.0 - w_n) * JB_CV[p_lo][n_lo] + w_n * JB_CV[p_lo][n_hi];
  } else {
    f64 w_n = ((f64)n - JB_N_GRID[n_lo]) / (JB_N_GRID[n_hi] - JB_N_GRID[n_lo]);
    f64 w_p =
        (p - JB_PVAL_GRID[p_lo]) / (JB_PVAL_GRID[p_hi] - JB_PVAL_GRID[p_lo]);
    f64 val_lo =
        (1.0 - w_n) * JB_CV[p_lo][n_lo] + w_n * JB_CV[p_lo][n_hi];
    f64 val_hi =
        (1.0 - w_n) * JB_CV[p_hi][n_lo] + w_n * JB_CV[p_hi][n_hi];
    return (1.0 - w_p) * val_lo + w_p * val_hi;
  }
}

f64 sdsge_jb_pval_interp(i64 n, f64 x) {
  if (isnan(x))
    return NAN;
  if (x <= 0.0)
    return 1.0;
  if (isinf(x))
    return 0.0;

  i64 n_lo, n_hi;
  sdsge_jb_find_hilo_ascending((f64)n, JB_N_GRID, JB_NN, &n_lo, &n_hi);

  /* Critical-value column(s), copied contiguous for the descending bracket. */
  f64 cv[JB_NP];
  i64 p_lo, p_hi;

  if (n_lo == n_hi) {
    for (i64 i = 0; i < JB_NP; ++i)
      cv[i] = JB_CV[i][n_lo];
    sdsge_jb_find_hilo_descending(x, cv, JB_NP, &p_lo, &p_hi);
    if (p_lo == p_hi)
      return JB_PVAL_GRID[p_lo];
    f64 w_p = (x - cv[p_lo]) / (cv[p_hi] - cv[p_lo]);
    return (1.0 - w_p) * JB_PVAL_GRID[p_lo] + w_p * JB_PVAL_GRID[p_hi];
  } else {
    f64 w_n = ((f64)n - JB_N_GRID[n_lo]) / (JB_N_GRID[n_hi] - JB_N_GRID[n_lo]);
    for (i64 i = 0; i < JB_NP; ++i)
      cv[i] = (1.0 - w_n) * JB_CV[i][n_lo] + w_n * JB_CV[i][n_hi];
    sdsge_jb_find_hilo_descending(x, cv, JB_NP, &p_lo, &p_hi);
    if (p_lo == p_hi)
      return JB_PVAL_GRID[p_lo];
    f64 w_p = (x - cv[p_lo]) / (cv[p_hi] - cv[p_lo]);
    return (1.0 - w_p) * JB_PVAL_GRID[p_lo] + w_p * JB_PVAL_GRID[p_hi];
  }
}

void sdsge_jb_isf_interp_into(i64 n, const f64 *SDSGE_RESTRICT p, i64 m,
                              f64 *SDSGE_RESTRICT out) {
  for (i64 i = 0; i < m; ++i)
    out[i] = sdsge_jb_isf_interp(n, p[i]);
}

void sdsge_jb_pval_interp_into(i64 n, const f64 *SDSGE_RESTRICT x, i64 m,
                               f64 *SDSGE_RESTRICT out) {
  for (i64 i = 0; i < m; ++i)
    out[i] = sdsge_jb_pval_interp(n, x[i]);
}
