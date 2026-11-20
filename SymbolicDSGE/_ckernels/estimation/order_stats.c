#include "order_stats.h"
#include "../_common/sdsge_common.h"
#include <math.h>

/*--- partition ---*/

/* Index of the median of the three range endpoints. */
static inline i64 med3_index(const f64 *SDSGE_RESTRICT a, const i64 lo,
                             const i64 hi) {
  const i64 mid = lo + ((hi - lo) >> 1);
  const f64 x = a[lo], y = a[mid], z = a[hi];

  if (x < y) {
    if (y < z) {
      return mid;
    }
    return (x < z) ? hi : lo;
  }
  if (x < z) {
    return lo;
  }
  return (y < z) ? hi : mid;
}

/* Hoare partition of a[lo..hi]: returns j with a[lo..j] <= a[j+1..hi] and
 * lo <= j < hi.
 *
 * The scans must stay strict and the pivot must be held by value. Non-strict
 * comparisons run the pointers past the range; a pivot held by index breaks
 * once its slot is swapped. */
static i64 hoare_partition(f64 *SDSGE_RESTRICT a, const i64 lo, const i64 hi) {
  const f64 p = a[med3_index(a, lo, hi)];
  i64 i = lo - 1;
  i64 j = hi + 1;

  while (1) {
    do {
      ++i;
    } while (a[i] < p);

    do {
      --j;
    } while (a[j] > p);

    if (i >= j) {
      return j;
    }

    const f64 t = a[i];
    a[i] = a[j];
    a[j] = t;
  }
}

/* Places the k-th smallest of a[lo..hi] at a[k], leaving a[lo..k-1] <= a[k] and
 * a[k] <= a[k+1..hi]. */
void sdsge_select_kth(f64 *SDSGE_RESTRICT a, i64 lo, i64 hi, const i64 k) {
  while (lo < hi) {
    const i64 j = hoare_partition(a, lo, hi);
    if (k <= j) {
      hi = j;
    } else {
      lo = j + 1;
    }
  }
}

static inline f64 min_in_range(const f64 *SDSGE_RESTRICT a, const i64 lo,
                               const i64 hi) {
  f64 m = a[lo];
  for (i64 i = lo + 1; i <= hi; ++i) {
    if (a[i] < m) {
      m = a[i];
    }
  }
  return m;
}

static inline f64 max_in_range(const f64 *SDSGE_RESTRICT a, const i64 lo,
                               const i64 hi) {
  f64 m = a[lo];
  for (i64 i = lo + 1; i <= hi; ++i) {
    if (a[i] > m) {
      m = a[i];
    }
  }
  return m;
}

/* --- quantiles ---
 *
 * sdsge_quantile1d: Single empirical quantile of buf at level q, returned by
 * value.
 *
 * sdsge_quantiles1d: Empirical quantiles of buf at the ascending levels
 * q[0..n_q-1]. returned into *out. */
f64 sdsge_quantile1d(f64 *SDSGE_RESTRICT buf, const i64 S, const f64 q) {
  f64 lo, hi;
  const f64 h = (f64)(S - 1) * q;
  const i64 k = (i64)floor(h);
  const f64 g = h - (f64)k;

  if (g <= 0.0 || k + 1 >= S) {
    sdsge_select_kth(buf, 0, S - 1, k);
    return buf[k];
  }

  if (k + 1 < S - k - 1) {
    sdsge_select_kth(buf, 0, S - 1, k + 1);
    hi = buf[k + 1];
    lo = max_in_range(buf, 0, k);
  } else {
    sdsge_select_kth(buf, 0, S - 1, k);
    lo = buf[k];
    hi = min_in_range(buf, k + 1, S - 1);
  }
  return lo + g * (hi - lo);
}

void sdsge_quantiles1d(f64 *SDSGE_RESTRICT buf, const i64 S,
                       const f64 *SDSGE_RESTRICT q, const i64 n_q,
                       f64 *SDSGE_RESTRICT out) {
  i64 base = 0;
  i64 last_k = -1;
  f64 last_lo = 0.0;
  f64 last_hi = 0.0;

  for (i64 t = 0; t < n_q; ++t) {
    const f64 h = (f64)(S - 1) * q[t];
    const i64 k = (i64)floor(h);
    const f64 g = h - (f64)k;

    /* Close levels share a k. `base` may already sit at k + 1, and selecting a
     * rank below base walks a range that no longer holds it. */
    if (k == last_k) {
      out[t] = (g > 0.0) ? last_lo + g * (last_hi - last_lo) : last_lo;
      continue;
    }

    if (g <= 0.0 || k + 1 >= S) {
      sdsge_select_kth(buf, base, S - 1, k);
      out[t] = buf[k];
      base = k;
      continue;
    }

    f64 lo, hi;
    /* Both neighbors come off one selection plus a scan of the side that was
     * left unordered. */
    if (k - base + 1 < S - k - 1) {
      sdsge_select_kth(buf, base, S - 1, k + 1);
      hi = buf[k + 1];
      lo = max_in_range(buf, base, k);
      base = k + 1;
    } else {
      sdsge_select_kth(buf, base, S - 1, k);
      lo = buf[k];
      hi = min_in_range(buf, k + 1, S - 1);
      base = k;
    }

    out[t] = lo + g * (hi - lo);
    last_k = k;
    last_lo = lo;
    last_hi = hi;
  }
}

/*--- sort ---*/

static void insertion_sort_range(f64 *SDSGE_RESTRICT a, const i64 lo,
                                 const i64 hi) {
  for (i64 i = lo + 1; i <= hi; ++i) {
    const f64 v = a[i];
    i64 j = i - 1;
    while (j >= lo && a[j] > v) {
      a[j + 1] = a[j];
      --j;
    }
    a[j + 1] = v;
  }
}

static void sift_down(f64 *SDSGE_RESTRICT b, const i64 start, const i64 n) {
  const f64 v = b[start];
  i64 root = start;

  while (1) {
    i64 child = 2 * root + 1;
    if (child >= n) {
      break;
    }
    if (child + 1 < n && b[child] < b[child + 1]) {
      ++child;
    }
    if (b[child] <= v) {
      break;
    }
    b[root] = b[child];
    root = child;
  }
  b[root] = v;
}

static void heap_sort_range(f64 *SDSGE_RESTRICT a, const i64 lo, const i64 hi) {
  f64 *b = a + lo;
  const i64 n = hi - lo + 1;

  for (i64 i = n / 2 - 1; i >= 0; --i) {
    sift_down(b, i, n);
  }
  for (i64 end = n - 1; end > 0; --end) {
    const f64 t = b[0];
    b[0] = b[end];
    b[end] = t;
    sift_down(b, 0, end);
  }
}

/* Introsort: recurse into the smaller side and loop on the larger. The stack
 * stays logarithmic. Fall back to heapsort once the partition budget runs
 * out. */
static void introsort_range(f64 *SDSGE_RESTRICT a, i64 lo, i64 hi, i64 depth) {
  const i64 CUTOFF = 24;

  while (hi - lo + 1 > CUTOFF) {
    if (depth == 0) {
      heap_sort_range(a, lo, hi);
      return;
    }
    --depth;

    const i64 j = hoare_partition(a, lo, hi);
    if (j - lo < hi - j - 1) {
      introsort_range(a, lo, j, depth);
      lo = j + 1;
    } else {
      introsort_range(a, j + 1, hi, depth);
      hi = j;
    }
  }
  insertion_sort_range(a, lo, hi);
}

static void sort_range(f64 *SDSGE_RESTRICT a, const i64 lo, const i64 hi) {
  const i64 n = hi - lo + 1;
  if (n < 2) {
    return;
  }

  i64 lg = 0;
  for (i64 t = n; t > 1; t >>= 1) {
    ++lg;
  }
  introsort_range(a, lo, hi, 2 * lg);
}

/*--- hpd ---*/
static inline i64 sdsge_hpd_window(const i64 S, const f64 alpha) {
  return (i64)ceil((1.0 - alpha) * S);
}
static void sdsge_hpd1d(f64 *SDSGE_RESTRICT buf, const i64 S, const i64 window,
                        f64 *SDSGE_RESTRICT lo, f64 *SDSGE_RESTRICT hi) {
  const i64 k = S - window + 1;

  if (window >= S) {
    f64 mn = buf[0];
    f64 mx = buf[0];
    for (i64 i = 1; i < S; ++i) {
      if (buf[i] < mn) {
        mn = buf[i];
      }
      if (buf[i] > mx) {
        mx = buf[i];
      }
    }
    *lo = mn;
    *hi = mx;
    return;
  }

  /* The upper cut has to come first. Selecting it afterwards would repartition
   * the lower region and scatter the k smallest out of buf[0..k-1]. */
  sdsge_select_kth(buf, 0, S - 1, window - 1);
  sort_range(buf, window - 1, S - 1);
  sdsge_select_kth(buf, 0, window - 2, k - 1);
  sort_range(buf, 0, k - 1);

  i64 best = 0;
  f64 best_width = buf[window - 1] - buf[0];
  for (i64 j = 1; j < k; ++j) {
    const f64 width = buf[window - 1 + j] - buf[j];
    if (width < best_width) {
      best_width = width;
      best = j;
    }
  }
  *lo = buf[best];
  *hi = buf[window - 1 + best];
}

void sdsge_hpd_intervals(f64 *SDSGE_RESTRICT pooled, const f64 alpha,
                         const i64 S, const i64 n_par, const i64 n_workers,
                         f64 *SDSGE_RESTRICT lo, f64 *SDSGE_RESTRICT hi) {
  const i64 window = sdsge_hpd_window(S, alpha);

#pragma omp parallel for num_threads(n_workers) if (n_par > 1) schedule(static)
  for (i64 j = 0; j < n_par; ++j) {
    sdsge_hpd1d(&pooled[j * S], S, window, &lo[j], &hi[j]);
  }
}
