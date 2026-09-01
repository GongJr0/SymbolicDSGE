#include "transforms.h"
#include "layout.h"
#include <math.h>
#include <stddef.h>

/* Column means and sums of squared deviations of a row-major (n, p) buffer, in
 * one Welford pass. Writes mean(p) and m2(p); both are zeroed here. The
 * reciprocal count is hoisted out of the column loop, so a row costs one
 * division instead of p. */
static void welford_ax0(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 p,
                        f64 *SDSGE_RESTRICT mean, f64 *SDSGE_RESTRICT m2) {
  for (i64 j = 0; j < p; ++j) {
    mean[j] = 0.0;
    m2[j] = 0.0;
  }

  for (i64 i = 0; i < n; ++i) {
    const f64 *row = x + i * p;
    const f64 inv_count = 1.0 / (f64)(i + 1);

    for (i64 j = 0; j < p; ++j) {
      const f64 value = row[j];
      const f64 delta = value - mean[j];

      mean[j] += delta * inv_count;
      m2[j] += delta * (value - mean[j]);
    }
  }
}

i64 sdsge_standardize_ax0(const f64 *SDSGE_RESTRICT x, const i64 ddof,
                          const i64 n, const i64 p, f64 *SDSGE_RESTRICT scratch,
                          f64 *SDSGE_RESTRICT out) {
  if (n <= 0 || p <= 0 || n - ddof <= 0) {
    return SDSGE_TRANSFORM_BAD_ARG;
  }

  f64 *mean = scratch;
  f64 *inv_std = scratch + p;

  welford_ax0(x, n, p, mean, inv_std); /* inv_std holds m2 for now */

  const f64 denominator = (f64)(n - ddof);

  for (i64 j = 0; j < p; ++j) {
    const f64 variance = inv_std[j] / denominator;

    /* A zero-variance column scales to zeros rather than dividing through, and
     * the comparison also absorbs the small negative m2 that cancellation can
     * leave on a constant column. */
    inv_std[j] = (variance > 0.0) ? 1.0 / sqrt(variance) : 0.0;
  }

  for (i64 i = 0; i < n; ++i) {
    const f64 *row = x + i * p;
    f64 *dst = out + i * p;

    for (i64 j = 0; j < p; ++j) {
      dst[j] = (row[j] - mean[j]) * inv_std[j];
    }
  }

  return SDSGE_TRANSFORM_SUCCESS;
}

i64 sdsge_log(const f64 *SDSGE_RESTRICT x, const f64 offset, const i64 n,
              const i64 p, f64 *SDSGE_RESTRICT out) {
  if (n <= 0 || p <= 0) {
    return SDSGE_TRANSFORM_BAD_ARG;
  }

  const i64 total = n * p;

  for (i64 i = 0; i < total; ++i) {
    out[i] = log(x[i] + offset);
  }

  return SDSGE_TRANSFORM_SUCCESS;
}

i64 sdsge_log_diff(const f64 *SDSGE_RESTRICT x, const f64 offset, const i64 n,
                   const i64 p, f64 *SDSGE_RESTRICT scratch,
                   f64 *SDSGE_RESTRICT out) {
  if (n <= 0 || p <= 0) {
    return SDSGE_TRANSFORM_BAD_ARG;
  }

  f64 *previous = scratch;

  for (i64 j = 0; j < p; ++j) {
    previous[j] = log(x[j] + offset);
  }

  /* Row i of the input produces row i - 1 of the output, so a single-row input
   * writes nothing. */
  for (i64 i = 1; i < n; ++i) {
    const f64 *row = x + i * p;
    f64 *dst = out + (i - 1) * p;

    for (i64 j = 0; j < p; ++j) {
      const f64 current = log(row[j] + offset);

      dst[j] = current - previous[j];
      previous[j] = current;
    }
  }

  return SDSGE_TRANSFORM_SUCCESS;
}

i64 sdsge_diff(const f64 *SDSGE_RESTRICT x, const i64 order, const i64 n,
               const i64 p, f64 *SDSGE_RESTRICT scratch,
               f64 *SDSGE_RESTRICT out) {
  if (n <= 0 || p <= 0 || order < 1) {
    return SDSGE_TRANSFORM_BAD_ARG;
  }

  /*
   * Streams all `order` difference levels in one pass over the input, so no
   * intermediate (n - 1, p) array is needed and `out` is exactly (n - order,
   * p). `scratch` holds the previous value of each level: state[m * p + j] is
   * the last difference of order m seen in column j.
   *
   * Level m is undefined until row m, where it is primed instead of
   * differenced. Once every level has been primed, row t emits row t - order.
   * The subtractions are the same ones repeated `np.diff` performs, in the same
   * order, so the results agree to the bit.
   */
  for (i64 i = 0; i < n; ++i) {
    const f64 *row = x + i * p;
    const i64 out_row = i - order;
    f64 *dst = (out_row >= 0) ? out + out_row * p : NULL;

    for (i64 j = 0; j < p; ++j) {
      f64 value = row[j];
      i64 level = 0;

      for (; level < order; ++level) {
        f64 *state = scratch + level * p + j;

        if (i <= level) { /* first arrival at this level */
          *state = value;
          break;
        }

        const f64 previous = *state;

        *state = value;
        value -= previous;
      }

      if (dst != NULL) {
        dst[j] = value;
      }
    }
  }

  return SDSGE_TRANSFORM_SUCCESS;
}

i64 sdsge_rolling_mean(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 p,
                       const i64 window, f64 *SDSGE_RESTRICT scratch,
                       f64 *SDSGE_RESTRICT out) {
  if (n <= 0 || p <= 0 || window < 1 || window > n) {
    return SDSGE_TRANSFORM_BAD_ARG;
  }

  f64 *sum = scratch;

  for (i64 j = 0; j < p; ++j) {
    sum[j] = 0.0;
  }

  /* Initial window: x[0:window]. */
  for (i64 i = 0; i < window; ++i) {
    const f64 *row = x + i * p;

    for (i64 j = 0; j < p; ++j) {
      sum[j] += row[j];
    }
  }

  const f64 scale = 1.0 / (f64)window;

  for (i64 j = 0; j < p; ++j) {
    out[j] = sum[j] * scale;
  }

  const i64 n_out = n - window + 1;

  /*
   * Slide from x[i - 1 : i - 1 + window]
   *         to x[i     : i     + window].
   */
  for (i64 i = 1; i < n_out; ++i) {
    const f64 *leaving = x + (i - 1) * p;
    const f64 *entering = x + (i + window - 1) * p;
    f64 *dst = out + i * p;

    for (i64 j = 0; j < p; ++j) {
      sum[j] += entering[j] - leaving[j];
      dst[j] = sum[j] * scale;
    }
  }

  return SDSGE_TRANSFORM_SUCCESS;
}

/* Shared body of the rolling variance and standard deviation.
 *
 * The window slides by removing the leaving point from the Welford state and
 * adding the entering one, which is O(n * p) rather than the O(n * p * window)
 * of recomputing each window. The downdate reintroduces the cancellation
 * Welford avoids, so m2 drifts on a long window over a large mean; it is
 * clamped at zero, and callers comparing against a per-window recomputation
 * should expect agreement to a tolerance rather than to the bit. */

static i64 rolling_moment_ax0(const f64 *SDSGE_RESTRICT x, const i64 n,
                              const i64 p, const i64 window, const i64 ddof,
                              const int take_sqrt, f64 *SDSGE_RESTRICT scratch,
                              f64 *SDSGE_RESTRICT out) {
  if (n <= 0 || p <= 0 || window < 1 || window > n || window - ddof <= 0) {
    return SDSGE_TRANSFORM_BAD_ARG;
  }

  f64 *mean = scratch;
  f64 *m2 = scratch + p;

  welford_ax0(x, window, p, mean, m2); /* Welford state for x[0:window] */

  const i64 n_out = n - window + 1;
  const f64 inv_denominator = 1.0 / (f64)(window - ddof);
  const f64 inv_window = 1.0 / (f64)window;
  const f64 inv_reduced = (window > 1) ? 1.0 / (f64)(window - 1) : 0.0;

  for (i64 i = 0; i < n_out; ++i) {
    f64 *dst = out + i * p;

    for (i64 j = 0; j < p; ++j) {
      const f64 numerator = (m2[j] > 0.0) ? m2[j] : 0.0;
      const f64 moment = numerator * inv_denominator;

      dst[j] = take_sqrt ? sqrt(moment) : moment;
    }

    if (i + 1 == n_out) {
      break;
    }

    const f64 *leaving = x + i * p;
    const f64 *entering = x + (i + window) * p;

    for (i64 j = 0; j < p; ++j) {
      f64 reduced_mean = 0.0;
      f64 reduced_m2 = 0.0;

      /* A one-wide window carries no state to reduce, and its only valid ddof
       * is 0, so the reduced window is empty. */
      if (window > 1) {
        const f64 old_mean = mean[j];
        const f64 old_value = leaving[j];

        reduced_mean = ((f64)window * old_mean - old_value) * inv_reduced;
        reduced_m2 =
            m2[j] - (old_value - old_mean) * (old_value - reduced_mean);
      }

      /* Add entering[j] back, bringing the window to `window` points. */
      const f64 value = entering[j];
      const f64 delta = value - reduced_mean;
      const f64 new_mean = reduced_mean + delta * inv_window;

      mean[j] = new_mean;
      m2[j] = reduced_m2 + delta * (value - new_mean);

      if (m2[j] < 0.0) {
        m2[j] = 0.0;
      }
    }
  }

  return SDSGE_TRANSFORM_SUCCESS;
}
i64 sdsge_rolling_var(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 p,
                      const i64 window, const i64 ddof,
                      f64 *SDSGE_RESTRICT scratch, f64 *SDSGE_RESTRICT out) {
  return rolling_moment_ax0(x, n, p, window, ddof, 0, scratch, out);
}

i64 sdsge_rolling_std(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 p,
                      const i64 window, const i64 ddof,
                      f64 *SDSGE_RESTRICT scratch, f64 *SDSGE_RESTRICT out) {
  return rolling_moment_ax0(x, n, p, window, ddof, 1, scratch, out);
}

static int sdsge_mc_transform_status(const i64 status,
                                     i64 *SDSGE_RESTRICT int_out) {
  if (int_out != NULL) {
    int_out[0] = status;
  }
  return (int)status;
}

int sdsge_mc_standardize_runner(const i64 rep_idx,
                                f64 *SDSGE_RESTRICT float_in_work,
                                f64 *SDSGE_RESTRICT float_out,
                                i64 *SDSGE_RESTRICT int_work,
                                i64 *SDSGE_RESTRICT int_out, const void *ctx) {
  (void)rep_idx;
  (void)int_work;
  const sdsge_mc_standardize_step_ctx *config = ctx;
  const arena_offset in_off = sdsge_mc_transform_arena_offset(
      SDSGE_MC_TRANSFORM_STANDARDIZE, config->n, config->p, 0);
  const i64 status =
      sdsge_standardize_ax0(float_in_work, config->ddof, config->n, config->p,
                            float_in_work + in_off.foffset[0], float_out);
  return sdsge_mc_transform_status(status, int_out);
}

int sdsge_mc_log_runner(const i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
                        f64 *SDSGE_RESTRICT float_out,
                        i64 *SDSGE_RESTRICT int_work,
                        i64 *SDSGE_RESTRICT int_out, const void *ctx) {
  (void)rep_idx;
  (void)int_work;
  const sdsge_mc_log_step_ctx *config = ctx;
  const i64 status =
      sdsge_log(float_in_work, config->offset, config->n, config->p, float_out);
  return sdsge_mc_transform_status(status, int_out);
}

int sdsge_mc_log_diff_runner(const i64 rep_idx,
                             f64 *SDSGE_RESTRICT float_in_work,
                             f64 *SDSGE_RESTRICT float_out,
                             i64 *SDSGE_RESTRICT int_work,
                             i64 *SDSGE_RESTRICT int_out, const void *ctx) {
  (void)rep_idx;
  (void)int_work;
  const sdsge_mc_log_diff_step_ctx *config = ctx;
  const arena_offset in_off = sdsge_mc_transform_arena_offset(
      SDSGE_MC_TRANSFORM_LOG_DIFF, config->n, config->p, 0);
  const i64 status =
      sdsge_log_diff(float_in_work, config->offset, config->n, config->p,
                     float_in_work + in_off.foffset[0], float_out);
  return sdsge_mc_transform_status(status, int_out);
}

int sdsge_mc_diff_runner(const i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
                         f64 *SDSGE_RESTRICT float_out,
                         i64 *SDSGE_RESTRICT int_work,
                         i64 *SDSGE_RESTRICT int_out, const void *ctx) {
  (void)rep_idx;
  (void)int_work;
  const sdsge_mc_diff_step_ctx *config = ctx;
  const arena_offset in_off = sdsge_mc_transform_arena_offset(
      SDSGE_MC_TRANSFORM_DIFF, config->n, config->p, config->order);
  const i64 status =
      sdsge_diff(float_in_work, config->order, config->n, config->p,
                 float_in_work + in_off.foffset[0], float_out);
  return sdsge_mc_transform_status(status, int_out);
}

int sdsge_mc_rolling_mean_runner(const i64 rep_idx,
                                 f64 *SDSGE_RESTRICT float_in_work,
                                 f64 *SDSGE_RESTRICT float_out,
                                 i64 *SDSGE_RESTRICT int_work,
                                 i64 *SDSGE_RESTRICT int_out, const void *ctx) {
  (void)rep_idx;
  (void)int_work;
  const sdsge_mc_rolling_mean_step_ctx *config = ctx;
  const arena_offset in_off = sdsge_mc_transform_arena_offset(
      SDSGE_MC_TRANSFORM_ROLLING_MEAN, config->n, config->p, 0);
  const i64 status =
      sdsge_rolling_mean(float_in_work, config->n, config->p, config->window,
                         float_in_work + in_off.foffset[0], float_out);
  return sdsge_mc_transform_status(status, int_out);
}

int sdsge_mc_rolling_var_runner(const i64 rep_idx,
                                f64 *SDSGE_RESTRICT float_in_work,
                                f64 *SDSGE_RESTRICT float_out,
                                i64 *SDSGE_RESTRICT int_work,
                                i64 *SDSGE_RESTRICT int_out, const void *ctx) {
  (void)rep_idx;
  (void)int_work;
  const sdsge_mc_rolling_var_step_ctx *config = ctx;
  const arena_offset in_off = sdsge_mc_transform_arena_offset(
      SDSGE_MC_TRANSFORM_ROLLING_VAR, config->n, config->p, 0);
  const i64 status = sdsge_rolling_var(
      float_in_work, config->n, config->p, config->window, config->ddof,
      float_in_work + in_off.foffset[0], float_out);
  return sdsge_mc_transform_status(status, int_out);
}

int sdsge_mc_rolling_std_runner(const i64 rep_idx,
                                f64 *SDSGE_RESTRICT float_in_work,
                                f64 *SDSGE_RESTRICT float_out,
                                i64 *SDSGE_RESTRICT int_work,
                                i64 *SDSGE_RESTRICT int_out, const void *ctx) {
  (void)rep_idx;
  (void)int_work;
  const sdsge_mc_rolling_std_step_ctx *config = ctx;
  const arena_offset in_off = sdsge_mc_transform_arena_offset(
      SDSGE_MC_TRANSFORM_ROLLING_STD, config->n, config->p, 0);
  const i64 status = sdsge_rolling_std(
      float_in_work, config->n, config->p, config->window, config->ddof,
      float_in_work + in_off.foffset[0], float_out);
  return sdsge_mc_transform_status(status, int_out);
}

int sdsge_mc_user_transform_runner(const i64 rep_idx,
                                   f64 *SDSGE_RESTRICT float_in_work,
                                   f64 *SDSGE_RESTRICT float_out,
                                   i64 *SDSGE_RESTRICT int_work,
                                   i64 *SDSGE_RESTRICT int_out,
                                   const void *ctx) {
  (void)rep_idx;
  (void)int_work;
  const sdsge_mc_user_transform_step_ctx *config = ctx;
  const i64 status = config->fn(float_in_work, float_out, config->n_in,
                                config->p_in, config->n_out, config->p_out);
  return sdsge_mc_transform_status(status, int_out);
}
