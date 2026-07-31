#ifndef SDSGE_MC_TRANSFORMS_H
#define SDSGE_MC_TRANSFORMS_H

#include "../_common/sdsge_common.h"
#include "runner.h"

/* Per-replication sample transforms, mirroring
 * `monte_carlo.step_factories`. Every kernel is allocation-free:
 * buffers are caller-owned so the replication loop allocates once at prep.
 * `x` is a row-major (n, p) sample with the time axis first, and `out` never
 * aliases `x` or `scratch`. Output row counts differ per transform and are
 * documented per function; a transform whose output is empty (e.g. `order >=
 * n`) writes nothing and reports success.
 *
 * Scratch buffers are written before they are read, so they need not be zeroed
 * between calls. */

#define SDSGE_TRANSFORM_SUCCESS 0
/* n, p, window, order, or ddof outside the range the transform is defined on.
 */
#define SDSGE_TRANSFORM_BAD_ARG -6

/* Static configuration for future generic native MC transform dispatch.
 * Dynamic sample data, scratch, and output remain in caller-owned arenas. */
typedef struct {
  i64 n;
  i64 p;
  i64 ddof;
} sdsge_mc_standardize_step_ctx;

typedef struct {
  i64 n;
  i64 p;
  f64 offset;
} sdsge_mc_log_step_ctx;

typedef struct {
  i64 n;
  i64 p;
  f64 offset;
} sdsge_mc_log_diff_step_ctx;

typedef struct {
  i64 n;
  i64 p;
  i64 order;
} sdsge_mc_diff_step_ctx;

typedef struct {
  i64 n;
  i64 p;
  i64 window;
} sdsge_mc_rolling_mean_step_ctx;

typedef struct {
  i64 n;
  i64 p;
  i64 window;
  i64 ddof;
} sdsge_mc_rolling_var_step_ctx;

typedef sdsge_mc_rolling_var_step_ctx sdsge_mc_rolling_std_step_ctx;

/* Custom transform callable and context */
typedef i64 (*user_transform_fn)(f64 *SDSGE_RESTRICT inp,
                                 f64 *SDSGE_RESTRICT out, i64 n_in, i64 p_in,
                                 i64 n_out, i64 p_out);

typedef struct {
  user_transform_fn fn;
  i64 n_in;
  i64 p_in;
  i64 n_out;
  i64 p_out;
} sdsge_mc_user_transform_step_ctx;

/* Per-column z-score over axis 0; writes out(n, p). `scratch` is 2*p.
 * Columns whose standard deviation is zero are written as zeros rather than
 * dividing through, matching `run_standardize`. */
arena_size sdsge_standardize_ax0_arena_size(i64 n, i64 p);
i64 sdsge_standardize_ax0(const f64 *SDSGE_RESTRICT x, const i64 ddof,
                          const i64 n, const i64 p, f64 *SDSGE_RESTRICT scratch,
                          f64 *SDSGE_RESTRICT out);

/* log(x + offset) elementwise; writes out(n, p). */
arena_size sdsge_log_arena_size(i64 n, i64 p);
i64 sdsge_log(const f64 *SDSGE_RESTRICT x, const f64 offset, const i64 n,
              const i64 p, f64 *SDSGE_RESTRICT out);

/* One-period log differences down the time axis; writes out(n - 1, p).
 * `scratch` is p. */
arena_size sdsge_log_diff_arena_size(i64 n, i64 p);
i64 sdsge_log_diff(const f64 *SDSGE_RESTRICT x, const f64 offset, const i64 n,
                   const i64 p, f64 *SDSGE_RESTRICT scratch,
                   f64 *SDSGE_RESTRICT out);

/* `order`-th difference down the time axis; writes out(n - order, p).
 * `scratch` is order*p. `order` must be at least 1. */
arena_size sdsge_diff_arena_size(i64 n, i64 p, i64 order);
i64 sdsge_diff(const f64 *SDSGE_RESTRICT x, const i64 order, const i64 n,
               const i64 p, f64 *SDSGE_RESTRICT scratch,
               f64 *SDSGE_RESTRICT out);

/* Trailing rolling mean; writes out(n - window + 1, p). `scratch` is p. */
arena_size sdsge_rolling_mean_arena_size(i64 n, i64 p, i64 window);
i64 sdsge_rolling_mean(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 p,
                       const i64 window, f64 *SDSGE_RESTRICT scratch,
                       f64 *SDSGE_RESTRICT out);

/* Trailing rolling variance; writes out(n - window + 1, p). `scratch` is 2*p.
 */
arena_size sdsge_rolling_var_arena_size(i64 n, i64 p, i64 window);
i64 sdsge_rolling_var(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 p,
                      const i64 window, const i64 ddof,
                      f64 *SDSGE_RESTRICT scratch, f64 *SDSGE_RESTRICT out);

/* Trailing rolling standard deviation; writes out(n - window + 1, p).
 * `scratch` is 2*p. */
arena_size sdsge_rolling_std_arena_size(i64 n, i64 p, i64 window);
i64 sdsge_rolling_std(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 p,
                      const i64 window, const i64 ddof,
                      f64 *SDSGE_RESTRICT scratch, f64 *SDSGE_RESTRICT out);

/* Generic-runner adapters. ``float_in_work`` begins with x(n, p), followed by
 * the transform's scratch span. Each adapter writes its status to ``int_out``
 * when supplied. */
int sdsge_mc_standardize_runner(i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
                                f64 *SDSGE_RESTRICT float_out,
                                i64 *SDSGE_RESTRICT int_work,
                                i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_log_runner(i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
                        f64 *SDSGE_RESTRICT float_out,
                        i64 *SDSGE_RESTRICT int_work,
                        i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_log_diff_runner(i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
                             f64 *SDSGE_RESTRICT float_out,
                             i64 *SDSGE_RESTRICT int_work,
                             i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_diff_runner(i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
                         f64 *SDSGE_RESTRICT float_out,
                         i64 *SDSGE_RESTRICT int_work,
                         i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_rolling_mean_runner(i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
                                 f64 *SDSGE_RESTRICT float_out,
                                 i64 *SDSGE_RESTRICT int_work,
                                 i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_rolling_var_runner(i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
                                f64 *SDSGE_RESTRICT float_out,
                                i64 *SDSGE_RESTRICT int_work,
                                i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_rolling_std_runner(i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
                                f64 *SDSGE_RESTRICT float_out,
                                i64 *SDSGE_RESTRICT int_work,
                                i64 *SDSGE_RESTRICT int_out, const void *ctx);
int sdsge_mc_user_transform_runner(i64 rep_idx,
                                   f64 *SDSGE_RESTRICT float_in_work,
                                   f64 *SDSGE_RESTRICT float_out,
                                   i64 *SDSGE_RESTRICT int_work,
                                   i64 *SDSGE_RESTRICT int_out,
                                   const void *ctx);

#endif /* SDSGE_MC_TRANSFORMS_H */
