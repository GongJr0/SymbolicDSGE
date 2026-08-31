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
#define SDSGE_TRANSFORM_BAD_ARG -1301

/* Static configuration for generic native MC transform dispatch.
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

i64 sdsge_standardize_ax0(const f64 *SDSGE_RESTRICT x, const i64 ddof,
                          const i64 n, const i64 p, f64 *SDSGE_RESTRICT scratch,
                          f64 *SDSGE_RESTRICT out);

i64 sdsge_log(const f64 *SDSGE_RESTRICT x, const f64 offset, const i64 n,
              const i64 p, f64 *SDSGE_RESTRICT out);

i64 sdsge_log_diff(const f64 *SDSGE_RESTRICT x, const f64 offset, const i64 n,
                   const i64 p, f64 *SDSGE_RESTRICT scratch,
                   f64 *SDSGE_RESTRICT out);

i64 sdsge_diff(const f64 *SDSGE_RESTRICT x, const i64 order, const i64 n,
               const i64 p, f64 *SDSGE_RESTRICT scratch,
               f64 *SDSGE_RESTRICT out);

i64 sdsge_rolling_mean(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 p,
                       const i64 window, f64 *SDSGE_RESTRICT scratch,
                       f64 *SDSGE_RESTRICT out);

i64 sdsge_rolling_var(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 p,
                      const i64 window, const i64 ddof,
                      f64 *SDSGE_RESTRICT scratch, f64 *SDSGE_RESTRICT out);

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
