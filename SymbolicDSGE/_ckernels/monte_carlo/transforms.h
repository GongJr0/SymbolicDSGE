#ifndef SDSGE_MC_TRANSFORMS_H
#define SDSGE_MC_TRANSFORMS_H

#include "../_common/sdsge_common.h"

/* Per-replication sample transforms, mirroring
 * `monte_carlo.operations.transforms.ops`. Every kernel is allocation-free:
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

/* Per-column z-score over axis 0; writes out(n, p). `scratch` is 2*p.
 * Columns whose standard deviation is zero are written as zeros rather than
 * dividing through, matching `run_standardize`. */
i64 sdsge_standardize_ax0(const f64 *SDSGE_RESTRICT x, const i64 ddof,
                          const i64 n, const i64 p, f64 *SDSGE_RESTRICT scratch,
                          f64 *SDSGE_RESTRICT out);

/* log(x + offset) elementwise; writes out(n, p). */
i64 sdsge_log(const f64 *SDSGE_RESTRICT x, const f64 offset, const i64 n,
              const i64 p, f64 *SDSGE_RESTRICT out);

/* One-period log differences down the time axis; writes out(n - 1, p).
 * `scratch` is p. */
i64 sdsge_log_diff(const f64 *SDSGE_RESTRICT x, const f64 offset, const i64 n,
                   const i64 p, f64 *SDSGE_RESTRICT scratch,
                   f64 *SDSGE_RESTRICT out);

/* `order`-th difference down the time axis; writes out(n - order, p).
 * `scratch` is order*p. `order` must be at least 1. */
i64 sdsge_diff(const f64 *SDSGE_RESTRICT x, const i64 order, const i64 n,
               const i64 p, f64 *SDSGE_RESTRICT scratch,
               f64 *SDSGE_RESTRICT out);

/* Trailing rolling mean; writes out(n - window + 1, p). `scratch` is p. */
i64 sdsge_rolling_mean(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 p,
                       const i64 window, f64 *SDSGE_RESTRICT scratch,
                       f64 *SDSGE_RESTRICT out);

/* Trailing rolling variance; writes out(n - window + 1, p). `scratch` is 2*p.
 */
i64 sdsge_rolling_var(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 p,
                      const i64 window, const i64 ddof,
                      f64 *SDSGE_RESTRICT scratch, f64 *SDSGE_RESTRICT out);

/* Trailing rolling standard deviation; writes out(n - window + 1, p).
 * `scratch` is 2*p. */
i64 sdsge_rolling_std(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 p,
                      const i64 window, const i64 ddof,
                      f64 *SDSGE_RESTRICT scratch, f64 *SDSGE_RESTRICT out);

#endif /* SDSGE_MC_TRANSFORMS_H */
