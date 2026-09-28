#ifndef SDSGE_SHOCKS_H
#define SDSGE_SHOCKS_H

#include "../_common/sdsge_common.h"
#include "../rng/philox.h" /* sdsge_sampler_params */

/* Per-replication shock draw, executed inside the MC hot loop (issue #374).
 * Draws into a (T, n_exog) block via the Philox engine in ../rng/philox.h,
 * keyed so a replication's draw depends only on `rep_idx`. */

typedef enum {
  /* Not a distribution: a prematerialized path, dispatched ahead of the draw
   * table and opening no stream. */
  SDSGE_SHOCK_PATH = -1,
  /* Dense from zero; the draw table in shocks.c is indexed by these. */
  SDSGE_SHOCK_NORMAL = 0,
  SDSGE_SHOCK_UNIFORM = 1,
  SDSGE_SHOCK_STUDENT_T = 2,
  SDSGE_SHOCK_EXPONENTIAL = 3,
  SDSGE_SHOCK_GAMMA = 4,
} native_shock;

/* One resolved entry of a shock spec. A univariate entry is the width-1 case,
 * its `factor` the 1x1 standard deviation. */
typedef struct {
  native_shock family;
  /* (T, width) row-major, set for SDSGE_SHOCK_PATH and NULL otherwise. */
  const f64 *path;
  i64 width;
  /* width-long, the exogenous columns this entry drives, in factor order. */
  const i64 *columns;
  /* width x width, row-major, with factor @ factor.T == cov. */
  const f64 *factor;
  /* width-long mean vector */
  const f64 *loc;
  /* Distribution parameters for families that need them. */
  sdsge_sampler_params params;
  u64 key; /* the spec's seed; columns[0] separates entries sharing one. */
} sdsge_shock_entry;

/* A whole spec, resolved once at lowering and shared read-only by every
 * worker. */
typedef struct {
  const sdsge_shock_entry *entries;
  i64 n_entries;
  i64 T;
  i64 n_exog;
  f64 shock_scale;
} sdsge_shock_plan;

/* One family's whole draw: it seeds its own streams, fills its own scratch
 * layout, and writes `out`. */
typedef void (*sdsge_shock_draw_fn)(const sdsge_shock_plan *plan,
                                    const sdsge_shock_entry *entry, i64 rep_idx,
                                    f64 *SDSGE_RESTRICT scratch,
                                    f64 *SDSGE_RESTRICT out);

arena_size sdsge_shock_entry_arena_size(native_shock family, i64 width, i64 T);
arena_size sdsge_shock_plan_arena_size(const sdsge_shock_plan *plan);

/* Draw replication `rep_idx` into `out`, a (T, n_exog) row-major block. Writes
 * only the columns the spec names, so the caller owns zeroing the rest.
 * `scratch` is caller-owned and holds at least
 * `sdsge_shock_arena_size(plan).n_float` elements. Allocates nothing and is
 * safe to call from every worker concurrently. */
void sdsge_shock_draw(const sdsge_shock_plan *plan, i64 rep_idx,
                      f64 *SDSGE_RESTRICT scratch, f64 *SDSGE_RESTRICT out);

#endif /* sdsge_SHOCKS_H */
