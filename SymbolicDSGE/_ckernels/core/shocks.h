#ifndef SDSGE_SHOCKS_H
#define SDSGE_SHOCKS_H

#include "../_common/sdsge_common.h"

/* Per-replication shock draw, executed inside the MC hot loop (issue #374).
 * Draws into a (T, n_exog) block in the step's own arena, via the Philox engine
 * in ../rng/philox.h, keyed so a replication's draw depends only on `rep_idx`.
 *
 * Student-t, scipy distribution objects, user callables, and literal arrays are
 * not representable here; lowering routes such specs to Python
 * prematerialization instead. */

typedef enum {
  SDSGE_SHOCK_NORMAL = 0,
  SDSGE_SHOCK_UNIFORM = 1,
} native_shock;

/* One resolved entry of a shock spec. A univariate entry is the width-1 case,
 * its `factor` the 1x1 standard deviation. */
typedef struct {
  native_shock family;
  i64 width;
  /* width-long, the exogenous columns this entry drives, in factor order. */
  const i64 *columns;
  /* width x width, row-major, with factor @ factor.T == cov. */
  const f64 *factor;
  /* width-long mean vector */
  const f64 *loc;
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
  i64 max_width;
} sdsge_shock_plan;

/* Scratch elements `sdsge_shock_draw` needs, for arena sizing. */
i64 sdsge_shock_scratch_size(const sdsge_shock_plan *plan);

/* Draw replication `rep_idx` into `out`, a (T, n_exog) row-major block.
 * Untargeted columns are zeroed, so `out` needs no preparation. `scratch` is
 * caller-owned and holds at least `sdsge_shock_scratch_size(plan)` elements;
 * this allocates nothing and is safe to call from every worker concurrently. */
void sdsge_shock_draw(const sdsge_shock_plan *plan, i64 rep_idx,
                      f64 *SDSGE_RESTRICT scratch, f64 *SDSGE_RESTRICT out);

#endif /* sdsge_SHOCKS_H */
