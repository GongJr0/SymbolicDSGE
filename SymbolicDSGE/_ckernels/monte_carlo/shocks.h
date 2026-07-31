#ifndef SDSGE_MC_SHOCKS_H
#define SDSGE_MC_SHOCKS_H

#include "../_common/sdsge_common.h"

/* Per-replication shock draw, executed inside the MC hot loop (issue #374).
 *
 * Shocks used to be materialized in Python before the loop: one numpy
 * Generator constructed per replication per entry, drawn into a
 * (n_rep, T, n_exog) slab that the runner then streamed row by row. At 100k
 * replications that Python prep cost as much as the entire native loop, and
 * the slab alone ran to hundreds of megabytes.
 *
 * Drawing here instead makes the cost O(1) in Python and replaces the slab
 * with a (T, n_exog) block in the step's own arena. The engine is the
 * counter-based Philox in ../rng/philox.h, keyed so that a replication's draw
 * depends only on `rep_idx`: identical no matter which worker runs it, how
 * many workers there are, or in what order they are scheduled.
 *
 * Not every spec can be drawn natively. Student-t, arbitrary scipy
 * distribution objects, user callables, and literal arrays stay on the Python
 * prematerialization route, which the lowering selects when any entry is not
 * representable here. */

#define SDSGE_MC_SHOCK_NORMAL 0
#define SDSGE_MC_SHOCK_UNIFORM 1

/* One resolved entry of a shock spec.
 *
 * `columns` names the exogenous columns this entry drives, in the canonical
 * order its factor was built in. A univariate entry is just the width-1 case:
 * its `factor` is the 1x1 matrix holding the standard deviation, so normal
 * draws take one code path regardless of width.
 *
 * `key` is the spec's own seed and `entry_idx` its position in the spec. The
 * latter is what keeps two entries that happen to share a seed on independent
 * streams. */
typedef struct {
  int family;
  i64 width;
  const i64 *columns;
  /* width x width, row-major, with factor @ factor.T == cov. Normal only. */
  const f64 *factor;
  /* width-long mean vector, or NULL for zeros. Normal only. */
  const f64 *loc;
  /* Uniform only: draws land in [low, low + span). */
  f64 low;
  f64 span;
  u64 key;
  u64 entry_idx;
} sdsge_mc_shock_entry;

/* A whole spec, resolved once at lowering and shared read-only by every
 * worker. `max_width` sizes the scratch the draw needs. */
typedef struct {
  const sdsge_mc_shock_entry *entries;
  i64 n_entries;
  i64 T;
  i64 n_exog;
  f64 shock_scale;
  i64 max_width;
} sdsge_mc_shock_plan;

/* Scratch elements `sdsge_mc_shock_draw` needs, for arena sizing. */
i64 sdsge_mc_shock_scratch_size(const sdsge_mc_shock_plan *plan);

/* Draw replication `rep_idx` into `out`, a (T, n_exog) row-major block.
 *
 * Columns no entry targets are zeroed, so `out` needs no preparation by the
 * caller. `scratch` must hold at least `sdsge_mc_shock_scratch_size(plan)`
 * elements and is caller-owned, which keeps this function allocation-free and
 * therefore safe to call from every worker concurrently. */
void sdsge_mc_shock_draw(const sdsge_mc_shock_plan *plan, i64 rep_idx,
                         f64 *SDSGE_RESTRICT scratch, f64 *SDSGE_RESTRICT out);

#endif /* SDSGE_MC_SHOCKS_H */
