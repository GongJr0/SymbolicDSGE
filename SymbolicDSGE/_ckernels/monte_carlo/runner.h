#ifndef SDSGE_MC_RUNNER_H
#define SDSGE_MC_RUNNER_H

#include "../_common/sdsge_common.h"

/* Generic native Monte Carlo step ABI. The caller supplies one reusable
 * per-replication input/work arena and writes each replication to its own
 * output arena slice. ``ctx`` points to immutable, step-specific static
 * configuration owned by the compiled pipeline plan. */
typedef int (*sdsge_mc_step_fn)(i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
                                f64 *SDSGE_RESTRICT float_out,
                                i64 *SDSGE_RESTRICT int_work,
                                i64 *SDSGE_RESTRICT int_out, const void *ctx);

/* One compiled pipeline step. Output offsets are element offsets from the
 * per-replication output bases. Input and workspace arenas are temporary and
 * reused by every sequential step within a replication. */
typedef struct {
  sdsge_mc_step_fn fn;
  i64 float_out_offset;
  i64 int_out_offset;
  const void *ctx;
} sdsge_mc_step_desc;

/* Immutable native execution plan. The compiler owns the descriptor array and
 * all step contexts, retaining every backing array referenced by a context for
 * at least as long as this plan remains live. Output strides advance the
 * runner from one replication's persistent outputs to the next. */

typedef struct {
  const sdsge_mc_step_desc *steps;
  i64 n_steps;
  i64 float_out_stride;
  i64 int_out_stride;
} sdsge_mc_runner_ctx;

#endif /* SDSGE_MC_RUNNER_H */
