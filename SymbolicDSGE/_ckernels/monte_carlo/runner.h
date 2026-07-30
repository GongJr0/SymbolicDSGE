#ifndef SDSGE_MC_RUNNER_H
#define SDSGE_MC_RUNNER_H

#include "../_common/sdsge_common.h"

/* Generic native Monte Carlo step ABI. Each descriptor owns worker-local
 * input/work and live-output arena rows. ``ctx`` points to immutable,
 * step-specific static configuration owned by the compiled pipeline plan. */
typedef int (*sdsge_mc_step_fn)(i64 rep_idx, f64 *SDSGE_RESTRICT float_in_work,
                                f64 *SDSGE_RESTRICT float_out,
                                i64 *SDSGE_RESTRICT int_work,
                                i64 *SDSGE_RESTRICT int_out, const void *ctx);

/* One compiled float-lane source transfer. ``source_step_idx`` is -1 for a
 * constant fill, less than -1 for immutable static backing, and otherwise
 * identifies an earlier producer's live output lane. Columns and row spans
 * are resolved in Python, while the runner copies only the selected values
 * into the consumer kernel's native input layout. */
typedef struct {
  i64 source_step_idx;
  i64 source_offset;
  i64 source_row_stride;
  i64 row_start;
  i64 n_rows;
  const i64 *columns;
  i64 n_columns;
  i64 target_offset;
  i64 target_row_stride;
  f64 fill_value;
  const f64 *static_source;
  i64 static_rep_stride;
  int static_batched;
} sdsge_mc_float_input_binding;

/* One compiled pipeline step. All arena bases come from Cython-owned NumPy
 * arrays. Worker strides select the temporary row for the executing worker.
 * The runner calls ``fn`` with that row's input/work and live output lanes,
 * then copies those live lanes to a compact retained row when the replication
 * is selected by ``retained_row_by_rep``. */
typedef struct {
  sdsge_mc_step_fn fn;
  f64 *float_in_work;
  i64 *int_in_work;
  f64 *float_live_out;
  i64 *int_live_out;
  f64 *float_retained;
  i64 *int_retained;
  const i64 *retained_row_by_rep;
  i64 float_in_work_worker_stride;
  i64 int_in_work_worker_stride;
  i64 float_live_out_worker_stride;
  i64 int_live_out_worker_stride;
  i64 float_retained_stride;
  i64 int_retained_stride;
  const void *ctx;
  const sdsge_mc_float_input_binding *float_input_bindings;
  i64 n_float_input_bindings;
} sdsge_mc_step_desc;

/* One failure selected by the runner's atomic halt protocol. ``rep_idx`` and
 * ``step_idx`` are -1 before execution. */
typedef struct {
  i64 rep_idx;
  i64 step_idx;
  int status;
} sdsge_mc_failure;

/* ``SDSGE_MC_NOT_RUN`` marks failure-lane entries and integer retained
 * outputs for replications that failed or were skipped by a fail-fast halt.
 * Float retained outputs for the same rows are set to NAN. This prevents
 * NumPy's uninitialized ``empty`` allocations from being observable after the
 * run returns. */
#define SDSGE_MC_NOT_RUN INT64_MIN
#define SDSGE_MC_RUN_OK 0
#define SDSGE_MC_RUN_HALTED 1
#define SDSGE_MC_RUN_BAD_ARG -1

/* Native execution plan. The compiler owns the descriptor array, all step
 * contexts, and every Cython backing array referenced by them for as long as
 * this plan remains live. The runner synchronizes access to ``halt``. A worker
 * that wins the transition from zero to one records ``halt_failure`` and stops
 * new replications when ``fail_fast`` is nonzero.
 *
 * Before returning, the runner sets every retained row belonging to a failed
 * or unfinished replication to defined sentinels. When ``profile_steps`` is
 * nonzero, each profiling array has ``n_workers * n_steps`` entries in
 * worker-major order. The runner clears and writes only the executing
 * worker's row, so profiling adds no synchronization to the hot loop. */

typedef struct {
  const sdsge_mc_step_desc *steps;
  i64 n_steps;
  i64 n_rep;
  i64 n_workers;
  int fail_fast;
  volatile i64 halt;
  sdsge_mc_failure halt_failure;
  i64 *failure_step_by_rep;
  i64 *failure_status_by_rep;
  int profile_steps;
  f64 *step_elapsed_s_by_worker;
  i64 *step_counts_by_worker;
  i64 *step_failures_by_worker;
} sdsge_mc_runner_ctx;

int sdsge_mc_run(sdsge_mc_runner_ctx *runner);

#endif /* SDSGE_MC_RUNNER_H */
