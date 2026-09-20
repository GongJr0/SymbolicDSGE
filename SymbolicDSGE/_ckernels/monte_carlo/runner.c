#include "runner.h"

#include <limits.h>
#include <omp.h>
#include <string.h>

static f64 *worker_float_lane(f64 *base, i64 worker_idx, i64 worker_stride);
static i64 *worker_int_lane(i64 *base, i64 worker_idx, i64 worker_stride);

static int valid_runner(const sdsge_mc_runner_ctx *runner) {
  if (runner == NULL || runner->steps == NULL ||
      runner->step_status_by_rep == NULL || runner->n_steps <= 0 ||
      runner->n_rep <= 0 || runner->n_workers <= 0 ||
      runner->n_workers > INT_MAX) {
    return 0;
  }
  if (runner->profile_steps && (runner->step_elapsed_s_by_worker == NULL ||
                                runner->step_counts_by_worker == NULL ||
                                runner->step_failures_by_worker == NULL)) {
    return 0;
  }

  for (i64 step_idx = 0; step_idx < runner->n_steps; ++step_idx) {
    const sdsge_mc_step_desc *step = runner->steps + step_idx;
    if (step->fn == NULL || step->retained_row_by_rep == NULL ||
        step->float_retained_stride != step->float_live_out_worker_stride ||
        step->int_retained_stride != step->int_live_out_worker_stride ||
        step->float_retained_stride < 0 || step->int_retained_stride < 0 ||
        step->float_in_work_worker_stride < 0 ||
        step->int_in_work_worker_stride < 0 ||
        step->float_live_out_worker_stride < 0 ||
        step->int_live_out_worker_stride < 0 ||
        step->n_float_input_bindings < 0 ||
        (step->n_float_input_bindings > 0 &&
         step->float_input_bindings == NULL)) {
      return 0;
    }
    if ((step->float_retained_stride > 0 && step->float_retained == NULL) ||
        (step->int_retained_stride > 0 && step->int_retained == NULL) ||
        (step->float_in_work_worker_stride > 0 &&
         step->float_in_work == NULL) ||
        (step->int_in_work_worker_stride > 0 && step->int_in_work == NULL) ||
        (step->float_live_out_worker_stride > 0 &&
         step->float_live_out == NULL) ||
        (step->int_live_out_worker_stride > 0 && step->int_live_out == NULL)) {
      return 0;
    }
  }
  return 1;
}

static int materialize_step_inputs(const sdsge_mc_runner_ctx *runner,
                                   const i64 step_idx, const i64 worker_idx,
                                   const i64 rep_idx) {
  const sdsge_mc_step_desc *step = runner->steps + step_idx;
  f64 *target = worker_float_lane(step->float_in_work, worker_idx,
                                  step->float_in_work_worker_stride);

  for (i64 binding_idx = 0; binding_idx < step->n_float_input_bindings;
       ++binding_idx) {
    const sdsge_mc_float_input_binding *binding =
        step->float_input_bindings + binding_idx;
    if (binding->n_rows < 0 || binding->n_columns < 0 ||
        binding->target_offset < 0 || binding->target_row_stride < 0 ||
        binding->target_row_stride < binding->n_columns ||
        (binding->n_rows > 0 &&
         binding->target_offset +
                 (binding->n_rows - 1) * binding->target_row_stride +
                 binding->n_columns >
             step->float_in_work_worker_stride)) {
      return SDSGE_MC_RUN_BAD_ARG;
    }
    if (binding->n_rows == 0 || binding->n_columns == 0) {
      continue;
    }
    if (binding->source_step_idx == -1) {
      for (i64 row = 0; row < binding->n_rows; ++row) {
        f64 *target_row =
            target + binding->target_offset + row * binding->target_row_stride;
        for (i64 column = 0; column < binding->n_columns; ++column) {
          target_row[column] = binding->fill_value;
        }
      }
      continue;
    }

    if (binding->source_offset < 0 || binding->source_row_stride <= 0 ||
        binding->row_start < 0 ||
        (binding->n_columns > 0 && binding->columns == NULL)) {
      return SDSGE_MC_RUN_BAD_ARG;
    }
    const f64 *source_lane;
    if (binding->source_step_idx < -1) {
      if (binding->static_source == NULL || binding->static_rep_stride < 0) {
        return SDSGE_MC_RUN_BAD_ARG;
      }
      source_lane = binding->static_source + binding->source_offset;
      if (binding->static_batched) {
        source_lane += rep_idx * binding->static_rep_stride;
      }
    } else {
      if (binding->source_step_idx >= step_idx) {
        return SDSGE_MC_RUN_BAD_ARG;
      }
      const sdsge_mc_step_desc *source =
          runner->steps + binding->source_step_idx;
      if (binding->source_offset + (binding->row_start + binding->n_rows - 1) *
                                       binding->source_row_stride >=
          source->float_live_out_worker_stride) {
        return SDSGE_MC_RUN_BAD_ARG;
      }
      source_lane = worker_float_lane(source->float_live_out, worker_idx,
                                      source->float_live_out_worker_stride) +
                    binding->source_offset;
    }
    for (i64 row = 0; row < binding->n_rows; ++row) {
      const f64 *source_row =
          source_lane + (binding->row_start + row) * binding->source_row_stride;
      f64 *target_row =
          target + binding->target_offset + row * binding->target_row_stride;
      for (i64 column = 0; column < binding->n_columns; ++column) {
        const i64 source_column = binding->columns[column];
        if (source_column < 0 || source_column >= binding->source_row_stride) {
          return SDSGE_MC_RUN_BAD_ARG;
        }
        target_row[column] = source_row[source_column];
      }
    }
  }
  return SDSGE_MC_RUN_OK;
}

static int halt_requested(const sdsge_mc_runner_ctx *runner) {
  int requested;
#pragma omp critical(sdsge_mc_halt)
  {
    requested = runner->halt != 0;
  }
  return requested;
}

static void record_halt(sdsge_mc_runner_ctx *runner, const i64 rep_idx,
                        const i64 step_idx, const int status) {
#pragma omp critical(sdsge_mc_halt)
  {
    if (runner->halt == 0) {
      runner->halt_failure.rep_idx = rep_idx;
      runner->halt_failure.step_idx = step_idx;
      runner->halt_failure.status = status;
      runner->halt = 1;
    }
  }
}

/* Whether any step this one reads from failed to produce its output.  A failure
 * more than one edge back still arrives, carried by the ``SDSGE_MC_NOT_RUN``
 * the runner writes for every step it declines to run.
 *
 * A binding with a negative ``source_step_idx`` is a constant or a static fill
 * and has no status to carry. A source whose output declares fewer than two int
 * slots has nowhere to state one, and is read as not failed. */
static i64 sources_have_failed(const sdsge_mc_runner_ctx *runner,
                               const sdsge_mc_step_desc *step,
                               const i64 worker_idx) {
  for (i64 binding_idx = 0; binding_idx < step->n_float_input_bindings;
       ++binding_idx) {
    const i64 source_step_idx =
        step->float_input_bindings[binding_idx].source_step_idx;
    if (source_step_idx < 0) {
      continue;
    }
    const sdsge_mc_step_desc *source = runner->steps + source_step_idx;
    if (source->int_live_out_worker_stride < 2) {
      continue;
    }
    const i64 *source_lane = worker_int_lane(
        source->int_live_out, worker_idx, source->int_live_out_worker_stride);
    if (source_lane[1] != SDSGE_MC_RUN_OK) {
      return 1;
    }
  }
  return 0;
}

/* Copy one step's live lanes into its retained row. Every step the runner
 * visits retains, whatever became of it: a step that ran leaves its output and
 * its status, one that failed leaves whatever it got to before failing, and one
 * that was never dispatched leaves the previous replication's floats under a
 * status of ``SDSGE_MC_NOT_RUN``. The row is written unconditionally so that
 * slot 1 of the retained int lane answers for every step of every replication
 * that started, which is what lets the float row be judged one step at a time
 * rather than one replication at a time. */
static void retain_step_output(const sdsge_mc_step_desc *step,
                               const i64 rep_idx, const i64 worker_idx) {
  const i64 retained_row = step->retained_row_by_rep[rep_idx];
  if (retained_row < 0) {
    return;
  }

  if (step->float_retained_stride > 0) {
    memcpy(step->float_retained + retained_row * step->float_retained_stride,
           step->float_live_out +
               worker_idx * step->float_live_out_worker_stride,
           (size_t)step->float_retained_stride * sizeof(f64));
  }
  if (step->int_retained_stride > 0) {
    memcpy(step->int_retained + retained_row * step->int_retained_stride,
           step->int_live_out + worker_idx * step->int_live_out_worker_stride,
           (size_t)step->int_retained_stride * sizeof(i64));
  }
}

/* Withdraw the float output of every step that did not produce one. Each step
 * is judged on its own entry in the status record, so a replication that failed
 * part way keeps the output of every step that ran before it and beside it. The
 * retained int row is left as it stands: both its slots are bookkeeping that
 * the runner and the kernel wrote on purpose, and the failure status is of more
 * use to a reader than a sentinel over the top of it would be.
 *
 * A replication the fail-fast halt dropped ahead of its first step visited no
 * step and wrote no row, so its retained int row holds nothing at all and is
 * filled outright. Its whole status row still reads as the seed, which is what
 * tells it apart from a replication whose steps were merely declined. */
static void sanitize_replication(const sdsge_mc_runner_ctx *runner,
                                 const i64 rep_idx) {
  const i64 *status_row =
      runner->step_status_by_rep + rep_idx * runner->n_steps;
  int clean = 1;
  int never_started = 1;

  for (i64 step_idx = 0; step_idx < runner->n_steps; ++step_idx) {
    if (status_row[step_idx] != SDSGE_MC_RUN_OK) {
      clean = 0;
    }
    if (status_row[step_idx] != SDSGE_MC_NOT_RUN) {
      never_started = 0;
    }
    if (!clean && !never_started) {
      break;
    }
  }
  if (clean) {
    return;
  }

  for (i64 step_idx = 0; step_idx < runner->n_steps; ++step_idx) {
    const sdsge_mc_step_desc *step = runner->steps + step_idx;
    const i64 retained_row = step->retained_row_by_rep[rep_idx];
    if (retained_row < 0 || status_row[step_idx] == SDSGE_MC_RUN_OK) {
      continue;
    }

    if (step->float_retained_stride > 0) {
      f64 *float_row =
          step->float_retained + retained_row * step->float_retained_stride;
      for (i64 index = 0; index < step->float_retained_stride; ++index) {
        float_row[index] = NAN;
      }
    }
    if (never_started && step->int_retained_stride > 0) {
      i64 *int_row =
          step->int_retained + retained_row * step->int_retained_stride;
      for (i64 index = 0; index < step->int_retained_stride; ++index) {
        int_row[index] = SDSGE_MC_NOT_RUN;
      }
    }
  }
}

static f64 *worker_float_lane(f64 *base, const i64 worker_idx,
                              const i64 worker_stride) {
  return worker_stride == 0 ? NULL : base + worker_idx * worker_stride;
}

static i64 *worker_int_lane(i64 *base, const i64 worker_idx,
                            const i64 worker_stride) {
  return worker_stride == 0 ? NULL : base + worker_idx * worker_stride;
}

static void initialize_run_state(sdsge_mc_runner_ctx *runner) {
  runner->halt = 0;
  runner->halt_failure.rep_idx = -1;
  runner->halt_failure.step_idx = -1;
  runner->halt_failure.status = SDSGE_MC_RUN_OK;
  const i64 n_status = runner->n_rep * runner->n_steps;
  for (i64 index = 0; index < n_status; ++index) {
    runner->step_status_by_rep[index] = SDSGE_MC_NOT_RUN;
  }
  if (runner->profile_steps) {
    for (i64 worker_idx = 0; worker_idx < runner->n_workers; ++worker_idx) {
      const i64 offset = worker_idx * runner->n_steps;
      for (i64 step_idx = 0; step_idx < runner->n_steps; ++step_idx) {
        runner->step_elapsed_s_by_worker[offset + step_idx] = 0.0;
        runner->step_counts_by_worker[offset + step_idx] = 0;
        runner->step_failures_by_worker[offset + step_idx] = 0;
      }
    }
  }
}

int sdsge_mc_run(sdsge_mc_runner_ctx *runner) {
  i64 rep_idx;

  if (!valid_runner(runner)) {
    return SDSGE_MC_RUN_BAD_ARG;
  }
  initialize_run_state(runner);

#pragma omp parallel for schedule(static) num_threads((int)runner->n_workers)  \
    private(rep_idx)
  for (rep_idx = 0; rep_idx < runner->n_rep; ++rep_idx) {
    const i64 worker_idx = (i64)omp_get_thread_num();

    if (runner->fail_fast && halt_requested(runner)) {
      continue;
    }

    for (i64 step_idx = 0; step_idx < runner->n_steps; ++step_idx) {
      const sdsge_mc_step_desc *step = runner->steps + step_idx;
      f64 step_started_s = 0.0;
      if (runner->profile_steps) {
        step_started_s = omp_get_wtime();
      }
      int status = SDSGE_MC_RUN_OK;

      /* Slot 0 is the runner's to answer, and it is answered before dispatch
       * so a step with a failed source is never run. The kernel is not told:
       * it reads its own lane for nothing, and a family that wants to act on
       * an inherited failure can start doing so without the runner changing. */
      i64 *int_out = worker_int_lane(step->int_live_out, worker_idx,
                                     step->int_live_out_worker_stride);
      const i64 has_failed_sources =
          sources_have_failed(runner, step, worker_idx);
      const int dispatch = !has_failed_sources;
      int kernel_ran = 0;
      if (int_out != NULL) {
        int_out[0] = has_failed_sources;
      }

      if (dispatch) {
        status = materialize_step_inputs(runner, step_idx, worker_idx, rep_idx);
        if (status == SDSGE_MC_RUN_OK) {
          kernel_ran = 1;
          status =
              step->fn(rep_idx,
                       worker_float_lane(step->float_in_work, worker_idx,
                                         step->float_in_work_worker_stride),
                       worker_float_lane(step->float_live_out, worker_idx,
                                         step->float_live_out_worker_stride),
                       worker_int_lane(step->int_in_work, worker_idx,
                                       step->int_in_work_worker_stride),
                       int_out, step->ctx);
        }
      } else {
        status = SDSGE_MC_NOT_RUN;
      }

      /* Slot 1 is the kernel's to state, and it states one on every path out
       * of ``fn``. A step the runner declined, or one whose bindings would not
       * materialize, never reached ``fn`` and has stated nothing, while the
       * live lane still holds whatever this worker last wrote there. The
       * runner states those two itself.
       *
       * The record takes the same value regardless. It is what outlives the
       * replication: the live lane is the worker's, and the retained lane
       * exists only for a replication the plan chose to keep. */
      if (!kernel_ran && step->int_live_out_worker_stride >= 2) {
        int_out[1] = status;
      }
      runner->step_status_by_rep[rep_idx * runner->n_steps + step_idx] = status;
      retain_step_output(step, rep_idx, worker_idx);

      if (status != SDSGE_MC_RUN_OK) {
        if (runner->profile_steps && dispatch) {
          const i64 offset = worker_idx * runner->n_steps + step_idx;
          runner->step_elapsed_s_by_worker[offset] +=
              omp_get_wtime() - step_started_s;
          runner->step_counts_by_worker[offset] += 1;
          runner->step_failures_by_worker[offset] += 1;
        }
        if (runner->fail_fast) {
          record_halt(runner, rep_idx, step_idx, status);
        }
        continue;
      }

      if (runner->profile_steps && dispatch) {
        const i64 offset = worker_idx * runner->n_steps + step_idx;
        runner->step_elapsed_s_by_worker[offset] +=
            omp_get_wtime() - step_started_s;
        runner->step_counts_by_worker[offset] += 1;
      }
    }
  }

  /* Every replication is swept. A clean one leaves after the scan that proves
   * it clean, which is the same scan a gate on the record would have paid. */
  for (i64 rep_idx = 0; rep_idx < runner->n_rep; ++rep_idx) {
    sanitize_replication(runner, rep_idx);
  }
  return halt_requested(runner) ? SDSGE_MC_RUN_HALTED : SDSGE_MC_RUN_OK;
}
