#include "runner.h"

#include <limits.h>
#include <omp.h>
#include <string.h>

static int valid_runner(const sdsge_mc_runner_ctx *runner) {
  if (runner == NULL || runner->steps == NULL ||
      runner->failure_step_by_rep == NULL ||
      runner->failure_status_by_rep == NULL || runner->n_steps <= 0 ||
      runner->n_rep <= 0 || runner->n_workers <= 0 ||
      runner->n_workers > INT_MAX) {
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
        step->int_live_out_worker_stride < 0) {
      return 0;
    }
    if ((step->float_retained_stride > 0 && step->float_retained == NULL) ||
        (step->int_retained_stride > 0 && step->int_retained == NULL) ||
        (step->float_in_work_worker_stride > 0 && step->float_in_work == NULL) ||
        (step->int_in_work_worker_stride > 0 && step->int_in_work == NULL) ||
        (step->float_live_out_worker_stride > 0 &&
         step->float_live_out == NULL) ||
        (step->int_live_out_worker_stride > 0 && step->int_live_out == NULL)) {
      return 0;
    }
  }
  return 1;
}

static int halt_requested(const sdsge_mc_runner_ctx *runner) {
  int requested;
#pragma omp critical(sdsge_mc_halt)
  { requested = runner->halt != 0; }
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

static void sanitize_replication(const sdsge_mc_runner_ctx *runner,
                                 const i64 rep_idx) {
  for (i64 step_idx = 0; step_idx < runner->n_steps; ++step_idx) {
    const sdsge_mc_step_desc *step = runner->steps + step_idx;
    const i64 retained_row = step->retained_row_by_rep[rep_idx];
    if (retained_row < 0) {
      continue;
    }

    if (step->float_retained_stride > 0) {
      f64 *float_row =
          step->float_retained + retained_row * step->float_retained_stride;
      for (i64 index = 0; index < step->float_retained_stride; ++index) {
        float_row[index] = NAN;
      }
    }
    if (step->int_retained_stride > 0) {
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
  for (i64 rep_idx = 0; rep_idx < runner->n_rep; ++rep_idx) {
    runner->failure_step_by_rep[rep_idx] = SDSGE_MC_NOT_RUN;
    runner->failure_status_by_rep[rep_idx] = SDSGE_MC_NOT_RUN;
  }
}

int sdsge_mc_run(sdsge_mc_runner_ctx *runner) {
  i64 rep_idx;

  if (!valid_runner(runner)) {
    return SDSGE_MC_RUN_BAD_ARG;
  }
  initialize_run_state(runner);

#pragma omp parallel for schedule(static) num_threads((int)runner->n_workers) \
    private(rep_idx)
  for (rep_idx = 0; rep_idx < runner->n_rep; ++rep_idx) {
    const i64 worker_idx = (i64)omp_get_thread_num();

    if (runner->fail_fast && halt_requested(runner)) {
      continue;
    }

    int status = SDSGE_MC_RUN_OK;
    for (i64 step_idx = 0; step_idx < runner->n_steps; ++step_idx) {
      const sdsge_mc_step_desc *step = runner->steps + step_idx;
      status = step->fn(
          rep_idx,
          worker_float_lane(step->float_in_work, worker_idx,
                            step->float_in_work_worker_stride),
          worker_float_lane(step->float_live_out, worker_idx,
                            step->float_live_out_worker_stride),
          worker_int_lane(step->int_in_work, worker_idx,
                          step->int_in_work_worker_stride),
          worker_int_lane(step->int_live_out, worker_idx,
                          step->int_live_out_worker_stride),
          step->ctx);
      if (status != SDSGE_MC_RUN_OK) {
        runner->failure_step_by_rep[rep_idx] = step_idx;
        runner->failure_status_by_rep[rep_idx] = status;
        if (runner->fail_fast) {
          record_halt(runner, rep_idx, step_idx, status);
        }
        break;
      }
      retain_step_output(step, rep_idx, worker_idx);
    }

    if (status == SDSGE_MC_RUN_OK) {
      runner->failure_step_by_rep[rep_idx] = -1;
      runner->failure_status_by_rep[rep_idx] = SDSGE_MC_RUN_OK;
    }
  }

  for (i64 rep_idx = 0; rep_idx < runner->n_rep; ++rep_idx) {
    if (runner->failure_step_by_rep[rep_idx] != -1) {
      sanitize_replication(runner, rep_idx);
    }
  }
  return halt_requested(runner) ? SDSGE_MC_RUN_HALTED : SDSGE_MC_RUN_OK;
}
