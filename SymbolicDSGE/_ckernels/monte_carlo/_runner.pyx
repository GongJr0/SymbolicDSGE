# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Lower allocated Monte Carlo arenas into the native runner ABI.

``ArenaAllocation`` owns all dynamic NumPy buffers. ``NativeStep`` owns the
static C context and any Python array backing that context references. This
module combines both at call time, releases the GIL for the native replication
loop, and leaves results in the allocation's retained arrays.
"""

from typing import NamedTuple

import numpy as np
cimport numpy as cnp

from cpython.mem cimport PyMem_Free, PyMem_Malloc
from libc.stdint cimport int64_t


cdef extern from "runner.h":
    ctypedef int (*sdsge_mc_step_fn)(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil

    ctypedef struct sdsge_mc_step_desc:
        sdsge_mc_step_fn fn
        double *float_in_work
        int64_t *int_in_work
        double *float_live_out
        int64_t *int_live_out
        double *float_retained
        int64_t *int_retained
        const int64_t *retained_row_by_rep
        int64_t float_in_work_worker_stride
        int64_t int_in_work_worker_stride
        int64_t float_live_out_worker_stride
        int64_t int_live_out_worker_stride
        int64_t float_retained_stride
        int64_t int_retained_stride
        const void *ctx

    ctypedef struct sdsge_mc_failure:
        int64_t rep_idx
        int64_t step_idx
        int status

    ctypedef struct sdsge_mc_runner_ctx:
        const sdsge_mc_step_desc *steps
        int64_t n_steps
        int64_t n_rep
        int64_t n_workers
        int fail_fast
        int64_t halt
        sdsge_mc_failure halt_failure
        int64_t *failure_step_by_rep
        int64_t *failure_status_by_rep

    int sdsge_mc_run(sdsge_mc_runner_ctx *runner) noexcept nogil


cdef extern from "core_steps.h":
    ctypedef struct sdsge_mc_payload_step_ctx:
        const double *input
        int64_t n
        int input_batched

    int sdsge_mc_payload_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil


cdef extern from "transforms.h":
    ctypedef struct sdsge_mc_standardize_step_ctx:
        int64_t n
        int64_t p
        int64_t ddof

    ctypedef struct sdsge_mc_log_step_ctx:
        int64_t n
        int64_t p
        double offset

    ctypedef struct sdsge_mc_log_diff_step_ctx:
        int64_t n
        int64_t p
        double offset

    ctypedef struct sdsge_mc_diff_step_ctx:
        int64_t n
        int64_t p
        int64_t order

    ctypedef struct sdsge_mc_rolling_mean_step_ctx:
        int64_t n
        int64_t p
        int64_t window

    ctypedef struct sdsge_mc_rolling_var_step_ctx:
        int64_t n
        int64_t p
        int64_t window
        int64_t ddof

    int sdsge_mc_standardize_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil
    int sdsge_mc_log_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil
    int sdsge_mc_log_diff_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil
    int sdsge_mc_diff_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil
    int sdsge_mc_rolling_mean_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil
    int sdsge_mc_rolling_var_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil
    int sdsge_mc_rolling_std_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil


class NativeRunResult(NamedTuple):
    """Status returned by one native runner invocation."""

    status: int
    halt_rep_idx: int
    halt_step_idx: int
    halt_status: int


cdef class NativeStep:
    """One native step function plus its persistent static C context."""

    cdef sdsge_mc_step_fn _fn
    cdef const void *_ctx
    cdef object _name
    cdef object _backing
    cdef int64_t _n_batch
    cdef int64_t _input_n_float
    cdef int64_t _input_n_int
    cdef int64_t _output_n_float
    cdef int64_t _output_n_int
    cdef sdsge_mc_payload_step_ctx _payload_ctx
    cdef sdsge_mc_standardize_step_ctx _standardize_ctx
    cdef sdsge_mc_log_step_ctx _log_ctx
    cdef sdsge_mc_log_diff_step_ctx _log_diff_ctx
    cdef sdsge_mc_diff_step_ctx _diff_ctx
    cdef sdsge_mc_rolling_mean_step_ctx _rolling_mean_ctx
    cdef sdsge_mc_rolling_var_step_ctx _rolling_var_ctx

    def __cinit__(self):
        self._n_batch = 0
        self._input_n_float = -1
        self._input_n_int = -1
        self._output_n_float = -1
        self._output_n_int = -1

    @property
    def name(self):
        return self._name


cdef cnp.ndarray _require_arena(
    object value,
    object dtype,
    int64_t n_rows,
    str label,
):
    cdef cnp.ndarray array = np.asarray(value)
    if array.dtype != dtype:
        raise TypeError(f"{label} must have dtype {dtype}.")
    if array.ndim != 2:
        raise ValueError(f"{label} must be a two-dimensional array.")
    if array.shape[0] != n_rows:
        raise ValueError(f"{label} has an unexpected row count.")
    if not array.flags.c_contiguous:
        raise ValueError(f"{label} must be C-contiguous.")
    return array


cdef inline double *_float_data(cnp.ndarray array):
    return <double *>cnp.PyArray_DATA(array)


cdef inline int64_t *_int_data(cnp.ndarray array):
    return <int64_t *>cnp.PyArray_DATA(array)


cdef void _check_lanes(
    NativeStep step,
    cnp.ndarray float_in_work,
    cnp.ndarray int_in_work,
    cnp.ndarray float_live_out,
    cnp.ndarray int_live_out,
) except *:
    if step._input_n_float >= 0 and float_in_work.shape[1] != step._input_n_float:
        raise ValueError(f"Step {step.name!r} has an incompatible float input arena.")
    if step._input_n_int >= 0 and int_in_work.shape[1] != step._input_n_int:
        raise ValueError(f"Step {step.name!r} has an incompatible integer input arena.")
    if step._output_n_float >= 0 and float_live_out.shape[1] != step._output_n_float:
        raise ValueError(f"Step {step.name!r} has an incompatible float output arena.")
    if step._output_n_int >= 0 and int_live_out.shape[1] != step._output_n_int:
        raise ValueError(f"Step {step.name!r} has an incompatible "
                         "integer output arena.")


def payload_step(str name, value):
    """Bind a native payload materialization step to immutable input data."""
    cdef NativeStep step = NativeStep()
    cdef cnp.ndarray input_array = np.ascontiguousarray(value, dtype=np.float64)
    cdef int64_t n

    if not name:
        raise ValueError("Native step name must be non-empty.")
    if input_array.ndim not in (1, 2, 3):
        raise ValueError("Payload input must be one-, two-, or three-dimensional.")
    if input_array.ndim == 3:
        if input_array.shape[0] == 0:
            raise ValueError("Batched payload input must contain a replication.")
        n = input_array.size // input_array.shape[0]
        step._n_batch = input_array.shape[0]
    else:
        n = input_array.size

    step._name = name
    step._backing = input_array
    step._payload_ctx.input = <const double *>cnp.PyArray_DATA(input_array)
    step._payload_ctx.n = n
    step._payload_ctx.input_batched = input_array.ndim == 3
    step._fn = sdsge_mc_payload_runner
    step._ctx = <const void *>&step._payload_ctx
    step._input_n_float = 0
    step._input_n_int = 0
    step._output_n_float = n
    step._output_n_int = 0
    return step


def transform_step(
    str name,
    str kind,
    int64_t n,
    int64_t p,
    int64_t ddof=0,
    double offset=0.0,
    int64_t order=1,
    int64_t window=1,
):
    """Bind one native transform adapter using its resolved scalar settings."""
    cdef NativeStep step = NativeStep()
    cdef int64_t input_count
    cdef int64_t output_rows

    if not name:
        raise ValueError("Native step name must be non-empty.")
    if n < 0 or p < 0:
        raise ValueError("Transform dimensions must be non-negative.")

    input_count = n * p
    step._name = name
    step._input_n_int = 0
    step._output_n_int = 0
    if kind == "standardize":
        step._standardize_ctx.n = n
        step._standardize_ctx.p = p
        step._standardize_ctx.ddof = ddof
        step._fn = sdsge_mc_standardize_runner
        step._ctx = <const void *>&step._standardize_ctx
        step._input_n_float = input_count + 2 * p
        step._output_n_float = input_count
    elif kind == "log":
        step._log_ctx.n = n
        step._log_ctx.p = p
        step._log_ctx.offset = offset
        step._fn = sdsge_mc_log_runner
        step._ctx = <const void *>&step._log_ctx
        step._input_n_float = input_count
        step._output_n_float = input_count
    elif kind == "log_diff":
        step._log_diff_ctx.n = n
        step._log_diff_ctx.p = p
        step._log_diff_ctx.offset = offset
        step._fn = sdsge_mc_log_diff_runner
        step._ctx = <const void *>&step._log_diff_ctx
        step._input_n_float = input_count + p
        step._output_n_float = max(0, n - 1) * p
    elif kind == "diff":
        step._diff_ctx.n = n
        step._diff_ctx.p = p
        step._diff_ctx.order = order
        step._fn = sdsge_mc_diff_runner
        step._ctx = <const void *>&step._diff_ctx
        step._input_n_float = input_count + max(0, order) * p
        step._output_n_float = max(0, n - order) * p
    elif kind == "rolling_mean":
        step._rolling_mean_ctx.n = n
        step._rolling_mean_ctx.p = p
        step._rolling_mean_ctx.window = window
        step._fn = sdsge_mc_rolling_mean_runner
        step._ctx = <const void *>&step._rolling_mean_ctx
        step._input_n_float = input_count + p
        output_rows = max(0, n - window + 1)
        step._output_n_float = output_rows * p
    elif kind == "rolling_var" or kind == "rolling_std":
        step._rolling_var_ctx.n = n
        step._rolling_var_ctx.p = p
        step._rolling_var_ctx.window = window
        step._rolling_var_ctx.ddof = ddof
        step._fn = (
            sdsge_mc_rolling_var_runner
            if kind == "rolling_var"
            else sdsge_mc_rolling_std_runner
        )
        step._ctx = <const void *>&step._rolling_var_ctx
        step._input_n_float = input_count + 2 * p
        output_rows = max(0, n - window + 1)
        step._output_n_float = output_rows * p
    else:
        raise ValueError(f"Unsupported native transform kind: {kind!r}.")
    return step


def run(allocation, steps, bint fail_fast=False):
    """Lower ``allocation`` and static bindings, then invoke the native loop.

    Bindings must follow the allocation plan's step order. The allocation keeps
    all NumPy arenas alive during the no-GIL call, and retained outputs remain
    available through ``allocation.steps`` after this function returns.
    """
    cdef int64_t n_steps = len(steps)
    cdef int64_t step_idx
    cdef int status
    cdef object step_names = tuple(allocation.plan)
    cdef object step_arenas
    cdef NativeStep step
    cdef cnp.ndarray float_in_work
    cdef cnp.ndarray int_in_work
    cdef cnp.ndarray float_live_out
    cdef cnp.ndarray int_live_out
    cdef cnp.ndarray float_retained
    cdef cnp.ndarray int_retained
    cdef cnp.ndarray retained_row_by_rep
    cdef cnp.ndarray failure_step_by_rep
    cdef cnp.ndarray failure_status_by_rep
    cdef sdsge_mc_step_desc *descs
    cdef sdsge_mc_runner_ctx runner

    if n_steps == 0:
        raise ValueError("Native runner requires at least one step binding.")
    if n_steps != len(step_names):
        raise ValueError("Native bindings must cover every "
                         "allocated step exactly once.")

    failure_step_by_rep = _require_arena(
        np.asarray(allocation.failure_step_by_rep).reshape(1, -1),
        np.dtype(np.int64),
        1,
        "failure_step_by_rep",
    )
    failure_status_by_rep = _require_arena(
        np.asarray(allocation.failure_status_by_rep).reshape(1, -1),
        np.dtype(np.int64),
        1,
        "failure_status_by_rep",
    )
    if failure_step_by_rep.shape[1] != allocation.n_rep:
        raise ValueError("failure_step_by_rep has an unexpected length.")
    if failure_status_by_rep.shape[1] != allocation.n_rep:
        raise ValueError("failure_status_by_rep has an unexpected length.")

    descs = <sdsge_mc_step_desc *>PyMem_Malloc(
        n_steps * sizeof(sdsge_mc_step_desc)
    )
    if descs == NULL:
        raise MemoryError("Unable to allocate native Monte Carlo descriptors.")
    try:
        for step_idx in range(n_steps):
            step = steps[step_idx]
            if step._fn == NULL or step._ctx == NULL:
                raise ValueError(f"Native step binding at index {step_idx} "
                                 "is incomplete.")
            if step.name != step_names[step_idx]:
                raise ValueError("Native bindings must match the "
                                 "allocation plan order.")
            if step._n_batch and step._n_batch < allocation.n_rep:
                raise ValueError(
                    f"Step {step.name!r} has fewer batched inputs than n_rep."
                )

            step_arenas = allocation.steps[step.name]
            float_in_work = _require_arena(
                step_arenas.float_in_work,
                np.dtype(np.float64),
                allocation.n_workers,
                f"{step.name}.float_in_work",
            )
            int_in_work = _require_arena(
                step_arenas.int_in_work,
                np.dtype(np.int64),
                allocation.n_workers,
                f"{step.name}.int_in_work",
            )
            float_live_out = _require_arena(
                step_arenas.float_live_out,
                np.dtype(np.float64),
                allocation.n_workers,
                f"{step.name}.float_live_out",
            )
            int_live_out = _require_arena(
                step_arenas.int_live_out,
                np.dtype(np.int64),
                allocation.n_workers,
                f"{step.name}.int_live_out",
            )
            float_retained = _require_arena(
                step_arenas.float_retained,
                np.dtype(np.float64),
                step_arenas.retained_reps.shape[0],
                f"{step.name}.float_retained",
            )
            int_retained = _require_arena(
                step_arenas.int_retained,
                np.dtype(np.int64),
                step_arenas.retained_reps.shape[0],
                f"{step.name}.int_retained",
            )
            retained_row_by_rep = np.asarray(step_arenas.retained_row_by_rep)
            if (
                retained_row_by_rep.dtype != np.dtype(np.int64)
                or retained_row_by_rep.ndim != 1
                or retained_row_by_rep.shape[0] != allocation.n_rep
                or not retained_row_by_rep.flags.c_contiguous
            ):
                raise ValueError(
                    f"{step.name}.retained_row_by_rep must be a "
                    "contiguous int64 vector."
                )
            _check_lanes(
                step,
                float_in_work,
                int_in_work,
                float_live_out,
                int_live_out,
            )
            if (
                float_live_out.shape[1] != float_retained.shape[1]
                or int_live_out.shape[1] != int_retained.shape[1]
            ):
                raise ValueError(f"Step {step.name!r} has mismatched "
                                 "live and retained lanes.")

            descs[step_idx].fn = step._fn
            descs[step_idx].float_in_work = _float_data(float_in_work)
            descs[step_idx].int_in_work = _int_data(int_in_work)
            descs[step_idx].float_live_out = _float_data(float_live_out)
            descs[step_idx].int_live_out = _int_data(int_live_out)
            descs[step_idx].float_retained = _float_data(float_retained)
            descs[step_idx].int_retained = _int_data(int_retained)
            descs[step_idx].retained_row_by_rep = _int_data(retained_row_by_rep)
            descs[step_idx].float_in_work_worker_stride = float_in_work.shape[1]
            descs[step_idx].int_in_work_worker_stride = int_in_work.shape[1]
            descs[step_idx].float_live_out_worker_stride = float_live_out.shape[1]
            descs[step_idx].int_live_out_worker_stride = int_live_out.shape[1]
            descs[step_idx].float_retained_stride = float_retained.shape[1]
            descs[step_idx].int_retained_stride = int_retained.shape[1]
            descs[step_idx].ctx = step._ctx

        runner.steps = descs
        runner.n_steps = n_steps
        runner.n_rep = allocation.n_rep
        runner.n_workers = allocation.n_workers
        runner.fail_fast = fail_fast
        runner.halt = 0
        runner.halt_failure.rep_idx = -1
        runner.halt_failure.step_idx = -1
        runner.halt_failure.status = 0
        runner.failure_step_by_rep = _int_data(failure_step_by_rep)
        runner.failure_status_by_rep = _int_data(failure_status_by_rep)
        with nogil:
            status = sdsge_mc_run(&runner)
        return NativeRunResult(
            status,
            runner.halt_failure.rep_idx,
            runner.halt_failure.step_idx,
            runner.halt_failure.status,
        )
    finally:
        PyMem_Free(descs)
