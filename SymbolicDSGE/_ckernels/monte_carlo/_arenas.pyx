# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Dynamic arena allocation for compiled Monte Carlo buffer plans.

The Python planner resolves lane sizes and output layouts.  This module turns
that plan and a run's ``n_rep`` into the contiguous NumPy arrays owned by a
future native runner.  Step-context lowering supplies static data separately.
"""

import os

import numpy as np

from libc.stdint cimport int64_t


cdef class StepArenas:
    """Dynamic arenas and retention metadata for one compiled step."""

    cdef public object float_in_work
    cdef public object int_in_work
    cdef public object float_live_out
    cdef public object int_live_out
    cdef public object float_retained
    cdef public object int_retained
    cdef public object retained_reps
    cdef public object retained_row_by_rep


cdef class ArenaAllocation:
    """All dynamic arena allocations for one Monte Carlo run."""

    cdef public int64_t n_rep
    cdef public int64_t n_workers
    cdef public dict plan
    cdef public dict steps
    cdef public object failure_step_by_rep
    cdef public object failure_status_by_rep


def resolve_retention(int64_t n_retain, int64_t n_rep):
    """Return retained replication indices and their compact-row lookup."""
    cdef int64_t resolved_n_retain
    cdef int64_t row
    cdef object retained_reps
    cdef object retained_row_by_rep

    if n_rep <= 0:
        raise ValueError("n_rep must be positive.")
    if n_retain < -1:
        raise ValueError("n_retain must be -1 (retain all) or non-negative.")
    if n_retain > n_rep:
        raise ValueError("n_retain cannot exceed n_rep.")

    resolved_n_retain = n_rep if n_retain == -1 else n_retain
    retained_reps = np.empty(resolved_n_retain, dtype=np.int64)
    retained_row_by_rep = np.full(n_rep, -1, dtype=np.int64)

    if resolved_n_retain == 1:
        retained_reps[0] = 0
    elif resolved_n_retain > 1:
        for row in range(resolved_n_retain):
            retained_reps[row] = row * (n_rep - 1) // (resolved_n_retain - 1)

    for row in range(resolved_n_retain):
        retained_row_by_rep[retained_reps[row]] = row
    return retained_reps, retained_row_by_rep


def resolve_n_workers(object n_jobs=None):
    """Resolve joblib-style ``n_jobs`` to a positive native worker count."""
    cdef int64_t n_jobs_value
    cdef int64_t cpu_count

    if n_jobs is None:
        return 1
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
        raise TypeError("n_jobs must be an integer or None.")
    n_jobs_value = n_jobs

    if n_jobs_value == 0:
        raise ValueError("n_jobs must not be zero.")
    if n_jobs_value > 0:
        return n_jobs_value

    cpu_count = (
        os.process_cpu_count()
        if hasattr(os, "process_cpu_count")
        else os.cpu_count()
    )
    if cpu_count is None:
        cpu_count = 1

    return max(1, cpu_count + 1 + n_jobs_value)


def allocate_arenas(dict plan, int64_t n_rep, object n_jobs=None):
    """Allocate per-step dynamic arenas for a resolved ``BufferPlan``.

    ``plan`` contains Python ``StepBufferPlan`` instances.  Its input and live
    output lanes have one row per worker; retained lanes are compact,
    replication-major arrays whose row count is resolved from ``n_rep``.
    """
    cdef ArenaAllocation allocation = ArenaAllocation()
    cdef StepArenas step_arenas
    cdef object step_name
    cdef object step_plan
    cdef object retained_reps
    cdef object retained_row_by_rep
    cdef int64_t n_float_in
    cdef int64_t n_int_in
    cdef int64_t n_float_out
    cdef int64_t n_int_out
    cdef int64_t n_retain
    cdef int64_t n_workers

    if n_rep <= 0:
        raise ValueError("n_rep must be positive.")

    allocation.n_rep = n_rep
    n_workers = resolve_n_workers(n_jobs)
    allocation.n_workers = n_workers
    allocation.plan = dict(plan)
    allocation.steps = {}
    allocation.failure_step_by_rep = np.full(
        n_rep, np.iinfo(np.int64).min, dtype=np.int64
    )
    allocation.failure_status_by_rep = np.full(
        n_rep, np.iinfo(np.int64).min, dtype=np.int64
    )
    for step_name, step_plan in allocation.plan.items():
        n_float_in = step_plan.input_size.n_float
        n_int_in = step_plan.input_size.n_int
        n_float_out = step_plan.output_size.n_float
        n_int_out = step_plan.output_size.n_int
        n_retain = step_plan.n_retain
        retained_reps, retained_row_by_rep = resolve_retention(n_retain, n_rep)

        step_arenas = StepArenas()
        step_arenas.float_in_work = np.empty(
            (n_workers, n_float_in), dtype=np.float64
        )
        step_arenas.int_in_work = np.empty((n_workers, n_int_in), dtype=np.int64)
        step_arenas.float_live_out = np.empty(
            (n_workers, n_float_out), dtype=np.float64
        )
        step_arenas.int_live_out = np.empty(
            (n_workers, n_int_out), dtype=np.int64
        )
        step_arenas.float_retained = np.empty(
            (retained_reps.shape[0], n_float_out), dtype=np.float64
        )
        step_arenas.int_retained = np.empty(
            (retained_reps.shape[0], n_int_out), dtype=np.int64
        )
        step_arenas.retained_reps = retained_reps
        step_arenas.retained_row_by_rep = retained_row_by_rep
        allocation.steps[step_name] = step_arenas
    return allocation
