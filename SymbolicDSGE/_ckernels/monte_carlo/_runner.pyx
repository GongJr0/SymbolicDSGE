# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Lower allocated Monte Carlo arenas into the native runner ABI.

``ArenaAllocation`` owns all dynamic NumPy buffers. ``NativeStep`` owns the
static C context and any Python array backing that context references. This
module combines both at call time, releases the GIL for the native replication
loop, and leaves results in the allocation's retained arrays.
"""

from time import perf_counter
from typing import NamedTuple

import numpy as np
cimport numpy as cnp

from cpython.mem cimport PyMem_Free, PyMem_Malloc
from libc.stdint cimport int64_t, uint64_t, uintptr_t

from SymbolicDSGE._diag_tests.distributions import ReferenceDistribution


cdef extern from "shocks.h":
    int SDSGE_MC_SHOCK_NORMAL
    int SDSGE_MC_SHOCK_UNIFORM

    ctypedef struct sdsge_mc_shock_entry:
        int family
        int64_t width
        const int64_t *columns
        const double *factor
        const double *loc
        double low
        double span
        uint64_t key
        uint64_t entry_idx

    ctypedef struct sdsge_mc_shock_plan:
        const sdsge_mc_shock_entry *entries
        int64_t n_entries
        int64_t T
        int64_t n_exog
        double shock_scale
        int64_t max_width

    int64_t sdsge_mc_shock_scratch_size(
        const sdsge_mc_shock_plan *plan,
    ) noexcept nogil

    void sdsge_mc_shock_draw(
        const sdsge_mc_shock_plan *plan,
        int64_t rep_idx,
        double *scratch,
        double *out,
    ) noexcept nogil


cdef extern from "runner.h":
    ctypedef int (*sdsge_mc_step_fn)(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil

    ctypedef struct sdsge_mc_float_input_binding:
        int64_t source_step_idx
        int64_t source_offset
        int64_t source_row_stride
        int64_t row_start
        int64_t n_rows
        const int64_t *columns
        int64_t n_columns
        int64_t target_offset
        int64_t target_row_stride
        double fill_value
        const double *static_source
        int64_t static_rep_stride
        int static_batched

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
        const sdsge_mc_float_input_binding *float_input_bindings
        int64_t n_float_input_bindings

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
        int profile_steps
        double *step_elapsed_s_by_worker
        int64_t *step_counts_by_worker
        int64_t *step_failures_by_worker

    int sdsge_mc_run(sdsge_mc_runner_ctx *runner) noexcept nogil


cdef extern from "core_steps.h":
    ctypedef void (*sdsge_measurement_fn)(
        double *vars,
        double *par,
        double *out,
    ) noexcept nogil

    ctypedef void (*meas_fn)(
        const double *vars,
        const double *par,
        double *out,
    ) noexcept nogil

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

    ctypedef struct sdsge_mc_passthrough_step_ctx:
        int64_t n
        int64_t p

    int sdsge_mc_passthrough_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil

    ctypedef struct sdsge_mc_raw_model_data_step_ctx:
        const double *states_input
        int64_t n_states
        int states_batched
        const double *observables_input
        int64_t n_observables
        int observables_batched

    int sdsge_mc_raw_model_data_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil

    ctypedef struct sdsge_mc_simulate_order1_step_ctx:
        sdsge_measurement_fn measurement
        int64_t T
        int64_t n
        int64_t k
        int64_t n_par
        int64_t m
        const sdsge_mc_shock_plan *shocks
        int64_t shock_scratch_offset

    ctypedef struct sdsge_mc_simulate_order2_step_ctx:
        sdsge_measurement_fn measurement
        int64_t T
        int64_t n_state
        int64_t n_ctrl
        int64_t n_exog
        int64_t n_par
        int64_t m
        const sdsge_mc_shock_plan *shocks
        int64_t shock_scratch_offset

    int64_t sdsge_simulate_order1_arena_size(
        int64_t n,
        int64_t k,
        int64_t T,
        int64_t n_par,
    ) noexcept nogil

    int64_t sdsge_simulate_order2_arena_size(
        int64_t n_state,
        int64_t n_var,
        int64_t n_exog,
        int64_t T,
        int64_t n_par,
    ) noexcept nogil

    int sdsge_mc_simulate_order1_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil

    int sdsge_mc_simulate_order2_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil

    ctypedef struct sdsge_mc_filter_linear_step_ctx:
        int64_t T
        int64_t n
        int64_t m
        int64_t k
        int symmetrize
        int joseph_cov
        double jitter
        int return_shocks

    ctypedef struct sdsge_mc_filter_extended_step_ctx:
        meas_fn measurement
        meas_fn jacobian
        int64_t T
        int64_t n
        int64_t m
        int64_t k
        int64_t n_par
        int symmetrize
        int joseph_cov
        double jitter
        int return_shocks

    ctypedef struct sdsge_mc_filter_unscented_step_ctx:
        meas_fn measurement
        int64_t T
        int64_t n_state
        int64_t n_ctrl
        int64_t n_exog
        int64_t n_obs
        int64_t n_par
        double alpha
        double beta
        double kappa
        int symmetrize
        double jitter

    int64_t sdsge_filter_linear_output_arena_size(
        int64_t n,
        int64_t m,
        int64_t k,
        int64_t T,
        int return_shocks,
    ) noexcept nogil

    int64_t sdsge_filter_extended_output_arena_size(
        int64_t n,
        int64_t m,
        int64_t k,
        int64_t T,
        int return_shocks,
    ) noexcept nogil

    int64_t sdsge_filter_unscented_output_arena_size(
        int64_t n_state,
        int64_t n_ctrl,
        int64_t n_obs,
        int64_t T,
    ) noexcept nogil

    int sdsge_mc_filter_linear_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil

    int sdsge_mc_filter_extended_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil

    int sdsge_mc_filter_unscented_runner(
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

    ctypedef int64_t (*user_transform_fn)(
            double *inp,
            double *out,
            int64_t n_in, int64_t p_in,
            int64_t n_out, int64_t p_out,
            )

    ctypedef struct sdsge_mc_user_transform_step_ctx:
        user_transform_fn fn
        int64_t n_in
        int64_t p_in
        int64_t n_out
        int64_t p_out

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
    int sdsge_mc_user_transform_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil

cdef extern from "regression.h":
    ctypedef struct sdsge_mc_ols_step_ctx:
        int64_t n
        int64_t p
        int intercept

    ctypedef struct sdsge_mc_ridge_step_ctx:
        int64_t n
        int64_t p
        int intercept
        double alpha

    ctypedef struct sdsge_mc_ridge_gs_step_ctx:
        const double *alphas
        int64_t n
        int64_t p
        int64_t n_alpha
        int64_t criterion
        int intercept

    ctypedef struct sdsge_mc_lasso_step_ctx:
        int64_t n
        int64_t p
        int intercept
        int64_t max_iter
        double tol
        double alpha

    ctypedef struct sdsge_mc_lasso_gs_step_ctx:
        const double *alphas
        int64_t n
        int64_t p
        int64_t n_alpha
        int intercept
        int64_t max_iter
        double tol

    ctypedef struct sdsge_mc_elastic_net_step_ctx:
        int64_t n
        int64_t p
        int intercept
        int64_t max_iter
        double tol
        double alpha
        double l1_ratio

    ctypedef struct sdsge_mc_elastic_net_gs_step_ctx:
        const double *alphas
        int64_t n
        int64_t p
        int64_t n_alpha
        int64_t criterion
        int intercept
        int64_t max_iter
        double tol
        double l1_ratio

    int sdsge_mc_ols_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil

    int sdsge_mc_ridge_runner(
        int64_t, double *, double *, int64_t *, int64_t *, const void *
    ) noexcept nogil
    int sdsge_mc_ridge_gs_runner(
        int64_t, double *, double *, int64_t *, int64_t *, const void *
    ) noexcept nogil
    int sdsge_mc_lasso_runner(
        int64_t, double *, double *, int64_t *, int64_t *, const void *
    ) noexcept nogil
    int sdsge_mc_lasso_gs_runner(
        int64_t, double *, double *, int64_t *, int64_t *, const void *
    ) noexcept nogil
    int sdsge_mc_elastic_net_runner(
        int64_t, double *, double *, int64_t *, int64_t *, const void *
    ) noexcept nogil
    int sdsge_mc_elastic_net_gs_runner(
        int64_t, double *, double *, int64_t *, int64_t *, const void *
    ) noexcept nogil


cdef extern from "tests.h":
    ctypedef struct sdsge_mc_wald_test_ctx:
        const double *target
        int64_t n
        int64_t q
        int64_t manual_bandwidth
        int kernel_id
        int bandwidth_mode
        int kind

    ctypedef struct sdsge_mc_ljung_box_test_ctx:
        int64_t n
        int64_t lags

    ctypedef struct sdsge_mc_jarque_bera_test_ctx:
        int64_t n

    ctypedef struct sdsge_mc_breusch_pagan_test_ctx:
        int64_t n
        int64_t k
        int robust

    ctypedef struct sdsge_mc_breusch_godfrey_test_ctx:
        int64_t n
        int64_t k
        int64_t lags

    ctypedef struct sdsge_mc_cusum_test_ctx:
        int64_t n
        int64_t p

    ctypedef sdsge_mc_cusum_test_ctx sdsge_mc_cusumsq_test_ctx

    ctypedef struct sdsge_mc_chow_test_ctx:
        int64_t n
        int64_t p
        int64_t t_break

    int sdsge_mc_jarque_bera_test_runner(
        int64_t rep_idx,
        double *float_in_work,
        double *float_out,
        int64_t *int_work,
        int64_t *int_out,
        const void *ctx,
    ) noexcept nogil

    int sdsge_mc_wald_test_runner(
        int64_t, double *, double *, int64_t *, int64_t *, const void *
    ) noexcept nogil
    int sdsge_mc_ljung_box_test_runner(
        int64_t, double *, double *, int64_t *, int64_t *, const void *
    ) noexcept nogil
    int sdsge_mc_breusch_pagan_test_runner(
        int64_t, double *, double *, int64_t *, int64_t *, const void *
    ) noexcept nogil
    int sdsge_mc_breusch_godfrey_test_runner(
        int64_t, double *, double *, int64_t *, int64_t *, const void *
    ) noexcept nogil
    int sdsge_mc_cusum_test_runner(
        int64_t, double *, double *, int64_t *, int64_t *, const void *
    ) noexcept nogil
    int sdsge_mc_cusumsq_test_runner(
        int64_t, double *, double *, int64_t *, int64_t *, const void *
    ) noexcept nogil
    int sdsge_mc_chow_test_runner(
        int64_t, double *, double *, int64_t *, int64_t *, const void *
    ) noexcept nogil


class NativeRunResult(NamedTuple):
    """Status returned by one native runner invocation."""

    status: int
    halt_rep_idx: int
    halt_step_idx: int
    halt_status: int
    wall_elapsed_s: float
    step_elapsed_s_by_worker: object
    step_counts_by_worker: object
    step_failures_by_worker: object


cdef class NativeShockPlan:
    """A shock spec resolved into the layout the native draw reads.

    Owns the C entry array and holds a reference to every NumPy buffer its
    entries point at, so the plan is the single lifetime anchor: a simulation
    step keeps one of these alive for as long as its context references it.

    The plan is immutable and shared read-only across workers. Nothing here is
    touched during the run, which is what lets the draw be reentrant.
    """

    cdef sdsge_mc_shock_plan _plan
    cdef sdsge_mc_shock_entry *_entries
    cdef object _backing

    def __cinit__(self):
        self._entries = NULL
        self._backing = []

    def __dealloc__(self):
        if self._entries != NULL:
            PyMem_Free(self._entries)
            self._entries = NULL

    cdef const sdsge_mc_shock_plan *c_plan(self) noexcept:
        return &self._plan

    @property
    def scratch_size(self):
        """Extra float arena elements the draw needs, for step sizing."""
        return sdsge_mc_shock_scratch_size(&self._plan)

    @property
    def n_entries(self):
        return self._plan.n_entries

    def draw(self, int64_t rep_idx):
        """Materialize one replication's ``(T, n_exog)`` shock block.

        The run never calls this; it draws straight into its arena. This is the
        route back out for a caller holding a replication index and wanting the
        exact block that replication saw, which is what makes a single
        replication reproducible outside the loop.
        """
        if rep_idx < 0:
            raise ValueError("rep_idx must be non-negative.")
        cdef double[:, ::1] out = np.zeros(
            (self._plan.T, self._plan.n_exog), dtype=np.float64
        )
        cdef double[::1] scratch = np.empty(
            max(sdsge_mc_shock_scratch_size(&self._plan), 1), dtype=np.float64
        )
        with nogil:
            sdsge_mc_shock_draw(&self._plan, rep_idx, &scratch[0], &out[0, 0])
        return np.asarray(out)


def shock_plan(
    list entries,
    int64_t T,
    int64_t n_exog,
    double shock_scale,
):
    """Build a native shock plan from resolved entries.

    Each entry is ``(family, columns, factor, loc, low, span, key)``. ``family``
    is one of the ``SHOCK_*`` constants below. ``columns`` is the int64 array of
    exogenous column indices the entry drives, in the order its ``factor`` was
    built in. ``factor`` is the row-major ``(width, width)`` matrix with
    ``factor @ factor.T`` equal to the covariance (a 1x1 holding the standard
    deviation in the univariate case) and is required for normal entries.
    ``loc`` is an optional width-long mean vector. ``low`` and ``span`` apply to
    uniform entries only.

    An entry's position in this list becomes its stream selector, so two entries
    sharing a seed still draw independently.
    """
    cdef NativeShockPlan plan = NativeShockPlan()
    cdef int64_t n = len(entries)
    cdef int64_t i
    cdef int64_t width
    cdef int64_t max_width = 0
    cdef int family
    cdef cnp.ndarray columns_arr
    cdef cnp.ndarray factor_arr
    cdef cnp.ndarray loc_arr

    if T < 0 or n_exog < 0:
        raise ValueError("Native shock plan dimensions must be non-negative.")
    if n <= 0:
        raise ValueError("Native shock plan requires at least one entry.")

    plan._entries = <sdsge_mc_shock_entry *>PyMem_Malloc(
        <size_t>n * sizeof(sdsge_mc_shock_entry)
    )
    if plan._entries == NULL:
        raise MemoryError("Could not allocate native shock entries.")

    for i in range(n):
        family, columns, factor, loc, low, span, key = entries[i]

        columns_arr = np.ascontiguousarray(columns, dtype=np.int64)
        if columns_arr.ndim != 1 or columns_arr.shape[0] == 0:
            raise ValueError("Shock entry columns must be a non-empty 1D array.")
        width = columns_arr.shape[0]
        if (columns_arr < 0).any() or (columns_arr >= n_exog).any():
            raise ValueError("Shock entry columns must index the exogenous block.")
        plan._backing.append(columns_arr)

        if family == SDSGE_MC_SHOCK_UNIFORM:
            if width != 1:
                raise ValueError("Uniform shock entries must be univariate.")
            plan._entries[i].factor = NULL
            plan._entries[i].loc = NULL
        elif family == SDSGE_MC_SHOCK_NORMAL:
            if factor is None:
                raise ValueError("Normal shock entries require a factor matrix.")
            factor_arr = np.ascontiguousarray(factor, dtype=np.float64)
            if factor_arr.ndim != 2 or factor_arr.shape[0] != width \
                    or factor_arr.shape[1] != width:
                raise ValueError(
                    "Shock entry factor must be square and match its width."
                )
            plan._backing.append(factor_arr)
            plan._entries[i].factor = <const double *>cnp.PyArray_DATA(factor_arr)
            if loc is None:
                plan._entries[i].loc = NULL
            else:
                loc_arr = np.ascontiguousarray(loc, dtype=np.float64)
                if loc_arr.ndim != 1 or loc_arr.shape[0] != width:
                    raise ValueError("Shock entry loc must match its width.")
                plan._backing.append(loc_arr)
                plan._entries[i].loc = <const double *>cnp.PyArray_DATA(loc_arr)
        else:
            raise ValueError(f"Unsupported native shock family: {family!r}.")

        plan._entries[i].family = family
        plan._entries[i].width = width
        plan._entries[i].columns = <const int64_t *>cnp.PyArray_DATA(columns_arr)
        plan._entries[i].low = low
        plan._entries[i].span = span
        plan._entries[i].key = <uint64_t>key
        # Position in the spec, so entries sharing a seed stay independent.
        plan._entries[i].entry_idx = <uint64_t>i

        if width > max_width:
            max_width = width

    plan._plan.entries = plan._entries
    plan._plan.n_entries = n
    plan._plan.T = T
    plan._plan.n_exog = n_exog
    plan._plan.shock_scale = shock_scale
    plan._plan.max_width = max_width
    return plan


SHOCK_NORMAL = SDSGE_MC_SHOCK_NORMAL
SHOCK_UNIFORM = SDSGE_MC_SHOCK_UNIFORM


cdef class NativeStep:
    """One native step function plus its persistent static C context."""

    cdef sdsge_mc_step_fn _fn
    cdef const void *_ctx
    cdef object _name
    cdef object _backing
    cdef object _test_distribution
    cdef object _test_df
    cdef int64_t _n_batch
    cdef int64_t _input_n_float
    cdef int64_t _input_n_int
    cdef int64_t _output_n_float
    cdef int64_t _output_n_int
    cdef sdsge_mc_payload_step_ctx _payload_ctx
    cdef sdsge_mc_raw_model_data_step_ctx _raw_model_data_ctx
    cdef sdsge_mc_simulate_order1_step_ctx _simulate_order1_ctx
    cdef sdsge_mc_simulate_order2_step_ctx _simulate_order2_ctx
    cdef sdsge_mc_filter_linear_step_ctx _filter_linear_ctx
    cdef sdsge_mc_filter_extended_step_ctx _filter_extended_ctx
    cdef sdsge_mc_filter_unscented_step_ctx _filter_unscented_ctx
    cdef sdsge_mc_passthrough_step_ctx _passthrough_ctx
    cdef sdsge_mc_standardize_step_ctx _standardize_ctx
    cdef sdsge_mc_log_step_ctx _log_ctx
    cdef sdsge_mc_log_diff_step_ctx _log_diff_ctx
    cdef sdsge_mc_diff_step_ctx _diff_ctx
    cdef sdsge_mc_rolling_mean_step_ctx _rolling_mean_ctx
    cdef sdsge_mc_rolling_var_step_ctx _rolling_var_ctx
    cdef sdsge_mc_ols_step_ctx _ols_ctx
    cdef sdsge_mc_ridge_step_ctx _ridge_ctx
    cdef sdsge_mc_ridge_gs_step_ctx _ridge_gs_ctx
    cdef sdsge_mc_lasso_step_ctx _lasso_ctx
    cdef sdsge_mc_lasso_gs_step_ctx _lasso_gs_ctx
    cdef sdsge_mc_elastic_net_step_ctx _elastic_net_ctx
    cdef sdsge_mc_elastic_net_gs_step_ctx _elastic_net_gs_ctx
    cdef sdsge_mc_wald_test_ctx _wald_ctx
    cdef sdsge_mc_ljung_box_test_ctx _ljung_box_ctx
    cdef sdsge_mc_jarque_bera_test_ctx _jarque_bera_ctx
    cdef sdsge_mc_breusch_pagan_test_ctx _breusch_pagan_ctx
    cdef sdsge_mc_breusch_godfrey_test_ctx _breusch_godfrey_ctx
    cdef sdsge_mc_cusum_test_ctx _cusum_ctx
    cdef sdsge_mc_cusumsq_test_ctx _cusumsq_ctx
    cdef sdsge_mc_chow_test_ctx _chow_ctx
    cdef sdsge_mc_user_transform_step_ctx _user_transform_ctx

    def __cinit__(self):
        self._test_distribution = None
        self._test_df = None
        self._n_batch = 0
        self._input_n_float = -1
        self._input_n_int = -1
        self._output_n_float = -1
        self._output_n_int = -1

    cdef void _bind(
        self,
        str name,
        sdsge_mc_step_fn fn,
        const void *ctx,
        object backing,
        int64_t n_batch,
        int64_t input_n_float,
        int64_t input_n_int,
        int64_t output_n_float,
        int64_t output_n_int,
    ) except *:
        """Finish one typed context after its factory has initialized it."""
        if not name:
            raise ValueError("Native step name must be non-empty.")
        if fn == NULL or ctx == NULL:
            raise ValueError("Native steps require a function and context.")
        if n_batch < 0:
            raise ValueError("Native step batch count must be non-negative.")
        self._name = name
        self._fn = fn
        self._ctx = ctx
        self._backing = backing
        self._n_batch = n_batch
        self._input_n_float = input_n_float
        self._input_n_int = input_n_int
        self._output_n_float = output_n_float
        self._output_n_int = output_n_int

    @property
    def name(self):
        return self._name

    @property
    def test_distribution(self):
        return self._test_distribution

    @property
    def test_df(self):
        return self._test_df


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

    if input_array.ndim not in (1, 2, 3):
        raise ValueError("Payload input must be one-, two-, or three-dimensional.")
    if input_array.ndim == 3:
        if input_array.shape[0] == 0:
            raise ValueError("Batched payload input must contain a replication.")
        n = input_array.size // input_array.shape[0]
        step._n_batch = input_array.shape[0]
    else:
        n = input_array.size

    step._payload_ctx.input = <const double *>cnp.PyArray_DATA(input_array)
    step._payload_ctx.n = n
    step._payload_ctx.input_batched = input_array.ndim == 3
    step._bind(
        name,
        sdsge_mc_payload_runner,
        <const void *>&step._payload_ctx,
        input_array,
        step._n_batch,
        0,
        0,
        n,
        0,
    )
    return step


def raw_model_data_step(str name, states=None, observables=None):
    """Bind native raw data materialization with optional batched inputs."""
    cdef NativeStep step = NativeStep()
    cdef cnp.ndarray states_array
    cdef cnp.ndarray observables_array
    cdef int64_t n_states = 0
    cdef int64_t n_observables = 0
    cdef int64_t n_batch = 0
    cdef int states_batched = 0
    cdef int observables_batched = 0

    if states is None and observables is None:
        raise ValueError("Raw model data requires states or observables.")

    if states is not None:
        states_array = np.ascontiguousarray(states, dtype=np.float64)
        if states_array.ndim not in (1, 2, 3):
            raise ValueError("Raw states must be 1D, 2D, or 3D.")
        states_batched = states_array.ndim == 3
        n_states = (
            states_array.size // states_array.shape[0]
            if states_batched else states_array.size
        )
        n_batch = states_array.shape[0] if states_batched else 0
        step._raw_model_data_ctx.states_input = <const double *>cnp.PyArray_DATA(
            states_array
        )
    else:
        states_array = None
        step._raw_model_data_ctx.states_input = NULL

    if observables is not None:
        observables_array = np.ascontiguousarray(observables, dtype=np.float64)
        if observables_array.ndim not in (1, 2, 3):
            raise ValueError(
                "Raw observables must be 1D, 2D, or 3D."
            )
        observables_batched = observables_array.ndim == 3
        n_observables = (
            observables_array.size // observables_array.shape[0]
            if observables_batched else observables_array.size
        )
        if observables_batched:
            if n_batch and n_batch != observables_array.shape[0]:
                raise ValueError("Batched states and observables must share n_rep.")
            n_batch = observables_array.shape[0]
        step._raw_model_data_ctx.observables_input = <const double *>cnp.PyArray_DATA(
            observables_array
        )
    else:
        observables_array = None
        step._raw_model_data_ctx.observables_input = NULL

    step._raw_model_data_ctx.n_states = n_states
    step._raw_model_data_ctx.states_batched = states_batched
    step._raw_model_data_ctx.n_observables = n_observables
    step._raw_model_data_ctx.observables_batched = observables_batched
    step._bind(
        name,
        sdsge_mc_raw_model_data_runner,
        <const void *>&step._raw_model_data_ctx,
        (states_array, observables_array),
        n_batch,
        0,
        0,
        n_states + n_observables,
        0,
    )
    return step


def simulate1_step(
    str name,
    uintptr_t measurement_addr,
    int64_t T,
    int64_t n_var,
    int64_t n_exog,
    int64_t n_par,
    int64_t n_obs,
    NativeShockPlan shocks=None,
):
    """Bind a first-order simulation with resolved dimensions and callback.

    ``shocks`` makes the step draw its own shock block from ``rep_idx`` instead
    of reading one the runner copied in. The draw's scratch is appended to the
    simulation's own arena, and the plan is held for the step's lifetime.
    """
    cdef NativeStep step = NativeStep()
    cdef int64_t arena = sdsge_simulate_order1_arena_size(n_var, n_exog, T, n_par)

    if T < 0 or n_exog < 0 or n_par < 0:
        raise ValueError("Native simulation dimensions must be non-negative.")
    if n_var <= 0:
        raise ValueError("First-order simulation requires at least one variable.")
    if n_obs < 0:
        raise ValueError("Native simulation observation count must be non-negative.")
    if n_obs and measurement_addr == 0:
        raise ValueError("Observable simulation requires a measurement address.")

    step._simulate_order1_ctx.measurement = (
        <sdsge_measurement_fn><void *>measurement_addr
    )
    step._simulate_order1_ctx.T = T
    step._simulate_order1_ctx.n = n_var
    step._simulate_order1_ctx.k = n_exog
    step._simulate_order1_ctx.n_par = n_par
    step._simulate_order1_ctx.m = n_obs
    step._simulate_order1_ctx.shock_scratch_offset = arena
    if shocks is None:
        step._simulate_order1_ctx.shocks = NULL
    else:
        _check_shock_plan(shocks, T, n_exog)
        step._simulate_order1_ctx.shocks = shocks.c_plan()
        arena += shocks.scratch_size
    step._bind(
        name,
        sdsge_mc_simulate_order1_runner,
        <const void *>&step._simulate_order1_ctx,
        shocks,
        0,
        arena,
        0,
        T * (n_var + n_obs),
        0,
    )
    return step


cdef void _check_shock_plan(
    NativeShockPlan shocks,
    int64_t T,
    int64_t n_exog,
) except *:
    """The plan and the step must agree on the block the draw writes into."""
    if shocks._plan.T != T or shocks._plan.n_exog != n_exog:
        raise ValueError(
            "Native shock plan does not match its simulation step: plan is "
            f"({shocks._plan.T}, {shocks._plan.n_exog}), step expects "
            f"({T}, {n_exog})."
        )


def simulate2_step(
    str name,
    uintptr_t measurement_addr,
    int64_t T,
    int64_t n_state,
    int64_t n_ctrl,
    int64_t n_exog,
    int64_t n_par,
    int64_t n_obs,
    NativeShockPlan shocks=None,
):
    """Bind a second-order simulation with resolved dimensions and callback.

    ``shocks`` behaves as in :func:`simulate1_step`.
    """
    cdef NativeStep step = NativeStep()
    cdef int64_t n_var = n_state + n_ctrl
    cdef int64_t arena

    if T < 0 or n_state <= 0 or n_ctrl < 0 or n_exog < 0 or n_par < 0:
        raise ValueError("Native simulation dimensions must be valid and non-negative.")
    if n_obs < 0:
        raise ValueError("Native simulation observation count must be non-negative.")
    if n_obs and measurement_addr == 0:
        raise ValueError("Observable simulation requires a measurement address.")

    arena = sdsge_simulate_order2_arena_size(n_state, n_var, n_exog, T, n_par)
    step._simulate_order2_ctx.measurement = (
        <sdsge_measurement_fn><void *>measurement_addr
    )
    step._simulate_order2_ctx.T = T
    step._simulate_order2_ctx.n_state = n_state
    step._simulate_order2_ctx.n_ctrl = n_ctrl
    step._simulate_order2_ctx.n_exog = n_exog
    step._simulate_order2_ctx.n_par = n_par
    step._simulate_order2_ctx.m = n_obs
    step._simulate_order2_ctx.shock_scratch_offset = arena
    if shocks is None:
        step._simulate_order2_ctx.shocks = NULL
    else:
        _check_shock_plan(shocks, T, n_exog)
        step._simulate_order2_ctx.shocks = shocks.c_plan()
        arena += shocks.scratch_size
    step._bind(
        name,
        sdsge_mc_simulate_order2_runner,
        <const void *>&step._simulate_order2_ctx,
        shocks,
        0,
        arena,
        0,
        T * (n_var + n_obs),
        0,
    )
    return step


def filter_linear_step(
    str name,
    int64_t T,
    int64_t n_var,
    int64_t n_obs,
    int64_t n_exog,
    bint symmetrize=False,
    bint joseph_cov=True,
    double jitter=0.0,
    bint return_shocks=False,
):
    """Bind a linear Kalman filter with resolved scalar settings."""
    cdef NativeStep step = NativeStep()

    if T < 0 or n_var <= 0 or n_obs <= 0 or n_exog < 0:
        raise ValueError("Native linear filter dimensions must be valid.")

    step._filter_linear_ctx.T = T
    step._filter_linear_ctx.n = n_var
    step._filter_linear_ctx.m = n_obs
    step._filter_linear_ctx.k = n_exog
    step._filter_linear_ctx.symmetrize = symmetrize
    step._filter_linear_ctx.joseph_cov = joseph_cov
    step._filter_linear_ctx.jitter = jitter
    step._filter_linear_ctx.return_shocks = return_shocks
    step._bind(
        name,
        sdsge_mc_filter_linear_runner,
        <const void *>&step._filter_linear_ctx,
        None,
        0,
        -1,
        0,
        sdsge_filter_linear_output_arena_size(
            n_var,
            n_obs,
            n_exog,
            T,
            return_shocks,
        ),
        0,
    )
    return step


def filter_extended_step(
    str name,
    uintptr_t measurement_addr,
    uintptr_t jacobian_addr,
    int64_t T,
    int64_t n_var,
    int64_t n_obs,
    int64_t n_exog,
    int64_t n_par,
    bint symmetrize=False,
    bint joseph_cov=True,
    double jitter=0.0,
    bint return_shocks=False,
):
    """Bind an extended Kalman filter with resolved callbacks and settings."""
    cdef NativeStep step = NativeStep()

    if T < 0 or n_var <= 0 or n_obs <= 0 or n_exog < 0 or n_par < 0:
        raise ValueError("Native extended filter dimensions must be valid.")
    if measurement_addr == 0 or jacobian_addr == 0:
        raise ValueError("Native extended filtering requires both callback addresses.")

    step._filter_extended_ctx.measurement = <meas_fn><void *>measurement_addr
    step._filter_extended_ctx.jacobian = <meas_fn><void *>jacobian_addr
    step._filter_extended_ctx.T = T
    step._filter_extended_ctx.n = n_var
    step._filter_extended_ctx.m = n_obs
    step._filter_extended_ctx.k = n_exog
    step._filter_extended_ctx.n_par = n_par
    step._filter_extended_ctx.symmetrize = symmetrize
    step._filter_extended_ctx.joseph_cov = joseph_cov
    step._filter_extended_ctx.jitter = jitter
    step._filter_extended_ctx.return_shocks = return_shocks
    step._bind(
        name,
        sdsge_mc_filter_extended_runner,
        <const void *>&step._filter_extended_ctx,
        None,
        0,
        -1,
        0,
        sdsge_filter_extended_output_arena_size(
            n_var,
            n_obs,
            n_exog,
            T,
            return_shocks,
        ),
        0,
    )
    return step


def filter_unscented_step(
    str name,
    uintptr_t measurement_addr,
    int64_t T,
    int64_t n_state,
    int64_t n_ctrl,
    int64_t n_exog,
    int64_t n_obs,
    int64_t n_par,
    double alpha,
    double beta,
    double kappa,
    bint symmetrize=False,
    double jitter=0.0,
):
    """Bind an unscented Kalman filter with resolved callback and settings."""
    cdef NativeStep step = NativeStep()

    if T < 0 or n_state <= 0 or n_ctrl < 0 or n_exog < 0 or n_obs <= 0:
        raise ValueError("Native unscented filter dimensions must be valid.")
    if n_par < 0 or measurement_addr == 0:
        raise ValueError("Native unscented filtering requires a callback address.")

    step._filter_unscented_ctx.measurement = <meas_fn><void *>measurement_addr
    step._filter_unscented_ctx.T = T
    step._filter_unscented_ctx.n_state = n_state
    step._filter_unscented_ctx.n_ctrl = n_ctrl
    step._filter_unscented_ctx.n_exog = n_exog
    step._filter_unscented_ctx.n_obs = n_obs
    step._filter_unscented_ctx.n_par = n_par
    step._filter_unscented_ctx.alpha = alpha
    step._filter_unscented_ctx.beta = beta
    step._filter_unscented_ctx.kappa = kappa
    step._filter_unscented_ctx.symmetrize = symmetrize
    step._filter_unscented_ctx.jitter = jitter
    step._bind(
        name,
        sdsge_mc_filter_unscented_runner,
        <const void *>&step._filter_unscented_ctx,
        None,
        0,
        -1,
        0,
        sdsge_filter_unscented_output_arena_size(
            n_state,
            n_ctrl,
            n_obs,
            T,
        ),
        0,
    )
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
    uintptr_t function_address=0,
    object backing=None,
    int64_t output_n=-1,
    int64_t output_p=-1,
):
    """Bind one native transform adapter using its resolved scalar settings."""
    cdef NativeStep step = NativeStep()
    cdef int64_t input_count
    cdef int64_t output_rows
    cdef sdsge_mc_step_fn fn
    cdef const void *ctx

    if n < 0 or p < 0:
        raise ValueError("Transform dimensions must be non-negative.")

    input_count = n * p
    step._input_n_int = 0
    step._output_n_int = 0
    if kind == "passthrough":
        step._passthrough_ctx.n = n
        step._passthrough_ctx.p = p
        fn = sdsge_mc_passthrough_runner
        ctx = <const void *>&step._passthrough_ctx
        step._input_n_float = input_count
        step._output_n_float = input_count
    elif kind == "standardize":
        step._standardize_ctx.n = n
        step._standardize_ctx.p = p
        step._standardize_ctx.ddof = ddof
        fn = sdsge_mc_standardize_runner
        ctx = <const void *>&step._standardize_ctx
        step._input_n_float = input_count + 2 * p
        step._output_n_float = input_count
    elif kind == "log":
        step._log_ctx.n = n
        step._log_ctx.p = p
        step._log_ctx.offset = offset
        fn = sdsge_mc_log_runner
        ctx = <const void *>&step._log_ctx
        step._input_n_float = input_count
        step._output_n_float = input_count
    elif kind == "log_diff":
        step._log_diff_ctx.n = n
        step._log_diff_ctx.p = p
        step._log_diff_ctx.offset = offset
        fn = sdsge_mc_log_diff_runner
        ctx = <const void *>&step._log_diff_ctx
        step._input_n_float = input_count + p
        step._output_n_float = max(0, n - 1) * p
    elif kind == "diff":
        step._diff_ctx.n = n
        step._diff_ctx.p = p
        step._diff_ctx.order = order
        fn = sdsge_mc_diff_runner
        ctx = <const void *>&step._diff_ctx
        step._input_n_float = input_count + max(0, order) * p
        step._output_n_float = max(0, n - order) * p
    elif kind == "rolling_mean":
        step._rolling_mean_ctx.n = n
        step._rolling_mean_ctx.p = p
        step._rolling_mean_ctx.window = window
        fn = sdsge_mc_rolling_mean_runner
        ctx = <const void *>&step._rolling_mean_ctx
        step._input_n_float = input_count + p
        output_rows = max(0, n - window + 1)
        step._output_n_float = output_rows * p
    elif kind == "rolling_var" or kind == "rolling_std":
        step._rolling_var_ctx.n = n
        step._rolling_var_ctx.p = p
        step._rolling_var_ctx.window = window
        step._rolling_var_ctx.ddof = ddof
        fn = (
            sdsge_mc_rolling_var_runner
            if kind == "rolling_var"
            else sdsge_mc_rolling_std_runner
        )
        ctx = <const void *>&step._rolling_var_ctx
        step._input_n_float = input_count + 2 * p
        output_rows = max(0, n - window + 1)
        step._output_n_float = output_rows * p
    elif kind == "custom":
        if function_address == 0:
            raise ValueError("Native custom transforms require a callback address.")
        if backing is None:
            raise ValueError("Native custom transforms require callback backing.")
        if output_n < 0 or output_p < 0:
            raise ValueError("Native custom transform output dimensions are required.")
        step._user_transform_ctx.fn = <user_transform_fn><void *>function_address
        step._user_transform_ctx.n_in = n
        step._user_transform_ctx.p_in = p
        step._user_transform_ctx.n_out = output_n
        step._user_transform_ctx.p_out = output_p
        fn = sdsge_mc_user_transform_runner
        ctx = <const void *>&step._user_transform_ctx
        step._input_n_float = input_count
        step._output_n_float = output_n * output_p
    else:
        raise ValueError(f"Unsupported native transform kind: {kind!r}.")
    step._bind(
        name,
        fn,
        ctx,
        backing,
        0,
        step._input_n_float,
        step._input_n_int,
        step._output_n_float,
        step._output_n_int,
    )
    return step


def ols_step(str name, int64_t n, int64_t p, bint intercept=True):
    """Bind the native OLS adapter after source staging has been compiled."""
    cdef NativeStep step = NativeStep()

    if n < 0 or p <= 0:
        raise ValueError("OLS dimensions require n >= 0 and p > 0.")

    step._ols_ctx.n = n
    step._ols_ctx.p = p
    step._ols_ctx.intercept = intercept
    step._bind(
        name,
        sdsge_mc_ols_runner,
        <const void *>&step._ols_ctx,
        None,
        0,
        -1,
        0,
        2 * p + 2,
        1,
    )
    return step


cdef cnp.ndarray _alpha_grid(
    double start,
    double stop,
    int64_t num,
):
    if start <= 0.0 or stop <= 0.0:
        raise ValueError("Native grid-search penalties must be positive.")
    if num <= 0:
        raise ValueError("Native grid-search requires a positive num.")
    return np.ascontiguousarray(
        np.exp(np.linspace(np.log(start), np.log(stop), num=num)),
        dtype=np.float64,
    )


def ridge_step(
    str name,
    int64_t n,
    int64_t p,
    double alpha,
    bint intercept=True,
):
    cdef NativeStep step = NativeStep()
    if n < 0 or p <= 0 or alpha < 0.0:
        raise ValueError("Native ridge dimensions and alpha must be valid.")
    step._ridge_ctx.n = n
    step._ridge_ctx.p = p
    step._ridge_ctx.intercept = intercept
    step._ridge_ctx.alpha = alpha
    step._bind(name, sdsge_mc_ridge_runner, <const void *>&step._ridge_ctx,
               None, 0, -1, -1, p + 2, 1)
    return step


def ridge_gs_step(
    str name,
    int64_t n,
    int64_t p,
    double start,
    double stop,
    int64_t num,
    int64_t criterion,
    bint intercept=True,
):
    cdef NativeStep step = NativeStep()
    cdef cnp.ndarray alphas = _alpha_grid(start, stop, num)
    if n < 0 or p <= 0:
        raise ValueError("Native ridge grid-search dimensions must be valid.")
    step._ridge_gs_ctx.alphas = <const double *>cnp.PyArray_DATA(alphas)
    step._ridge_gs_ctx.n = n
    step._ridge_gs_ctx.p = p
    step._ridge_gs_ctx.n_alpha = num
    step._ridge_gs_ctx.criterion = criterion
    step._ridge_gs_ctx.intercept = intercept
    step._bind(name, sdsge_mc_ridge_gs_runner,
               <const void *>&step._ridge_gs_ctx, alphas, 0, -1, -1, p + 2, 1)
    return step


def lasso_step(
    str name,
    int64_t n,
    int64_t p,
    double alpha,
    int64_t max_iter=1000,
    double tol=1e-10,
    bint intercept=True,
):
    cdef NativeStep step = NativeStep()
    if n < 0 or p <= 0 or alpha < 0.0 or max_iter <= 0 or tol <= 0.0:
        raise ValueError("Native lasso settings must be valid.")
    step._lasso_ctx.n = n
    step._lasso_ctx.p = p
    step._lasso_ctx.intercept = intercept
    step._lasso_ctx.max_iter = max_iter
    step._lasso_ctx.tol = tol
    step._lasso_ctx.alpha = alpha
    step._bind(name, sdsge_mc_lasso_runner, <const void *>&step._lasso_ctx,
               None, 0, -1, -1, p + 2, 1)
    return step


def lasso_gs_step(
    str name,
    int64_t n,
    int64_t p,
    double start,
    double stop,
    int64_t num,
    int64_t max_iter=1000,
    double tol=1e-10,
    bint intercept=True,
):
    cdef NativeStep step = NativeStep()
    cdef cnp.ndarray alphas = _alpha_grid(start, stop, num)
    if n < 0 or p <= 0 or max_iter <= 0 or tol <= 0.0:
        raise ValueError("Native lasso grid-search settings must be valid.")
    step._lasso_gs_ctx.alphas = <const double *>cnp.PyArray_DATA(alphas)
    step._lasso_gs_ctx.n = n
    step._lasso_gs_ctx.p = p
    step._lasso_gs_ctx.n_alpha = num
    step._lasso_gs_ctx.intercept = intercept
    step._lasso_gs_ctx.max_iter = max_iter
    step._lasso_gs_ctx.tol = tol
    step._bind(name, sdsge_mc_lasso_gs_runner,
               <const void *>&step._lasso_gs_ctx, alphas, 0, -1, -1, p + 2, 1)
    return step


def elastic_net_step(
    str name,
    int64_t n,
    int64_t p,
    double alpha,
    double l1_ratio,
    int64_t max_iter=1000,
    double tol=1e-10,
    bint intercept=True,
):
    cdef NativeStep step = NativeStep()
    if (n < 0 or p <= 0 or alpha < 0.0 or l1_ratio < 0.0
            or l1_ratio > 1.0 or max_iter <= 0 or tol <= 0.0):
        raise ValueError("Native elastic-net settings must be valid.")
    step._elastic_net_ctx.n = n
    step._elastic_net_ctx.p = p
    step._elastic_net_ctx.intercept = intercept
    step._elastic_net_ctx.max_iter = max_iter
    step._elastic_net_ctx.tol = tol
    step._elastic_net_ctx.alpha = alpha
    step._elastic_net_ctx.l1_ratio = l1_ratio
    step._bind(name, sdsge_mc_elastic_net_runner,
               <const void *>&step._elastic_net_ctx, None, 0, -1, -1, p + 2, 1)
    return step


def elastic_net_gs_step(
    str name,
    int64_t n,
    int64_t p,
    double start,
    double stop,
    int64_t num,
    double l1_ratio,
    int64_t criterion,
    int64_t max_iter=1000,
    double tol=1e-10,
    bint intercept=True,
):
    cdef NativeStep step = NativeStep()
    cdef cnp.ndarray alphas = _alpha_grid(start, stop, num)
    if (n < 0 or p <= 0 or l1_ratio < 0.0 or l1_ratio > 1.0
            or max_iter <= 0 or tol <= 0.0):
        raise ValueError("Native elastic-net grid-search settings must be valid.")
    step._elastic_net_gs_ctx.alphas = <const double *>cnp.PyArray_DATA(alphas)
    step._elastic_net_gs_ctx.n = n
    step._elastic_net_gs_ctx.p = p
    step._elastic_net_gs_ctx.n_alpha = num
    step._elastic_net_gs_ctx.criterion = criterion
    step._elastic_net_gs_ctx.intercept = intercept
    step._elastic_net_gs_ctx.max_iter = max_iter
    step._elastic_net_gs_ctx.tol = tol
    step._elastic_net_gs_ctx.l1_ratio = l1_ratio
    step._bind(name, sdsge_mc_elastic_net_gs_runner,
               <const void *>&step._elastic_net_gs_ctx, alphas, 0, -1, -1, p + 2, 1)
    return step


def jarque_bera_step(str name, int64_t n):
    """Bind the native Jarque-Bera adapter after source staging has compiled."""
    cdef NativeStep step = NativeStep()

    if n < 0:
        raise ValueError("Jarque-Bera sample length must be non-negative.")

    step._jarque_bera_ctx.n = n
    step._test_distribution = ReferenceDistribution.JB_LOOKUP
    step._test_df = n
    step._bind(
        name,
        sdsge_mc_jarque_bera_test_runner,
        <const void *>&step._jarque_bera_ctx,
        None,
        0,
        -1,
        0,
        1,
        1,
    )
    return step


def wald_step(
    str name,
    target,
    int64_t n,
    int64_t q,
    int64_t manual_bandwidth,
    int kernel_id,
    int bandwidth_mode,
    int kind,
):
    cdef NativeStep step = NativeStep()
    cdef cnp.ndarray target_array = np.ascontiguousarray(target, dtype=np.float64)
    if n < 0 or q <= 0 or target_array.size == 0:
        raise ValueError("Native Wald dimensions and target must be valid.")
    step._wald_ctx.target = <const double *>cnp.PyArray_DATA(target_array)
    step._wald_ctx.n = n
    step._wald_ctx.q = q
    step._wald_ctx.manual_bandwidth = manual_bandwidth
    step._wald_ctx.kernel_id = kernel_id
    step._wald_ctx.bandwidth_mode = bandwidth_mode
    step._wald_ctx.kind = kind
    step._test_distribution = ReferenceDistribution.CHI2
    step._test_df = q if kind == 0 else q * (q + 1) // 2
    step._bind(name, sdsge_mc_wald_test_runner, <const void *>&step._wald_ctx,
               target_array, 0, -1, -1, 1, 1)
    return step


def ljung_box_step(str name, int64_t n, int64_t lags):
    cdef NativeStep step = NativeStep()
    if n < 0 or lags < 0:
        raise ValueError("Native Ljung-Box settings must be non-negative.")
    step._ljung_box_ctx.n = n
    step._ljung_box_ctx.lags = lags
    step._test_distribution = ReferenceDistribution.CHI2
    step._test_df = lags
    step._bind(name, sdsge_mc_ljung_box_test_runner,
               <const void *>&step._ljung_box_ctx, None, 0, -1, 0, 1, 1)
    return step


def breusch_pagan_step(str name, int64_t n, int64_t k, bint robust=False):
    cdef NativeStep step = NativeStep()
    if n < 0 or k < 0:
        raise ValueError("Native Breusch-Pagan dimensions must be non-negative.")
    step._breusch_pagan_ctx.n = n
    step._breusch_pagan_ctx.k = k
    step._breusch_pagan_ctx.robust = robust
    step._test_distribution = ReferenceDistribution.CHI2
    step._test_df = k
    step._bind(name, sdsge_mc_breusch_pagan_test_runner,
               <const void *>&step._breusch_pagan_ctx, None, 0, -1, 0, 1, 1)
    return step


def breusch_godfrey_step(str name, int64_t n, int64_t k, int64_t lags):
    cdef NativeStep step = NativeStep()
    if n < 0 or k < 0 or lags < 0:
        raise ValueError("Native Breusch-Godfrey settings must be non-negative.")
    step._breusch_godfrey_ctx.n = n
    step._breusch_godfrey_ctx.k = k
    step._breusch_godfrey_ctx.lags = lags
    step._test_distribution = ReferenceDistribution.CHI2
    step._test_df = lags
    step._bind(name, sdsge_mc_breusch_godfrey_test_runner,
               <const void *>&step._breusch_godfrey_ctx, None, 0, -1, 0, 1, 1)
    return step


def cusum_step(str name, int64_t n, int64_t p):
    cdef NativeStep step = NativeStep()
    if n < 0 or p <= 0:
        raise ValueError("Native CUSUM dimensions must be valid.")
    step._cusum_ctx.n = n
    step._cusum_ctx.p = p
    step._test_distribution = ReferenceDistribution.CUSUM
    step._test_df = np.nan
    step._bind(name, sdsge_mc_cusum_test_runner, <const void *>&step._cusum_ctx,
               None, 0, -1, 0, 1, 1)
    return step


def cusumsq_step(str name, int64_t n, int64_t p):
    cdef NativeStep step = NativeStep()
    if n < 0 or p <= 0:
        raise ValueError("Native CUSUMSQ dimensions must be valid.")
    step._cusumsq_ctx.n = n
    step._cusumsq_ctx.p = p
    step._test_distribution = ReferenceDistribution.CUSUMSQ
    step._test_df = max(0, n - p)
    step._bind(name, sdsge_mc_cusumsq_test_runner,
               <const void *>&step._cusumsq_ctx, None, 0, -1, 0, 1, 1)
    return step


def chow_step(str name, int64_t n, int64_t p, int64_t t_break):
    cdef NativeStep step = NativeStep()
    if n < 0 or p <= 0 or t_break < 0:
        raise ValueError("Native Chow settings must be valid.")
    step._chow_ctx.n = n
    step._chow_ctx.p = p
    step._chow_ctx.t_break = t_break
    step._test_distribution = ReferenceDistribution.F
    step._test_df = (p, n - 2 * p)
    step._bind(name, sdsge_mc_chow_test_runner, <const void *>&step._chow_ctx,
               None, 0, -1, 0, 1, 1)
    return step


def run(
    allocation,
    steps,
    input_bindings=None,
    bint fail_fast=False,
    bint profile_steps=False,
):
    """Lower ``allocation`` and static bindings, then invoke the native loop.

    Bindings must follow the allocation plan's step order. The allocation keeps
    all NumPy arenas alive during the no-GIL call, and retained outputs remain
    available through ``allocation.steps`` after this function returns.
    """
    cdef int64_t n_steps = len(steps)
    cdef int64_t step_idx
    cdef int status
    cdef double wall_started_s = 0.0
    cdef double wall_elapsed_s = 0.0
    cdef object step_names = tuple(allocation.plan)
    cdef object step_arenas
    cdef object step_binding_specs
    cdef object binding_spec
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
    cdef cnp.ndarray step_elapsed_s_by_worker
    cdef cnp.ndarray step_counts_by_worker
    cdef cnp.ndarray step_failures_by_worker
    cdef cnp.ndarray binding_columns
    cdef cnp.ndarray static_values
    cdef sdsge_mc_step_desc *descs
    cdef sdsge_mc_float_input_binding *bindings = NULL
    cdef sdsge_mc_runner_ctx runner
    cdef int64_t n_bindings = 0
    cdef int64_t binding_idx = 0
    cdef int64_t binding_offset = 0

    if n_steps == 0:
        raise ValueError("Native runner requires at least one step binding.")
    if n_steps != len(step_names):
        raise ValueError("Native bindings must cover every "
                         "allocated step exactly once.")
    if input_bindings is None:
        input_bindings = tuple(() for _ in range(n_steps))
    if len(input_bindings) != n_steps:
        raise ValueError("Native input bindings must match the step count.")
    for step_binding_specs in input_bindings:
        n_bindings += len(step_binding_specs)

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
        if n_bindings:
            bindings = <sdsge_mc_float_input_binding *>PyMem_Malloc(
                n_bindings * sizeof(sdsge_mc_float_input_binding)
            )
            if bindings == NULL:
                raise MemoryError("Unable to allocate native "
                                  "Monte Carlo input bindings.")
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
            step_binding_specs = input_bindings[step_idx]
            descs[step_idx].n_float_input_bindings = len(step_binding_specs)
            descs[step_idx].float_input_bindings = (
                bindings + binding_offset if step_binding_specs else NULL
            )
            for binding_idx in range(len(step_binding_specs)):
                binding_spec = step_binding_specs[binding_idx]
                binding_columns = np.asarray(binding_spec.columns)
                if (
                    binding_columns.dtype != np.dtype(np.int64)
                    or binding_columns.ndim != 1
                    or not binding_columns.flags.c_contiguous
                ):
                    raise ValueError(
                        "Native input binding columns must be contiguous int64 vectors."
                    )
                bindings[binding_offset + binding_idx].source_step_idx = (
                    binding_spec.source_step_idx
                )
                bindings[binding_offset + binding_idx].source_offset = (
                    binding_spec.source_offset
                )
                bindings[binding_offset + binding_idx].source_row_stride = (
                    binding_spec.source_row_stride
                )
                bindings[binding_offset + binding_idx].row_start = (
                        binding_spec.row_start
                )
                bindings[binding_offset + binding_idx].n_rows = binding_spec.n_rows
                bindings[binding_offset + binding_idx].columns = (
                    _int_data(binding_columns)
                    if binding_columns.shape[0]
                    else NULL
                )
                bindings[binding_offset + binding_idx].n_columns = (
                    binding_columns.shape[0]
                )
                bindings[binding_offset + binding_idx].target_offset = (
                    binding_spec.target_offset
                )
                bindings[binding_offset + binding_idx].target_row_stride = (
                    binding_spec.target_row_stride
                )
                bindings[binding_offset + binding_idx].fill_value = (
                        binding_spec.fill_value
                )
                if binding_spec.source_step_idx < -1:
                    static_values = np.asarray(binding_spec.static_values)
                    if (
                        static_values.dtype != np.dtype(np.float64)
                        or static_values.ndim != 1
                        or not static_values.flags.c_contiguous
                    ):
                        raise ValueError(
                            "Native static input bindings require "
                            "contiguous float64 vectors."
                        )
                    bindings[binding_offset + binding_idx].static_source = _float_data(
                        static_values
                    )
                    bindings[binding_offset + binding_idx].static_rep_stride = (
                        binding_spec.static_rep_stride
                    )
                    bindings[binding_offset + binding_idx].static_batched = (
                        binding_spec.static_batched
                    )
                else:
                    bindings[binding_offset + binding_idx].static_source = NULL
                    bindings[binding_offset + binding_idx].static_rep_stride = 0
                    bindings[binding_offset + binding_idx].static_batched = 0
            binding_offset += len(step_binding_specs)

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
        runner.profile_steps = profile_steps
        if profile_steps:
            step_elapsed_s_by_worker = np.empty(
                (allocation.n_workers, n_steps), dtype=np.float64
            )
            step_counts_by_worker = np.empty(
                (allocation.n_workers, n_steps), dtype=np.int64
            )
            step_failures_by_worker = np.empty(
                (allocation.n_workers, n_steps), dtype=np.int64
            )
            runner.step_elapsed_s_by_worker = _float_data(step_elapsed_s_by_worker)
            runner.step_counts_by_worker = _int_data(step_counts_by_worker)
            runner.step_failures_by_worker = _int_data(step_failures_by_worker)
            wall_started_s = perf_counter()
        else:
            runner.step_elapsed_s_by_worker = NULL
            runner.step_counts_by_worker = NULL
            runner.step_failures_by_worker = NULL
        with nogil:
            status = sdsge_mc_run(&runner)
        if profile_steps:
            wall_elapsed_s = perf_counter() - wall_started_s
        return NativeRunResult(
            status,
            runner.halt_failure.rep_idx,
            runner.halt_failure.step_idx,
            runner.halt_failure.status,
            wall_elapsed_s,
            step_elapsed_s_by_worker if profile_steps else None,
            step_counts_by_worker if profile_steps else None,
            step_failures_by_worker if profile_steps else None,
        )
    finally:
        PyMem_Free(bindings)
        PyMem_Free(descs)
