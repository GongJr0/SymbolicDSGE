# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Offset separators for each MC step in/out arena."""

from libc.stdint cimport int64_t


cdef extern from "sdsge_common.h":
    enum: SDSGE_ARENA_MAX_BUFFERS

    ctypedef struct arena_offset:
        int64_t foffset[SDSGE_ARENA_MAX_BUFFERS]
        int64_t ioffset[SDSGE_ARENA_MAX_BUFFERS]
        int64_t n_fbuf
        int64_t n_ibuf


cdef extern from "layout.h":
    # Core Steps
    arena_offset sdsge_passthrough_arena_offset(int64_t n,
                                                int64_t p)
    arena_offset sdsge_raw_model_data_output_arena_offset(
            int64_t n_states, int64_t n_shocks, int64_t n_observables
            )
    arena_offset sdsge_simulate_order1_arena_offset(
            int64_t n, int64_t k, int64_t T, int64_t n_par
            )
    arena_offset sdsge_simulate_order1_output_arena_offset(
            int64_t n, int64_t k, int64_t T, int64_t m
            )
    arena_offset sdsge_simulate_order2_arena_offset(
            int64_t n_state, int64_t n_var, int64_t n_exog, int64_t T,
            int64_t n_par
            )
    arena_offset sdsge_simulate_order2_output_arena_offset(
            int64_t n_var, int64_t n_exog, int64_t T, int64_t m
            )
    arena_offset sdsge_filter_linear_input_arena_offset(
        int64_t n, int64_t m, int64_t k, int64_t T
        )
    arena_offset sdsge_filter_linear_output_arena_offset(
            int64_t n, int64_t m, int64_t k, int64_t T,
            int return_shocks
            )
    arena_offset sdsge_filter_extended_input_arena_offset(
            int64_t n, int64_t m, int64_t k, int64_t T, int64_t n_par
            )
    arena_offset sdsge_filter_extended_output_arena_offset(
            int64_t n, int64_t m, int64_t k, int64_t T,
            int return_shocks
            )
    arena_offset sdsge_filter_unscented_input_arena_offset(
            int64_t n_state, int64_t n_ctrl, int64_t n_exog, int64_t n_obs,
            int64_t T, int64_t n_par
            )
    arena_offset sdsge_filter_unscented_output_arena_offset(
            int64_t n_state, int64_t n_ctrl, int64_t n_obs, int64_t T
            )

    # Regression Steps
    # One layout for all of them: the kind picks the scratch and decides whether
    # the output reports a standard error. `intercept`, `n_alpha` and `max_iter`
    # are read by the kinds that need them and ignored by the rest.
    ctypedef enum sdsge_mc_regression_kind:
        SDSGE_MC_REGRESSION_OLS
        SDSGE_MC_REGRESSION_RIDGE
        SDSGE_MC_REGRESSION_RIDGE_GS
        SDSGE_MC_REGRESSION_LASSO
        SDSGE_MC_REGRESSION_LASSO_GS
        SDSGE_MC_REGRESSION_ELASTIC_NET
        SDSGE_MC_REGRESSION_ELASTIC_NET_GS

    arena_offset sdsge_mc_regression_arena_offset(
            int64_t kind, int64_t n, int64_t p, int intercept,
            int64_t n_alpha, int64_t max_iter
            )
    arena_offset sdsge_mc_regression_output_arena_offset(int64_t kind,
                                                         int64_t p)

    # Transform Steps
    # One layout for all of them: the kind picks the scratch, `order` is read by
    # `diff` alone, and a rolling window never reaches the arena.
    ctypedef enum sdsge_mc_transform_kind:
        SDSGE_MC_TRANSFORM_STANDARDIZE
        SDSGE_MC_TRANSFORM_LOG
        SDSGE_MC_TRANSFORM_LOG_DIFF
        SDSGE_MC_TRANSFORM_DIFF
        SDSGE_MC_TRANSFORM_ROLLING_MEAN
        SDSGE_MC_TRANSFORM_ROLLING_VAR
        SDSGE_MC_TRANSFORM_ROLLING_STD

    arena_offset sdsge_mc_transform_arena_offset(
            int64_t kind, int64_t n, int64_t p, int64_t order
            )
