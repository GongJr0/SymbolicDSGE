# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Offset separators for each MC step in/out arena."""

from libc.stdint cimport int64_t

from typing import NamedTuple


cdef extern from "sdsge_common.h":
    enum: SDSGE_ARENA_MAX_BUFFERS

    ctypedef struct arena_size:
        int64_t n_float
        int64_t n_int

    ctypedef struct arena_offset:
        int64_t ioffset[SDSGE_ARENA_MAX_BUFFERS]
        int64_t foffset[SDSGE_ARENA_MAX_BUFFERS]
        int64_t n_fbuf
        int64_t n_ibuf

    arena_size make_sizer(int64_t n_float, int64_t n_int)


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

    # Diagnostic Steps
    # Three shapes cover every kind. The `diag` kernels size their own work, so
    # it arrives here as a parameter rather than being known to the layout.
    arena_offset sdsge_mc_diag_sample_arena_offset(int64_t n, int64_t q,
                                                   arena_size work)
    arena_offset sdsge_mc_diag_design_arena_offset(int64_t n, int64_t m,
                                                   arena_size work)
    arena_offset sdsge_mc_diag_augmented_arena_offset(int64_t n, int64_t k,
                                                      arena_size work)
    arena_offset sdsge_mc_diag_output_arena_offset()

    # Transform Steps
    # One layout for all of them: the kind picks the scratch, `order` is read by
    # `diff` alone, and a rolling window never reaches the arena. The window
    # does set the output rows, which are a scalar because the output is one
    # buffer.
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
    int64_t sdsge_mc_transform_output_rows(
            int64_t kind, int64_t n, int64_t order, int64_t window
            )


cdef extern from "diag.h":
    arena_size sdsge_bg_arena_size(int64_t n, int64_t k, int64_t lags)
    arena_size sdsge_bp_arena_size(int64_t n, int64_t p)
    arena_size sdsge_chow_arena_size(int64_t p)
    arena_size sdsge_cusum_arena_size(int64_t n, int64_t p)
    arena_size sdsge_cusumsq_arena_size(int64_t n, int64_t p)
    arena_size sdsge_lb_arena_size(int64_t n, int64_t lags)


cdef extern from "diag_wald.h":
    arena_size sdsge_wald_mean_hac_arena_size(int64_t n, int64_t q)
    arena_size sdsge_wald_covariance_hac_arena_size(int64_t n, int64_t q)
    arena_size sdsge_wald_second_moment_hac_arena_size(int64_t n, int64_t q)


class ArenaOffset(NamedTuple):
    """One layout's buffers, as a start and a width per lane.

    C stores the boundary closing each buffer. These are those same boundaries
    read the way an arena is sliced: entry i opens buffer i, and a buffer the
    configuration left out is present at a width of zero rather than absent.
    """

    foffset: tuple
    fwidth: tuple
    ioffset: tuple
    iwidth: tuple


cdef inline tuple _lane(const int64_t *close, const int64_t n_buf):
    """One lane's starts and widths, from the boundaries closing its buffers.

    Entry i closes buffer i, which is also what opens buffer i + 1. The first
    buffer opens on the arena, which C does not store, so the shift by one is
    the whole conversion.
    """
    cdef int64_t i
    cdef int64_t start = 0
    cdef list offsets = []
    cdef list widths = []
    for i in range(n_buf):
        offsets.append(start)
        widths.append(close[i] - start)
        start = close[i]
    return tuple(offsets), tuple(widths)


cdef inline object _offset(arena_offset off):
    """Both lanes of one layout, walked to the buffer count each declares."""
    cdef tuple flane = _lane(&off.foffset[0], off.n_fbuf)
    cdef tuple ilane = _lane(&off.ioffset[0], off.n_ibuf)
    return ArenaOffset(flane[0], flane[1], ilane[0], ilane[1])


cdef dict _REGRESSION_KINDS = {
    "ols": SDSGE_MC_REGRESSION_OLS,
    "ridge": SDSGE_MC_REGRESSION_RIDGE,
    "ridge_gs": SDSGE_MC_REGRESSION_RIDGE_GS,
    "lasso": SDSGE_MC_REGRESSION_LASSO,
    "lasso_gs": SDSGE_MC_REGRESSION_LASSO_GS,
    "elastic_net": SDSGE_MC_REGRESSION_ELASTIC_NET,
    "elastic_net_gs": SDSGE_MC_REGRESSION_ELASTIC_NET_GS,
}


cdef dict _TRANSFORM_KINDS = {
    "standardize": SDSGE_MC_TRANSFORM_STANDARDIZE,
    "log": SDSGE_MC_TRANSFORM_LOG,
    "log_diff": SDSGE_MC_TRANSFORM_LOG_DIFF,
    "diff": SDSGE_MC_TRANSFORM_DIFF,
    "rolling_mean": SDSGE_MC_TRANSFORM_ROLLING_MEAN,
    "rolling_var": SDSGE_MC_TRANSFORM_ROLLING_VAR,
    "rolling_std": SDSGE_MC_TRANSFORM_ROLLING_STD,
}


def transform_offsets(str kind, int64_t n, int64_t p, int64_t param=0):
    """Return the input and scratch boundaries for a transform.

    ``param`` is the difference order. A rolling window never reaches the
    arena, so it moves nothing here.
    """
    if kind == "passthrough":
        return _offset(sdsge_passthrough_arena_offset(n, p))
    cdef object code = _TRANSFORM_KINDS.get(kind)
    if code is None:
        raise ValueError(f"Unsupported native transform kind: {kind!r}.")
    return _offset(sdsge_mc_transform_arena_offset(<int64_t>code, n, p, param))


def transform_output_rows(str kind, int64_t n, int64_t order=0,
                          int64_t window=0):
    """Return the rows a transform writes from an ``n``-row input.

    The output is one buffer, so its rows are the whole layout. ``order`` is
    read by ``diff`` alone and ``window`` by the rolling kinds.
    """
    cdef object code = _TRANSFORM_KINDS.get(kind)
    if code is None:
        raise ValueError(f"Unsupported native transform kind: {kind!r}.")
    return sdsge_mc_transform_output_rows(<int64_t>code, n, order, window)


def regression_offsets(
    str kind,
    int64_t n,
    int64_t p,
    bint intercept,
    int64_t n_alpha=0,
    int64_t max_iter=0,
):
    """Return the staged-input and work boundaries for a regression."""
    cdef object code = _REGRESSION_KINDS.get(kind)
    if code is None:
        raise ValueError(f"Unsupported native regression kind: {kind!r}.")
    return _offset(
        sdsge_mc_regression_arena_offset(
            <int64_t>code, n, p, intercept, n_alpha, max_iter
        )
    )


def regression_output_offsets(str kind, int64_t p):
    """Return the output boundaries for a regression.

    Only OLS reports a standard error. For every other kind ``se`` closes where
    ``sst`` did, so it is present and empty.
    """
    cdef object code = _REGRESSION_KINDS.get(kind)
    if code is None:
        raise ValueError(f"Unsupported native regression kind: {kind!r}.")
    return _offset(sdsge_mc_regression_output_arena_offset(<int64_t>code, p))


def simulation_offsets(
    int order,
    int64_t n_state,
    int64_t n_var,
    int64_t n_exog,
    int64_t T,
    int64_t n_par,
):
    """Return the packed-input and work boundaries for a simulation."""
    if order == 1:
        return _offset(sdsge_simulate_order1_arena_offset(n_var, n_exog, T, n_par))
    if order == 2:
        return _offset(
            sdsge_simulate_order2_arena_offset(n_state, n_var, n_exog, T, n_par)
        )
    raise ValueError(f"Unsupported native simulation order: {order}.")


def simulation_output_offsets(
    int order,
    int64_t n_var,
    int64_t n_exog,
    int64_t T,
    int64_t n_obs,
):
    """Return the output boundaries for a simulation.

    ``observables`` closes where ``shocks`` did when the step was built without
    them.
    """
    if order == 1:
        return _offset(
            sdsge_simulate_order1_output_arena_offset(n_var, n_exog, T, n_obs)
        )
    if order == 2:
        return _offset(
            sdsge_simulate_order2_output_arena_offset(n_var, n_exog, T, n_obs)
        )
    raise ValueError(f"Unsupported native simulation order: {order}.")


def raw_model_data_output_offsets(
    int64_t n_states,
    int64_t n_shocks,
    int64_t n_observables,
):
    """Return the output boundaries for materialized raw model data.

    A field the step never carried closes where the one before it did.
    """
    return _offset(
        sdsge_raw_model_data_output_arena_offset(n_states, n_shocks, n_observables)
    )


def filter_offsets(
    str kind,
    int64_t n_state,
    int64_t n_ctrl,
    int64_t n_exog,
    int64_t n_obs,
    int64_t T,
    int64_t n_par,
):
    """Return the packed-input boundaries for a filter.

    The Kalman work block sits past the last entry. Only the kernel reads
    inside it and it sizes itself, so it is not a buffer here.
    """
    if kind == "linear":
        return _offset(sdsge_filter_linear_input_arena_offset(
            n_state + n_ctrl, n_obs, n_exog, T))
    if kind == "extended":
        return _offset(sdsge_filter_extended_input_arena_offset(
            n_state + n_ctrl, n_obs, n_exog, T, n_par))
    if kind == "unscented":
        return _offset(sdsge_filter_unscented_input_arena_offset(
            n_state, n_ctrl, n_exog, n_obs, T, n_par))
    raise ValueError(f"Unsupported native filter kind: {kind!r}.")


def filter_output_offsets(
    str kind,
    int64_t n_state,
    int64_t n_ctrl,
    int64_t n_exog,
    int64_t n_obs,
    int64_t T,
    bint return_shocks=False,
):
    """Return the output boundaries for a filter.

    ``eps_hat`` closes where ``S`` did unless ``return_shocks``. Unscented has
    no such field, and no such argument, so the flag does not reach it.
    """
    if kind == "linear":
        return _offset(sdsge_filter_linear_output_arena_offset(
            n_state + n_ctrl, n_obs, n_exog, T, return_shocks))
    if kind == "extended":
        return _offset(sdsge_filter_extended_output_arena_offset(
            n_state + n_ctrl, n_obs, n_exog, T, return_shocks))
    if kind == "unscented":
        return _offset(sdsge_filter_unscented_output_arena_offset(
            n_state, n_ctrl, n_obs, T))
    raise ValueError(f"Unsupported native filter kind: {kind!r}.")


def diagnostic_offsets(str kind, int64_t n, int64_t p=0, int64_t lags=0):
    """Return the staged-input and work boundaries for a diagnostic.

    Three shapes cover every kind. What a branch selects is the work sizer a
    kind names, not the layout it gets.
    """
    if kind == "wald_mean":
        return _offset(sdsge_mc_diag_sample_arena_offset(
            n, p, sdsge_wald_mean_hac_arena_size(n, p)))
    if kind == "wald_covariance":
        return _offset(sdsge_mc_diag_sample_arena_offset(
            n, p, sdsge_wald_covariance_hac_arena_size(n, p)))
    if kind == "wald_second_moment":
        return _offset(sdsge_mc_diag_sample_arena_offset(
            n, p, sdsge_wald_second_moment_hac_arena_size(n, p)))
    if kind == "ljung_box":
        return _offset(sdsge_mc_diag_sample_arena_offset(
            n, 1, sdsge_lb_arena_size(n, lags)))
    if kind == "jarque_bera":
        return _offset(sdsge_mc_diag_sample_arena_offset(n, 1, make_sizer(0, 0)))
    if kind == "breusch_pagan":
        return _offset(sdsge_mc_diag_augmented_arena_offset(
            n, p, sdsge_bp_arena_size(n, p + 1)))
    if kind == "breusch_godfrey":
        return _offset(sdsge_mc_diag_design_arena_offset(
            n, p, sdsge_bg_arena_size(n, p, lags)))
    if kind == "chow":
        return _offset(sdsge_mc_diag_design_arena_offset(
            n, p, sdsge_chow_arena_size(p)))
    if kind == "cusum":
        return _offset(sdsge_mc_diag_design_arena_offset(
            n, p, sdsge_cusum_arena_size(n, p)))
    if kind == "cusumsq":
        return _offset(sdsge_mc_diag_design_arena_offset(
            n, p, sdsge_cusumsq_arena_size(n, p)))
    raise ValueError(f"Unsupported native diagnostic kind: {kind!r}.")


def diagnostic_output_offsets():
    """Return the output boundaries every diagnostic writes."""
    return _offset(sdsge_mc_diag_output_arena_offset())
