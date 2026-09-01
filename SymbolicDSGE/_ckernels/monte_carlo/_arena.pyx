# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Native Monte Carlo arena-size boundary.

Python's pipeline compiler owns shape resolution. This module exposes the C
arena sizers so its allocation plan always follows the native kernel contract.
"""

from libc.stdint cimport int64_t


cdef extern from "sdsge_common.h":
    ctypedef struct arena_size:
        int64_t n_float
        int64_t n_int

    arena_size make_sizer(int64_t n_float, int64_t n_int) nogil
    arena_size add_arena(arena_size a, arena_size b) nogil

cdef extern from "layout.h":
    ctypedef enum sdsge_mc_transform_kind:
        SDSGE_MC_TRANSFORM_STANDARDIZE
        SDSGE_MC_TRANSFORM_LOG
        SDSGE_MC_TRANSFORM_LOG_DIFF
        SDSGE_MC_TRANSFORM_DIFF
        SDSGE_MC_TRANSFORM_ROLLING_MEAN
        SDSGE_MC_TRANSFORM_ROLLING_VAR
        SDSGE_MC_TRANSFORM_ROLLING_STD

    arena_size sdsge_mc_transform_arena_size(int64_t kind, int64_t n, int64_t p,
                                             int64_t order) nogil

    ctypedef enum sdsge_mc_regression_kind:
        SDSGE_MC_REGRESSION_OLS
        SDSGE_MC_REGRESSION_RIDGE
        SDSGE_MC_REGRESSION_RIDGE_GS
        SDSGE_MC_REGRESSION_LASSO
        SDSGE_MC_REGRESSION_LASSO_GS
        SDSGE_MC_REGRESSION_ELASTIC_NET
        SDSGE_MC_REGRESSION_ELASTIC_NET_GS

    arena_size sdsge_mc_regression_arena_size(int64_t kind, int64_t n,
                                              int64_t p, int intercept,
                                              int64_t n_alpha,
                                              int64_t max_iter) nogil

    arena_size sdsge_mc_diag_sample_arena_size(int64_t n, int64_t q,
                                               arena_size work) nogil
    arena_size sdsge_mc_diag_design_arena_size(int64_t n, int64_t m,
                                               arena_size work) nogil
    arena_size sdsge_mc_diag_augmented_arena_size(int64_t n, int64_t k,
                                                  arena_size work) nogil

    arena_size sdsge_passthrough_arena_size(int64_t n, int64_t p) nogil
    arena_size sdsge_simulate_order1_arena_size(int64_t n, int64_t k,
                                                int64_t T, int64_t n_par) nogil
    arena_size sdsge_simulate_order2_arena_size(int64_t n_state,
                                                int64_t n_var, int64_t n_exog,
                                                int64_t T, int64_t n_par) nogil
    arena_size sdsge_simulate_order1_output_arena_size(int64_t n, int64_t k,
                                                       int64_t T, int64_t m) nogil
    arena_size sdsge_simulate_order2_output_arena_size(int64_t n_var,
                                                       int64_t n_exog,
                                                       int64_t T, int64_t m) nogil
    arena_size sdsge_filter_linear_input_arena_size(int64_t n, int64_t m,
                                                    int64_t k, int64_t T) nogil
    arena_size sdsge_filter_extended_input_arena_size(int64_t n, int64_t m, int64_t k,
                                                      int64_t T, int64_t n_par) nogil
    arena_size sdsge_filter_unscented_input_arena_size(int64_t n_state, int64_t n_ctrl,
                                                       int64_t n_exog, int64_t n_obs,
                                                       int64_t T, int64_t n_par) nogil


cdef extern from "kalman.h":
    arena_size kf_arena_size(int64_t n, int64_t m, int64_t k) nogil
    arena_size ekf_arena_size(int64_t n, int64_t m, int64_t k) nogil
    arena_size ukf_arena_size(int64_t n_state, int64_t n_ctrl,
                              int64_t n_exog, int64_t n_obs) nogil

cdef extern from "diag.h":
    arena_size sdsge_bg_arena_size(int64_t n, int64_t k, int64_t lags) nogil
    arena_size sdsge_bp_arena_size(int64_t n, int64_t p) nogil
    arena_size sdsge_chow_arena_size(int64_t p) nogil
    arena_size sdsge_cusum_arena_size(int64_t n, int64_t p) nogil
    arena_size sdsge_cusumsq_arena_size(int64_t n, int64_t p) nogil
    arena_size sdsge_lb_arena_size(int64_t n, int64_t lags) nogil

cdef extern from "diag_wald.h":
    arena_size sdsge_wald_mean_hac_arena_size(int64_t n, int64_t q) nogil
    arena_size sdsge_wald_covariance_hac_arena_size(int64_t n, int64_t q) nogil
    arena_size sdsge_wald_second_moment_hac_arena_size(int64_t n, int64_t q) nogil


cdef inline tuple _size(arena_size size):
    return size.n_float, size.n_int


# One layout serves every transform, so the kind selects a code rather than a
# function. `passthrough` stays out: it is a core step that happens to share the
# shape, not a transform.
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


def transform_arena_size(str kind, int64_t n, int64_t p, int64_t param=0):
    """Return the complete input and scratch arena requirement for a transform.

    ``param`` is the difference order. A rolling window never reaches the arena;
    it sets the output shape, which the caller resolves from the field.
    """
    if kind == "passthrough":
        return _size(sdsge_passthrough_arena_size(n, p))
    cdef object code = _TRANSFORM_KINDS.get(kind)
    if code is None:
        raise ValueError(f"Unsupported native transform kind: {kind!r}.")
    return _size(sdsge_mc_transform_arena_size(<int64_t>code, n, p, param))


def regression_arena_size(
    str kind,
    int64_t n,
    int64_t p,
    bint intercept,
    int64_t n_alpha=0,
    int64_t max_iter=0,
):
    """Return the complete staged-input and work requirement for regression."""
    cdef object code = _REGRESSION_KINDS.get(kind)
    if code is None:
        raise ValueError(f"Unsupported native regression kind: {kind!r}.")
    return _size(
        sdsge_mc_regression_arena_size(
            <int64_t>code, n, p, intercept, n_alpha, max_iter
        )
    )


def simulation_arena_size(
    int order,
    int64_t n_state,
    int64_t n_var,
    int64_t n_exog,
    int64_t T,
    int64_t n_par,
):
    """Return the complete packed-input and work requirement for simulation."""
    if order == 1:
        return _size(sdsge_simulate_order1_arena_size(n_var, n_exog, T, n_par))
    if order == 2:
        return _size(sdsge_simulate_order2_arena_size(n_state, n_var, n_exog, T, n_par))
    raise ValueError(f"Unsupported native simulation order: {order}.")


def simulation_output_arena_size(
    int order,
    int64_t n_var,
    int64_t n_exog,
    int64_t T,
    int64_t n_obs,
):
    """Return the output requirement for simulation, as the kernel writes it.

    The Python compiler resolves the per-field layout itself, so this is what
    that layout is held to: the two must agree on the total or the step writes
    somewhere the plan did not reserve.
    """
    if order == 1:
        return _size(sdsge_simulate_order1_output_arena_size(n_var, n_exog, T, n_obs))
    if order == 2:
        return _size(sdsge_simulate_order2_output_arena_size(n_var, n_exog, T, n_obs))
    raise ValueError(f"Unsupported native simulation order: {order}.")


def filter_arena_size(
    str kind,
    int64_t n_state,
    int64_t n_ctrl,
    int64_t n_exog,
    int64_t n_obs,
    int64_t T,
    int64_t n_par,
):
    """Return the complete packed-input and Kalman-work requirement."""
    cdef arena_size scratch
    cdef arena_size input_size
    if kind == "linear":
        input_size = sdsge_filter_linear_input_arena_size(n_state + n_ctrl,
                                                          n_obs, n_exog, T)
        scratch = kf_arena_size(n_state + n_ctrl, n_obs, n_exog)
    elif kind == "extended":
        input_size = sdsge_filter_extended_input_arena_size(n_state + n_ctrl, n_obs,
                                                            n_exog, T, n_par)
        scratch = ekf_arena_size(n_state + n_ctrl, n_obs, n_exog)
    elif kind == "unscented":
        input_size = sdsge_filter_unscented_input_arena_size(n_state, n_ctrl, n_exog,
                                                             n_obs, T, n_par)
        scratch = ukf_arena_size(n_state, n_ctrl, n_exog, n_obs)
    else:
        raise ValueError(f"Unsupported native filter kind: {kind!r}.")
    return _size(add_arena(input_size, scratch))


def diagnostic_arena_size(
    str kind,
    int64_t n,
    int64_t p=0,
    int64_t lags=0,
):
    """Return the complete staged-input and work requirement for a diagnostic."""
    if kind == "wald_mean":
        return _size(sdsge_mc_diag_sample_arena_size(
            n, p, sdsge_wald_mean_hac_arena_size(n, p)))
    if kind == "wald_covariance":
        return _size(sdsge_mc_diag_sample_arena_size(
            n, p, sdsge_wald_covariance_hac_arena_size(n, p)))
    if kind == "wald_second_moment":
        return _size(sdsge_mc_diag_sample_arena_size(
            n, p, sdsge_wald_second_moment_hac_arena_size(n, p)))
    if kind == "ljung_box":
        return _size(sdsge_mc_diag_sample_arena_size(
            n, 1, sdsge_lb_arena_size(n, lags)))
    if kind == "jarque_bera":
        return _size(sdsge_mc_diag_sample_arena_size(n, 1, make_sizer(0, 0)))
    if kind == "breusch_pagan":
        return _size(sdsge_mc_diag_augmented_arena_size(
            n, p, sdsge_bp_arena_size(n, p + 1)))
    if kind == "breusch_godfrey":
        return _size(sdsge_mc_diag_design_arena_size(
            n, p, sdsge_bg_arena_size(n, p, lags)))
    if kind == "chow":
        return _size(sdsge_mc_diag_design_arena_size(
            n, p, sdsge_chow_arena_size(p)))
    if kind == "cusum":
        return _size(sdsge_mc_diag_design_arena_size(
            n, p, sdsge_cusum_arena_size(n, p)))
    if kind == "cusumsq":
        return _size(sdsge_mc_diag_design_arena_size(
            n, p, sdsge_cusumsq_arena_size(n, p)))
    raise ValueError(f"Unsupported native diagnostic kind: {kind!r}.")
