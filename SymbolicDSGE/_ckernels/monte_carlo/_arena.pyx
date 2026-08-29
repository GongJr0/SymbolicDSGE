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

cdef extern from "transforms.h":
    arena_size sdsge_standardize_ax0_arena_size(int64_t n, int64_t p) nogil
    arena_size sdsge_log_arena_size(int64_t n, int64_t p) nogil
    arena_size sdsge_log_diff_arena_size(int64_t n, int64_t p) nogil
    arena_size sdsge_diff_arena_size(int64_t n, int64_t p, int64_t order) nogil
    arena_size sdsge_rolling_mean_arena_size(int64_t n, int64_t p, int64_t window) nogil
    arena_size sdsge_rolling_var_arena_size(int64_t n, int64_t p, int64_t window) nogil
    arena_size sdsge_rolling_std_arena_size(int64_t n, int64_t p, int64_t window) nogil

cdef extern from "regression.h":
    arena_size sdsge_mc_ols_work_arena_size(int64_t n, int64_t p) nogil
    arena_size sdsge_mc_ridge_work_arena_size(int64_t n, int64_t p) nogil
    arena_size sdsge_mc_ridge_gs_work_arena_size(int64_t n, int64_t p) nogil
    arena_size sdsge_mc_lasso_work_arena_size(int64_t n, int64_t p) nogil
    arena_size sdsge_mc_lasso_gs_work_arena_size(int64_t n, int64_t p, int intercept,
                                                 int64_t n_alpha,
                                                 int64_t max_iter) nogil
    arena_size sdsge_mc_elastic_net_work_arena_size(int64_t n, int64_t p) nogil
    arena_size sdsge_mc_elastic_net_gs_work_arena_size(int64_t n, int64_t p,
                                                       int intercept,
                                                       int64_t n_alpha) nogil

cdef extern from "core_steps.h":
    arena_size sdsge_passthrough_arena_size(int64_t n, int64_t p) nogil
    int64_t sdsge_simulate_order1_arena_size(int64_t n, int64_t k,
                                             int64_t T, int64_t n_par) nogil
    int64_t sdsge_simulate_order2_arena_size(int64_t n_state,
                                             int64_t n_var, int64_t n_exog,
                                             int64_t T, int64_t n_par) nogil
    int64_t sdsge_filter_linear_input_arena_size(int64_t n, int64_t m,
                                                 int64_t k, int64_t T) nogil
    int64_t sdsge_filter_extended_input_arena_size(int64_t n, int64_t m, int64_t k,
                                                   int64_t T, int64_t n_par) nogil
    int64_t sdsge_filter_unscented_input_arena_size(int64_t n_state, int64_t n_ctrl,
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


def transform_arena_size(str kind, int64_t n, int64_t p, int64_t param=0):
    """Return the complete input and scratch arena requirement for a transform."""
    cdef arena_size size
    if kind == "passthrough":
        size = sdsge_passthrough_arena_size(n, p)
    elif kind == "standardize":
        size = sdsge_standardize_ax0_arena_size(n, p)
    elif kind == "log":
        size = sdsge_log_arena_size(n, p)
    elif kind == "log_diff":
        size = sdsge_log_diff_arena_size(n, p)
    elif kind == "diff":
        size = sdsge_diff_arena_size(n, p, param)
    elif kind == "rolling_mean":
        size = sdsge_rolling_mean_arena_size(n, p, param)
    elif kind == "rolling_var":
        size = sdsge_rolling_var_arena_size(n, p, param)
    elif kind == "rolling_std":
        size = sdsge_rolling_std_arena_size(n, p, param)
    else:
        raise ValueError(f"Unsupported native transform kind: {kind!r}.")
    return _size(size)


def regression_arena_size(
    str kind,
    int64_t n,
    int64_t p,
    bint intercept,
    int64_t n_alpha=0,
    int64_t max_iter=0,
):
    """Return the complete staged-input and work requirement for regression."""
    cdef arena_size size
    if kind == "ols":
        size = sdsge_mc_ols_work_arena_size(n, p)
    elif kind == "ridge":
        size = sdsge_mc_ridge_work_arena_size(n, p)
    elif kind == "ridge_gs":
        size = sdsge_mc_ridge_gs_work_arena_size(n, p)
    elif kind == "lasso":
        size = sdsge_mc_lasso_work_arena_size(n, p)
    elif kind == "lasso_gs":
        size = sdsge_mc_lasso_gs_work_arena_size(n, p, intercept, n_alpha, max_iter)
    elif kind == "elastic_net":
        size = sdsge_mc_elastic_net_work_arena_size(n, p)
    elif kind == "elastic_net_gs":
        size = sdsge_mc_elastic_net_gs_work_arena_size(n, p, intercept, n_alpha)
    else:
        raise ValueError(f"Unsupported native regression kind: {kind!r}.")
    return _size(size)


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
        return sdsge_simulate_order1_arena_size(n_var, n_exog, T, n_par), 0
    if order == 2:
        return sdsge_simulate_order2_arena_size(n_state, n_var, n_exog, T, n_par), 0
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
    cdef int64_t input_size
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
    return input_size + scratch.n_float, scratch.n_int


def diagnostic_arena_size(
    str kind,
    int64_t n,
    int64_t p=0,
    int64_t lags=0,
):
    """Return the complete staged-input and work requirement for a diagnostic."""
    cdef arena_size size
    cdef int64_t input_size
    if kind == "wald_mean":
        size = sdsge_wald_mean_hac_arena_size(n, p)
        input_size = n * p
    elif kind == "wald_covariance":
        size = sdsge_wald_covariance_hac_arena_size(n, p)
        input_size = n * p
    elif kind == "wald_second_moment":
        size = sdsge_wald_second_moment_hac_arena_size(n, p)
        input_size = n * p
    elif kind == "ljung_box":
        size = sdsge_lb_arena_size(n, lags)
        input_size = n
    elif kind == "jarque_bera":
        size.n_float = 0
        size.n_int = 0
        input_size = n
    elif kind == "breusch_pagan":
        size = sdsge_bp_arena_size(n, p + 1)
        input_size = n + n * p + n * (p + 1)
    elif kind == "breusch_godfrey":
        size = sdsge_bg_arena_size(n, p, lags)
        input_size = n + n * p
    elif kind == "chow":
        size = sdsge_chow_arena_size(p)
        input_size = n + n * p
    elif kind == "cusum":
        size = sdsge_cusum_arena_size(n, p)
        input_size = n + n * p
    elif kind == "cusumsq":
        size = sdsge_cusumsq_arena_size(n, p)
        input_size = n + n * p
    else:
        raise ValueError(f"Unsupported native diagnostic kind: {kind!r}.")
    return input_size + size.n_float, size.n_int
