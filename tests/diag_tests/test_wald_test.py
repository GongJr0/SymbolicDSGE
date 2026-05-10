from __future__ import annotations

import numpy as np

from SymbolicDSGE._diag_tests.hac_covariance import (
    kernel_dispatcher,
    wooldridge_bandwidth,
)
from SymbolicDSGE._diag_tests.wald_test import (
    ERR_BAD_SHAPE,
    ERR_LINALG,
    OK,
    jit_wald_hac_stat,
    jit_wald_stat_from_mean_and_cov,
)


def _manual_bartlett_hac(r: np.ndarray, bandwidth: int) -> np.ndarray:
    n = r.shape[0]
    max_lag = min(bandwidth, n - 1)
    omega = r.T @ r / n

    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma_l = r[lag:].T @ r[:-lag] / n
        omega = omega + weight * (gamma_l + gamma_l.T)

    return omega


def test_jit_wald_hac_stat_matches_manual_statistic() -> None:
    g = np.ascontiguousarray(
        np.array(
            [
                [1.0, 2.0],
                [2.0, -1.0],
                [0.0, 1.0],
                [3.0, 0.0],
                [-1.0, 2.0],
            ],
            dtype=np.float64,
        )
    )
    target = np.zeros(2, dtype=np.float64)
    mean_buffer = np.empty(2, dtype=np.float64)
    centered_buffer = np.empty_like(g)
    omega_buffer = np.empty((2, 2), dtype=np.float64)
    bandwidth = wooldridge_bandwidth(g)

    err, stat, df = jit_wald_hac_stat(
        g,
        target,
        kernel_dispatcher("bartlett"),
        bandwidth,
        mean_buffer,
        centered_buffer,
        omega_buffer,
    )
    manual_mean = g.mean(axis=0)
    manual_centered = g - manual_mean
    manual_omega = _manual_bartlett_hac(manual_centered, bandwidth)
    manual_stat = g.shape[0] * manual_mean @ np.linalg.solve(manual_omega, manual_mean)

    assert err == OK
    assert df == 2
    assert np.isclose(stat, manual_stat)


def test_jit_wald_hac_stat_returns_shape_error_for_bad_target() -> None:
    g = np.ascontiguousarray(np.ones((5, 2), dtype=np.float64))
    target = np.zeros(3, dtype=np.float64)
    mean_buffer = np.empty(2, dtype=np.float64)
    centered_buffer = np.empty_like(g)
    omega_buffer = np.empty((2, 2), dtype=np.float64)

    err, stat, df = jit_wald_hac_stat(
        g,
        target,
        kernel_dispatcher("bartlett"),
        0,
        mean_buffer,
        centered_buffer,
        omega_buffer,
    )

    assert err == ERR_BAD_SHAPE
    assert np.isnan(stat)
    assert df == 2


def test_jit_wald_stat_from_mean_and_cov_returns_shape_error_for_bad_target() -> None:
    err, stat, df = jit_wald_stat_from_mean_and_cov(
        np.zeros(2, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(2, dtype=np.float64),
        10,
    )

    assert err == ERR_BAD_SHAPE
    assert np.isnan(stat)
    assert df == 2


def test_jit_wald_stat_from_mean_and_cov_returns_linalg_error_for_singular_covariance() -> (
    None
):
    err, stat, df = jit_wald_stat_from_mean_and_cov(
        np.array([1.0, 0.0], dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        10,
    )

    assert err == ERR_LINALG
    assert np.isnan(stat)
    assert df == 2
