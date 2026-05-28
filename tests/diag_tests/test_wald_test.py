from __future__ import annotations

import numpy as np

from SymbolicDSGE._diag_tests.distributions import PvalMethod
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
    wald_covariance_hac,
    wald_mean_hac,
    wald_second_moment_hac,
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


def _quadratic_wald_stat(
    g: np.ndarray,
    target: np.ndarray,
    *,
    center: bool,
    bandwidth: int,
) -> float:
    x = g - g.mean(axis=0) if center else g
    vech_idx = np.triu_indices(target.shape[0])
    quadratic_moments = np.empty((g.shape[0], vech_idx[0].size), dtype=np.float64)

    for i in range(g.shape[0]):
        quadratic_moments[i] = np.outer(x[i], x[i])[vech_idx]

    mean = quadratic_moments.mean(axis=0)
    centered_moments = quadratic_moments - mean
    omega = _manual_bartlett_hac(centered_moments, bandwidth)
    dev = mean - target[vech_idx]
    return float(g.shape[0] * dev @ np.linalg.solve(omega, dev))


def _quadratic_sample() -> np.ndarray:
    return np.ascontiguousarray(
        np.array(
            [
                [1.0, 2.0],
                [2.0, -1.0],
                [0.0, 1.0],
                [3.0, 0.0],
                [-1.0, 2.0],
                [1.5, -0.5],
                [-2.0, 1.0],
                [0.5, 3.0],
            ],
            dtype=np.float64,
        )
    )


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


def test_wald_mean_hac_uses_right_tail_p_value_method() -> None:
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

    out = wald_mean_hac(g, np.zeros(2, dtype=np.float64), bandwidth=0)

    assert out.pval_method is PvalMethod.SF


def test_wald_mean_hac_rejects_large_mean_deviation() -> None:
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

    out = wald_mean_hac(g, np.zeros(2, dtype=np.float64), bandwidth=0, alpha=0.05)

    assert out.df == 2
    assert out.statistic > 0.0
    assert out.pval < 0.05
    assert out.is_significant()


def test_wald_covariance_hac_matches_manual_vech_statistic() -> None:
    g = _quadratic_sample()
    target = np.eye(2, dtype=np.float64)

    out = wald_covariance_hac(g, target, bandwidth=0)
    manual_stat = _quadratic_wald_stat(
        g,
        target,
        center=True,
        bandwidth=0,
    )

    assert out.test_name == "wald_covariance_hac"
    assert out.pval_method is PvalMethod.SF
    assert out.df == 3
    assert out.statistic == np.float64(manual_stat)


def test_wald_second_moment_hac_matches_manual_vech_statistic() -> None:
    g = _quadratic_sample()
    target = np.eye(2, dtype=np.float64)

    out = wald_second_moment_hac(g, target, bandwidth=0)
    manual_stat = _quadratic_wald_stat(
        g,
        target,
        center=False,
        bandwidth=0,
    )

    assert out.test_name == "wald_second_moment_hac"
    assert out.pval_method is PvalMethod.SF
    assert out.df == 3
    assert out.statistic == np.float64(manual_stat)


def test_wald_quadratic_hac_rejects_non_symmetric_targets() -> None:
    g = _quadratic_sample()
    target = np.array([[1.0, 0.1], [0.2, 1.0]], dtype=np.float64)

    with np.testing.assert_raises_regex(ValueError, "covariance matrix.*symmetric"):
        wald_covariance_hac(g, target, bandwidth=0)

    with np.testing.assert_raises_regex(ValueError, "second moment matrix.*symmetric"):
        wald_second_moment_hac(g, target, bandwidth=0)


def test_wald_quadratic_hac_rejects_target_dimension_mismatch() -> None:
    g = _quadratic_sample()
    target = np.eye(3, dtype=np.float64)

    with np.testing.assert_raises_regex(ValueError, "covariance matrix dimension"):
        wald_covariance_hac(g, target, bandwidth=0)

    with np.testing.assert_raises_regex(ValueError, "second moment matrix dimension"):
        wald_second_moment_hac(g, target, bandwidth=0)


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
