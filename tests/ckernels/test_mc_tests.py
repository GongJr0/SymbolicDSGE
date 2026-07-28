"""Parity coverage for caller-buffer Monte Carlo diagnostic-test shims."""

from __future__ import annotations

import numpy as np

from SymbolicDSGE._ckernels.diag import _diag as diag
from SymbolicDSGE._ckernels.monte_carlo import _tests as mc_tests
from SymbolicDSGE._diag_tests.wald_test import (
    wald_covariance_hac,
    wald_mean_hac,
    wald_second_moment_hac,
)


def _sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(847)
    n = 100
    X = np.ascontiguousarray(
        np.column_stack((np.ones(n), rng.normal(size=(n, 3)))), dtype=np.float64
    )
    y = np.ascontiguousarray(
        X @ np.asarray((0.5, 0.3, -0.2, 0.1)) + rng.normal(size=n),
        dtype=np.float64,
    )
    residuals = np.ascontiguousarray(rng.normal(size=n), dtype=np.float64)
    return y, X, residuals


def test_mc_test_shims_match_native_diagnostic_statistics() -> None:
    y, X, residuals = _sample()
    n, p = X.shape
    lags = 5

    assert mc_tests.breusch_godfrey_arena_size(n, p - 1, lags) == (
        n * (1 + (p - 1) + lags)
        + 2 * (1 + (p - 1) + lags) ** 2
        + 2 * (1 + (p - 1) + lags)
    )
    assert mc_tests.breusch_pagan_arena_size(n, p) == n + 2 * p * p + 2 * p
    assert mc_tests.chow_arena_size(p) == 2 * p * p + 2 * p
    assert mc_tests.cusum_arena_size(n, p) == (2 * (n - p) + 5 * p * p + 5 * p)
    assert mc_tests.cusumsq_arena_size(n, p) == 3 * p * p + 3 * p + n - p
    wald_q = p - 1
    wald_v = wald_q * (wald_q + 1) // 2
    assert mc_tests.wald_mean_hac_arena_size(n, wald_q) == (
        n * wald_q + 3 * wald_q * wald_q + 4 * wald_q
    )
    assert mc_tests.wald_covariance_hac_arena_size(n, wald_q) == (
        n * wald_q + n * wald_v + 3 * wald_v * wald_v + 5 * wald_v
    )
    assert mc_tests.wald_second_moment_hac_arena_size(n, wald_q) == (
        n * wald_v + 3 * wald_v * wald_v + 5 * wald_v
    )

    statistic, status = mc_tests.ljung_box_runner(
        residuals, lags, np.empty(n), np.empty(lags + 1)
    )
    expected_status, expected = diag.lb_stat(residuals, lags)
    assert status == expected_status
    np.testing.assert_allclose(statistic, expected)

    wald_g = X[:, 1:]
    wald_q = wald_g.shape[1]
    target_mean = np.zeros(wald_q, dtype=np.float64)
    mean_arena = np.empty(mc_tests.wald_mean_hac_arena_size(n, wald_q))
    statistic, status = mc_tests.wald_mean_hac_runner(
        wald_g,
        target_mean,
        1,
        mc_tests.WALD_BW_MANUAL,
        2,
        mean_arena,
        np.empty(wald_q, dtype=np.int64),
    )
    expected = wald_mean_hac(wald_g, target_mean, kernel="parzen", bandwidth=2)
    assert status == 0
    np.testing.assert_allclose(statistic, expected.statistic)

    target_matrix = np.eye(wald_q, dtype=np.float64)
    v = wald_q * (wald_q + 1) // 2
    covariance_arena = np.empty(mc_tests.wald_covariance_hac_arena_size(n, wald_q))
    statistic, status = mc_tests.wald_covariance_hac_runner(
        wald_g,
        target_matrix,
        2,
        mc_tests.WALD_BW_MANUAL,
        2,
        covariance_arena,
        np.empty(v, dtype=np.int64),
    )
    expected = wald_covariance_hac(wald_g, target_matrix, kernel="qs", bandwidth=2)
    assert status == 0
    np.testing.assert_allclose(statistic, expected.statistic)

    statistic, status = mc_tests.wald_second_moment_hac_runner(
        wald_g,
        target_matrix,
        1,
        mc_tests.WALD_BW_AUTO,
        0,
        np.empty(mc_tests.wald_second_moment_hac_arena_size(n, wald_q)),
        np.empty(v, dtype=np.int64),
    )
    expected = wald_second_moment_hac(
        wald_g, target_matrix, kernel="parzen", bandwidth="auto"
    )
    assert status == 0
    np.testing.assert_allclose(statistic, expected.statistic)

    statistic, status = mc_tests.jarque_bera_runner(residuals)
    expected_status, expected = diag.jb_stat(residuals)
    assert status == expected_status
    np.testing.assert_allclose(statistic, expected)

    mean = np.ascontiguousarray(np.array([1.0, -2.0, 0.5]))
    target = np.zeros(3, dtype=np.float64)
    omega = np.array(
        [[0.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )
    statistic, status = mc_tests.wald_runner(
        mean,
        target,
        omega,
        n,
        np.empty(3),
        np.empty((3, 3)),
        np.empty(3, dtype=np.int64),
        np.empty(3),
    )
    expected_status, expected = diag.wald_stat_from_mean_and_cov(mean, target, omega, n)
    assert status == expected_status
    np.testing.assert_allclose(statistic, expected)

    arena = np.empty(n + 2 * p * p + 2 * p)
    statistic, status = mc_tests.breusch_pagan_runner(residuals, X, False, arena)
    expected_status, rss, tss = diag.bp_aux(residuals, X)
    assert status == expected_status
    np.testing.assert_allclose(statistic, max(0.0, 0.5 * (tss - rss)))

    statistic, status = mc_tests.breusch_pagan_runner(residuals, X, True, arena)
    expected = 0.0 if tss <= 0.0 else n * min(1.0, max(0.0, 1.0 - rss / tss))
    assert status == expected_status
    np.testing.assert_allclose(statistic, expected)

    k = p - 1
    bg_p = 1 + k + lags
    statistic, status = mc_tests.breusch_godfrey_runner(
        residuals, X[:, 1:], lags, np.empty(n * bg_p + 2 * bg_p * bg_p + 2 * bg_p)
    )
    expected_status, expected = diag.bg_stat(residuals, X[:, 1:], lags)
    assert status == expected_status
    np.testing.assert_allclose(statistic, expected)

    statistic, status = mc_tests.chow_runner(y, X, 50, np.empty(2 * p * p + 2 * p))
    expected_status, expected = diag.chow_stat(y, X, 50)
    assert status == expected_status
    np.testing.assert_allclose(statistic, expected)

    cusum_arena = np.empty(n - p + 2 * p * p + 2 * p + 3 * p * p + 3 * p + n - p)
    statistic, status = mc_tests.cusum_runner(y, X, cusum_arena)
    expected_status, expected = diag.cusum_stat(y, X)
    assert status == expected_status
    np.testing.assert_allclose(statistic, expected)

    cusumsq_arena = np.empty(3 * p * p + 3 * p + n - p)
    statistic, status = mc_tests.cusumsq_runner(y, X, cusumsq_arena)
    expected_status, _, expected = diag.cusumsq_stat(y, X)
    assert status == expected_status
    np.testing.assert_allclose(statistic, expected)
