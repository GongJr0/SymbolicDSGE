"""Parity coverage for caller-buffer Monte Carlo diagnostic-test shims."""

from __future__ import annotations

import numpy as np

from SymbolicDSGE._ckernels.diag import _diag as diag
from SymbolicDSGE._ckernels.monte_carlo import _tests as mc_tests


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

    statistic, status = mc_tests.ljung_box_fit(
        residuals, lags, np.empty(n), np.empty(lags + 1)
    )
    expected_status, expected = diag.lb_stat(residuals, lags)
    assert status == expected_status
    np.testing.assert_allclose(statistic, expected)

    statistic, status = mc_tests.jarque_bera_fit(residuals)
    expected_status, expected = diag.jb_stat(residuals)
    assert status == expected_status
    np.testing.assert_allclose(statistic, expected)

    arena = np.empty(n + 2 * p * p + 2 * p)
    statistic, status = mc_tests.breusch_pagan_fit(residuals, X, False, arena)
    expected_status, rss, tss = diag.bp_aux(residuals, X)
    assert status == expected_status
    np.testing.assert_allclose(statistic, max(0.0, 0.5 * (tss - rss)))

    statistic, status = mc_tests.breusch_pagan_fit(residuals, X, True, arena)
    expected = 0.0 if tss <= 0.0 else n * min(1.0, max(0.0, 1.0 - rss / tss))
    assert status == expected_status
    np.testing.assert_allclose(statistic, expected)

    k = p - 1
    bg_p = 1 + k + lags
    statistic, status = mc_tests.breusch_godfrey_fit(
        residuals, X[:, 1:], lags, np.empty(n * bg_p + 2 * bg_p * bg_p + 2 * bg_p)
    )
    expected_status, expected = diag.bg_stat(residuals, X[:, 1:], lags)
    assert status == expected_status
    np.testing.assert_allclose(statistic, expected)

    statistic, status = mc_tests.chow_fit(y, X, 50, np.empty(2 * p * p + 2 * p))
    expected_status, expected = diag.chow_stat(y, X, 50)
    assert status == expected_status
    np.testing.assert_allclose(statistic, expected)

    cusum_arena = np.empty(n - p + 2 * p * p + 2 * p + 3 * p * p + 3 * p + n - p)
    statistic, status = mc_tests.cusum_fit(y, X, cusum_arena)
    expected_status, expected = diag.cusum_stat(y, X)
    assert status == expected_status
    np.testing.assert_allclose(statistic, expected)

    cusumsq_arena = np.empty(3 * p * p + 3 * p + n - p)
    statistic, status = mc_tests.cusumsq_fit(y, X, cusumsq_arena)
    expected_status, _, expected = diag.cusumsq_stat(y, X)
    assert status == expected_status
    np.testing.assert_allclose(statistic, expected)
