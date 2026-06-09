from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from SymbolicDSGE._diag_tests.breusch_godfrey import breusch_godfrey
from SymbolicDSGE._diag_tests.distributions import PvalMethod, ReferenceDistribution
from SymbolicDSGE._diag_tests.status import TestStatus


def _manual_bg_stat(eps: np.ndarray, X: np.ndarray, lags: int) -> float:
    n = eps.size
    k = X.shape[1]
    design = np.empty((n, k + lags + 1), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1 : k + 1] = X
    for lag in range(1, lags + 1):
        col = k + lag
        design[:lag, col] = 0.0
        design[lag:, col] = eps[:-lag]
    bhat = np.linalg.lstsq(design, eps, rcond=None)[0]
    rss = np.sum((eps - design @ bhat) ** 2)
    tss = np.sum(eps**2)
    return float(n * (1.0 - rss / tss))


@pytest.mark.parametrize("lags", [1, 3])
def test_breusch_godfrey_matches_lm_definition(lags: int) -> None:
    rng = np.random.default_rng(2024)
    X = rng.normal(size=(150, 2))
    eps = rng.normal(size=150)

    out = breusch_godfrey(eps, X, lags=lags, alpha=0.1)
    expected = _manual_bg_stat(eps, X, lags)

    assert out.test_name == "breusch_godfrey"
    assert out.status is TestStatus.OK
    assert out.dist is ReferenceDistribution.CHI2
    assert out.pval_method is PvalMethod.SF
    assert out.df == lags
    assert out.alpha == np.float64(0.1)
    assert out.statistic == pytest.approx(expected)
    assert out.pval == pytest.approx(chi2(df=lags).sf(expected))


def test_breusch_godfrey_detects_serial_correlation() -> None:
    rng = np.random.default_rng(7)
    n = 400
    noise = rng.normal(size=n)
    correlated = np.empty(n, dtype=np.float64)
    correlated[0] = noise[0]
    for t in range(1, n):
        correlated[t] = 0.8 * correlated[t - 1] + noise[t]
    X = rng.normal(size=(n, 1))

    serial = breusch_godfrey(correlated, X, lags=1)
    white = breusch_godfrey(noise, X, lags=1)

    assert serial.status is TestStatus.OK
    assert white.status is TestStatus.OK
    # Strongly autocorrelated residuals yield a large LM statistic and a
    # rejection, while white noise does not.
    assert serial.statistic > white.statistic
    assert serial.pval < 0.01
    assert white.pval > serial.pval


def test_breusch_godfrey_reports_computation_statuses() -> None:
    insufficient = breusch_godfrey(
        np.zeros(2, dtype=np.float64), np.zeros((2, 1), dtype=np.float64), lags=3
    )
    bad_shape = breusch_godfrey(
        np.zeros(5, dtype=np.float64), np.zeros((4, 1), dtype=np.float64), lags=1
    )

    assert insufficient.status is TestStatus.INSUFFICIENT_SAMPLES
    assert bad_shape.status is TestStatus.BAD_SHAPE
    assert insufficient.df == 3
    assert np.isnan(insufficient.statistic)
    assert np.isnan(bad_shape.statistic)
