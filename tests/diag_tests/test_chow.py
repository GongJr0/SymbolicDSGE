from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import f as f_dist

from SymbolicDSGE._diag_tests.chow import _chow_stat, chow
from SymbolicDSGE._diag_tests.distributions import PvalMethod, ReferenceDistribution
from SymbolicDSGE._diag_tests.status import TestStatus


def _reference_chow(y: np.ndarray, X: np.ndarray, t_break: int) -> float:
    """Textbook Chow F: pooled vs. split residual sums of squares."""

    def rss(yy: np.ndarray, XX: np.ndarray) -> float:
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        resid = yy - XX @ beta
        return float(resid @ resid)

    T, p = X.shape
    rss_c = rss(y, X)
    rss_1 = rss(y[:t_break], X[:t_break])
    rss_2 = rss(y[t_break:], X[t_break:])
    num = (rss_c - (rss_1 + rss_2)) / p
    denom = (rss_1 + rss_2) / (T - 2 * p)
    return num / denom


def test_chow_matches_ols_partition() -> None:
    rng = np.random.default_rng(11)
    T, p = 120, 3
    X = np.column_stack([np.ones(T), rng.normal(size=(T, p - 1))])
    y = X @ np.array([1.0, 0.5, -0.25]) + rng.normal(size=T)
    t_break = 50

    status, stat = _chow_stat(y, X, t_break)
    expected = _reference_chow(y, X, t_break)

    assert status == int(TestStatus.OK)
    assert float(stat) == pytest.approx(expected, rel=1e-9)

    out = chow(y, X, t_break, alpha=0.1)
    assert out.test_name == "chow"
    assert out.status is TestStatus.OK
    assert out.dist is ReferenceDistribution.F
    assert out.pval_method is PvalMethod.SF
    assert out.df == (p, T - 2 * p)
    assert out.alpha == np.float64(0.1)
    assert out.statistic == pytest.approx(expected, rel=1e-9)
    assert out.pval == pytest.approx(f_dist.sf(expected, p, T - 2 * p))
    assert 0.0 <= out.pval <= 1.0


def test_chow_detects_coefficient_break() -> None:
    rng = np.random.default_rng(0)
    T, p = 200, 2
    X = np.column_stack([np.ones(T), rng.normal(size=T)])
    y = X @ np.array([1.0, 0.5]) + rng.normal(size=T)
    t_break = T // 2

    stable = chow(y, X, t_break, alpha=0.05)

    broken_y = y.copy()
    broken_y[t_break:] = X[t_break:] @ np.array([1.0, 3.0]) + rng.normal(
        size=T - t_break
    )
    broken = chow(broken_y, X, t_break, alpha=0.05)

    assert stable.status is TestStatus.OK
    assert broken.status is TestStatus.OK
    assert not stable.is_significant()
    assert broken.statistic > stable.statistic
    assert broken.pval < stable.pval
    assert broken.is_significant()


def test_chow_is_correctly_sized_under_stability() -> None:
    rng = np.random.default_rng(5)
    T = 200
    t_break = T // 2
    reps = 1000
    rejections = 0
    for _ in range(reps):
        X = np.column_stack([np.ones(T), rng.normal(size=T)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(size=T)
        rejections += chow(y, X, t_break, alpha=0.05).is_significant()
    # Exact F reference distribution: empirical size should sit near nominal.
    assert 0.02 < rejections / reps < 0.09


def test_chow_reports_computation_statuses() -> None:
    rng = np.random.default_rng(1)
    T, p = 40, 2
    X = np.column_stack([np.ones(T), rng.normal(size=T)])
    y = X @ np.array([1.0, 0.5]) + rng.normal(size=T)

    too_low = chow(y, X, 0)
    too_high = chow(y, X, T)
    bad_shape = chow(
        np.zeros(5, dtype=np.float64), np.zeros((6, 2), dtype=np.float64), 3
    )

    assert too_low.status is TestStatus.BAD_PARAMETER
    assert too_high.status is TestStatus.BAD_PARAMETER
    assert bad_shape.status is TestStatus.BAD_SHAPE
    assert np.isnan(too_low.statistic)
    assert np.isnan(too_high.statistic)
    assert np.isnan(bad_shape.statistic)
