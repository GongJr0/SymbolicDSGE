from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._diag_tests.cusum import (
    CONVERGED,
    CusumDist,
    _a_from_alpha,
    _alpha_from_a,
    cusum,
    cusum_series,
)
from SymbolicDSGE._diag_tests.distributions import PvalMethod, ReferenceDistribution
from SymbolicDSGE._diag_tests.status import TestStatus


def _reference_std_cusum(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Recursive (Brown-Durbin-Evans) residuals, refitting OLS from scratch at
    every step, then standardized and cumulated as the kernel does."""
    T, p = X.shape
    w = np.empty(T - p, dtype=np.float64)
    for i, t in enumerate(range(p, T)):
        beta = np.linalg.lstsq(X[:t], y[:t], rcond=None)[0]
        xtx_inv = np.linalg.inv(X[:t].T @ X[:t])
        xt = X[t]
        ft = 1.0 + xt @ xtx_inv @ xt
        w[i] = (y[t] - xt @ beta) / np.sqrt(ft)
    bhat = np.linalg.lstsq(X, y, rcond=None)[0]
    sigma = np.sqrt(np.sum((y - X @ bhat) ** 2) / (T - p))
    return np.cumsum(w) / sigma


def _reference_statistic(series: np.ndarray, T: int, p: int) -> float:
    sqrt_tp = np.sqrt(T - p)
    denom = sqrt_tp + 2.0 * np.arange(T - p) / sqrt_tp
    return float(np.max(np.abs(series) / denom))


def test_cusum_matches_recursive_residual_definition() -> None:
    rng = np.random.default_rng(11)
    T, p = 80, 3
    X = np.column_stack([np.ones(T), rng.normal(size=(T, p - 1))])
    y = X @ np.array([1.0, 0.5, -0.25]) + rng.normal(size=T)

    status, series = cusum_series(y, X)
    expected_series = _reference_std_cusum(y, X)

    assert status == int(TestStatus.OK)
    assert series.size == T - p
    np.testing.assert_allclose(series, expected_series, rtol=1e-6, atol=1e-9)

    out = cusum(y, X, alpha=0.1)
    expected_stat = _reference_statistic(expected_series, T, p)

    assert out.test_name == "cusum"
    assert out.status is TestStatus.OK
    assert out.dist is ReferenceDistribution.CUSUM
    assert out.pval_method is PvalMethod.SF
    assert np.isnan(out.df)
    assert out.alpha == np.float64(0.1)
    assert out.statistic == pytest.approx(expected_stat, rel=1e-6)
    assert out.pval == pytest.approx(min(1.0, _alpha_from_a(np.float64(out.statistic))))
    assert 0.0 <= out.pval <= 1.0


def test_cusum_distribution_is_clamped_survival_function() -> None:
    dist = CusumDist()

    # sf == clamped Durbin series; cdf is its complement.
    for a in (0.3, 0.8, 1.5):
        assert dist.sf(a) == pytest.approx(min(1.0, _alpha_from_a(np.float64(a))))
        assert dist.cdf(a) == pytest.approx(1.0 - dist.sf(a))

    # Small statistics push the raw series above 1; the survival function must
    # still report a valid probability.
    assert _alpha_from_a(np.float64(0.3)) > 1.0
    assert dist.sf(0.3) == 1.0
    assert dist.cdf(0.3) == 0.0

    # Vectorized evaluation matches the scalar path.
    xs = np.array([0.3, 0.8, 1.5])
    np.testing.assert_allclose(dist.sf(xs), [dist.sf(float(v)) for v in xs])

    # isf is the Newton inverse and round-trips with sf for proper alphas.
    for alpha in (0.01, 0.05, 0.2):
        crit = dist.isf(alpha)
        assert dist.sf(crit) == pytest.approx(alpha, abs=1e-8)
    assert dist.ppf(0.95) == pytest.approx(dist.isf(0.05))


def test_cusum_freeze_is_parameter_free() -> None:
    # The NaN df forwarded by the TestResult constructor is ignored.
    frozen = ReferenceDistribution.CUSUM.freeze(np.float64(np.nan))
    assert frozen.sf(0.7) == pytest.approx(min(1.0, _alpha_from_a(np.float64(0.7))))


@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9])
def test_cusum_inverse_converges_for_common_alphas(alpha: float) -> None:
    status, a = _a_from_alpha(np.float64(alpha))
    assert status == CONVERGED
    assert _alpha_from_a(a) == pytest.approx(alpha, abs=1e-10)


def test_cusum_detects_structural_break() -> None:
    rng = np.random.default_rng(0)
    T, p = 200, 2
    X = np.column_stack([np.ones(T), rng.normal(size=T)])
    y = X @ np.array([1.0, 0.5]) + rng.normal(size=T)

    stable = cusum(y, X, alpha=0.05)

    broken_y = y.copy()
    broken_y[T // 2 :] += 5.0
    broken = cusum(broken_y, X, alpha=0.05)

    assert stable.status is TestStatus.OK
    assert broken.status is TestStatus.OK
    assert not stable.is_significant()
    assert broken.statistic > stable.statistic
    assert broken.pval < stable.pval
    assert broken.is_significant()
    assert broken.pval < 0.01


def test_cusum_reports_computation_statuses() -> None:
    insufficient = cusum(
        np.zeros(2, dtype=np.float64), np.zeros((2, 3), dtype=np.float64)
    )
    bad_shape = cusum(np.zeros(5, dtype=np.float64), np.zeros((6, 2), dtype=np.float64))

    assert insufficient.status is TestStatus.INSUFFICIENT_SAMPLES
    assert bad_shape.status is TestStatus.BAD_SHAPE
    assert np.isnan(insufficient.statistic)
    assert np.isnan(bad_shape.statistic)
