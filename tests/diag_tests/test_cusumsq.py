from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._diag_tests.cusumsq import (
    CusumSq,
    _cusumsq_sf,
    _cusumsq_stat,
    cusumsq_test,
)
from SymbolicDSGE._diag_tests.distributions import PvalMethod, ReferenceDistribution
from SymbolicDSGE._diag_tests.status import TestStatus


def _reference_stat(y: np.ndarray, X: np.ndarray) -> tuple[int, float]:
    """Recursive-residual CUSUM of squares, refitting OLS from scratch at every
    step, then the variance-standardized maximum departure from the t/N line."""
    T, p = X.shape
    w = np.empty(T - p, dtype=np.float64)
    for i, t in enumerate(range(p, T)):
        beta = np.linalg.lstsq(X[:t], y[:t], rcond=None)[0]
        xtx_inv = np.linalg.inv(X[:t].T @ X[:t])
        xt = X[t]
        ft = 1.0 + xt @ xtx_inv @ xt
        w[i] = (y[t] - xt @ beta) / np.sqrt(ft)
    n = T - p
    s = np.cumsum(w**2) / np.sum(w**2)
    expected = np.arange(1, n + 1) / n
    return n, float(np.max(np.abs(s - expected)) / np.sqrt(2.0))


def test_cusumsq_matches_recursive_residual_definition() -> None:
    rng = np.random.default_rng(11)
    T, p = 80, 3
    X = np.column_stack([np.ones(T), rng.normal(size=(T, p - 1))])
    y = X @ np.array([1.0, 0.5, -0.25]) + rng.normal(size=T)

    status, n, stat = _cusumsq_stat(y, X)
    expected_n, expected_stat = _reference_stat(y, X)

    assert status == int(TestStatus.OK)
    assert n == expected_n == T - p
    assert float(stat) == pytest.approx(expected_stat, rel=1e-6)

    out = cusumsq_test(y, X, alpha=0.1)
    assert out.test_name == "CUSUMSQ"
    assert out.status is TestStatus.OK
    assert out.dist is ReferenceDistribution.CUSUMSQ
    assert out.pval_method is PvalMethod.SF
    assert out.df == T - p
    assert out.alpha == np.float64(0.1)
    assert out.statistic == pytest.approx(expected_stat, rel=1e-6)
    assert out.pval == pytest.approx(
        min(1.0, _cusumsq_sf(np.float64(out.statistic), T - p))
    )
    assert 0.0 <= out.pval <= 1.0


def test_cusumsq_distribution_is_clamped_survival_function() -> None:
    dist = CusumSq(60)

    # sf == clamped Kolmogorov-type series; cdf is its complement.
    for stat in (0.05, 0.1, 0.2):
        assert dist.sf(stat) == pytest.approx(
            min(1.0, _cusumsq_sf(np.float64(stat), 60))
        )
        assert dist.cdf(stat) == pytest.approx(1.0 - dist.sf(stat))

    # The survival function is always a valid probability.
    xs = np.array([0.05, 0.1, 0.2, 0.4])
    sf = dist.sf(xs)
    assert np.all((sf >= 0.0) & (sf <= 1.0))

    # Vectorized evaluation matches the scalar path.
    np.testing.assert_allclose(dist.sf(xs), [dist.sf(float(v)) for v in xs])


def test_cusumsq_freeze_requires_integer_sample_size() -> None:
    frozen = ReferenceDistribution.CUSUMSQ.freeze(60)
    assert isinstance(frozen, CusumSq)
    assert frozen.n == 60
    assert frozen.sf(0.1) == pytest.approx(min(1.0, _cusumsq_sf(np.float64(0.1), 60)))

    with pytest.raises(TypeError):
        ReferenceDistribution.CUSUMSQ.freeze(60.5)
    with pytest.raises(TypeError):
        ReferenceDistribution.CUSUMSQ.freeze()


def test_cusumsq_detects_variance_break() -> None:
    rng = np.random.default_rng(1)
    T, p = 200, 2
    X = np.column_stack([np.ones(T), rng.normal(size=T)])
    y = X @ np.array([1.0, 0.5]) + rng.normal(size=T)

    stable = cusumsq_test(y, X, alpha=0.05)

    broken_y = y.copy()
    broken_y[T // 2 :] += rng.normal(scale=5.0, size=T - T // 2)
    broken = cusumsq_test(broken_y, X, alpha=0.05)

    assert stable.status is TestStatus.OK
    assert broken.status is TestStatus.OK
    assert not stable.is_significant()
    assert broken.statistic > stable.statistic
    assert broken.pval < stable.pval
    assert broken.is_significant()


def test_cusumsq_is_correctly_sized_under_stability() -> None:
    rng = np.random.default_rng(123)
    T = 200
    reps = 400
    rejections = 0
    for _ in range(reps):
        X = np.column_stack([np.ones(T), rng.normal(size=T)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(size=T)
        rejections += cusumsq_test(y, X, alpha=0.05).is_significant()
    # Nominal size is 0.05; the chi^2(1) variance standardization keeps the
    # empirical rejection rate near it rather than the ~0.28 of the unscaled
    # statistic.
    assert rejections / reps < 0.12


def test_cusumsq_reports_computation_statuses() -> None:
    insufficient = cusumsq_test(
        np.zeros(2, dtype=np.float64), np.zeros((2, 3), dtype=np.float64)
    )
    bad_shape = cusumsq_test(
        np.zeros(5, dtype=np.float64), np.zeros((6, 2), dtype=np.float64)
    )

    assert insufficient.status is TestStatus.INSUFFICIENT_SAMPLES
    assert bad_shape.status is TestStatus.BAD_SHAPE
    assert np.isnan(insufficient.statistic)
    assert np.isnan(bad_shape.statistic)
