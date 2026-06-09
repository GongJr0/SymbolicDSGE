from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from SymbolicDSGE._diag_tests.breusch_pagan import (
    breusch_pagan,
    robust_breusch_pagan,
)
from SymbolicDSGE._diag_tests.distributions import PvalMethod, ReferenceDistribution
from SymbolicDSGE._diag_tests.status import TestStatus


def _manual_bp_stat(eps: np.ndarray, X: np.ndarray, *, robust: bool) -> float:
    augmented = np.column_stack((np.ones(eps.size, dtype=np.float64), X))
    g = eps**2 / np.mean(eps**2)
    coefficients = np.linalg.lstsq(augmented, g, rcond=None)[0]
    rss = np.sum((g - augmented @ coefficients) ** 2)
    tss = np.sum((g - g.mean()) ** 2)
    if robust:
        return float(eps.size * (1.0 - rss / tss))
    return float((tss - rss) * 0.5)


@pytest.mark.parametrize("robust", [False, True])
def test_breusch_pagan_matches_auxiliary_regression_definition(robust: bool) -> None:
    rng = np.random.default_rng(1024)
    X = rng.normal(size=(200, 2))
    eps = rng.normal(scale=np.exp(0.4 * X[:, 0]), size=200)
    func = robust_breusch_pagan if robust else breusch_pagan

    out = func(eps, X, alpha=0.1)
    expected = _manual_bp_stat(eps, X, robust=robust)

    assert out.test_name == ("robust_breusch_pagan" if robust else "breusch_pagan")
    assert out.status is TestStatus.OK
    assert out.dist is ReferenceDistribution.CHI2
    assert out.pval_method is PvalMethod.SF
    assert out.df == X.shape[1]
    assert out.alpha == np.float64(0.1)
    assert out.statistic == pytest.approx(expected)
    assert out.pval == pytest.approx(chi2(df=X.shape[1]).sf(expected))


def test_breusch_pagan_reports_computation_statuses() -> None:
    empty_X = np.empty((0, 1), dtype=np.float64)
    empty = breusch_pagan(np.empty(0, dtype=np.float64), empty_X)
    undefined = breusch_pagan(np.zeros(5, dtype=np.float64), np.arange(5.0)[:, None])
    robust_constant = robust_breusch_pagan(
        np.ones(5, dtype=np.float64), np.arange(5.0)[:, None]
    )
    x = np.arange(10.0)
    rank_deficient = breusch_pagan(
        np.linspace(-1.0, 1.0, 10),
        np.column_stack((x, x * 2.0)),
    )

    assert empty.status is TestStatus.INSUFFICIENT_SAMPLES
    assert undefined.status is TestStatus.UDEF_VARIANCE
    assert robust_constant.status is TestStatus.OK
    assert robust_constant.statistic == 0.0
    assert robust_constant.pval == 1.0
    assert rank_deficient.status is TestStatus.LINALG
    assert np.isnan(empty.statistic)
    assert np.isnan(undefined.statistic)
    assert np.isnan(rank_deficient.statistic)


@pytest.mark.parametrize(
    ("eps", "X", "message"),
    [
        (np.ones((3, 1)), np.ones((3, 1)), "residuals must be a 1D"),
        (np.ones(3), np.ones(3), "regressors must be a 2D"),
        (np.ones(3), np.ones((2, 1)), "row counts differ"),
        (np.ones(3), np.empty((3, 0)), "at least one variance regressor"),
        (np.array([1.0, np.nan]), np.arange(2.0)[:, None], "finite values"),
        (np.ones(3), np.ones((3, 1)), "must not contain a constant column"),
    ],
)
def test_breusch_pagan_validates_inputs(
    eps: np.ndarray, X: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        breusch_pagan(eps, X)
