from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

import SymbolicDSGE._diag_tests.ljung_box as ljung_module
from SymbolicDSGE._diag_tests.distributions import PvalMethod, ReferenceDistribution
from SymbolicDSGE._diag_tests.ljung_box import (
    BAD_LAG,
    BAD_SHAPE,
    OK,
    UDEF_VARIANCE,
    acorr,
    lb_stat,
    ljung_box,
)
from SymbolicDSGE._diag_tests.status import TestStatus


def _manual_acorr(x: np.ndarray, L: int) -> np.ndarray:
    z = x - x.mean()
    denom = z @ z
    out = np.empty(L + 1, dtype=np.float64)
    out[0] = 1.0
    for ell in range(1, L + 1):
        out[ell] = (z[ell:] @ z[:-ell]) / denom
    return out


def _manual_ljung_box(x: np.ndarray, L: int) -> float:
    rho = _manual_acorr(x, L)
    n = x.size
    out = 0.0
    for ell in range(1, L + 1):
        out += rho[ell] ** 2 / (n - ell)
    return n * (n + 2) * out


def test_acorr_matches_manual_autocorrelation() -> None:
    x = np.array([1.0, 2.0, 0.0, 4.0, 3.0], dtype=np.float64)
    L = 3

    err, out = acorr.py_func(x, L)

    assert err == OK
    assert OK == TestStatus.OK
    np.testing.assert_allclose(out, _manual_acorr(x, L))


def test_acorr_reports_undefined_variance_for_constant_series() -> None:
    err, out = acorr.py_func(np.ones(5, dtype=np.float64), 2)

    assert err == UDEF_VARIANCE
    assert UDEF_VARIANCE == TestStatus.UDEF_VARIANCE
    assert np.isnan(out).all()


def test_acorr_python_vectorized_branch_matches_manual_autocorrelation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.array([1.0, 2.0, 0.0, 4.0, 3.0], dtype=np.float64)
    monkeypatch.setattr(ljung_module, "LOOP_LIMIT_N", 1)

    err, out = acorr.py_func(x, 2)

    assert err == OK
    np.testing.assert_allclose(out, _manual_acorr(x, 2))


def test_lb_stat_matches_manual_ljung_box_statistic() -> None:
    x = np.array([1.0, 2.0, 0.0, 4.0, 3.0, -1.0], dtype=np.float64)
    L = 3

    err, stat = lb_stat.py_func(x, L)

    assert err == OK
    assert stat == pytest.approx(_manual_ljung_box(x, L))


def test_lb_stat_caps_lag_at_available_observations() -> None:
    x = np.array([1.0, 2.0, 0.0, 4.0], dtype=np.float64)

    err, stat = lb_stat.py_func(x, 20)

    assert err == OK
    assert stat == pytest.approx(_manual_ljung_box(x, x.size - 1))


def test_lb_stat_reports_input_status_codes() -> None:
    err, stat = lb_stat.py_func(np.ones((2, 2), dtype=np.float64), 1)
    assert err == BAD_SHAPE
    assert BAD_SHAPE == TestStatus.BAD_SHAPE
    assert np.isnan(stat)

    err, stat = lb_stat.py_func(np.array([1.0], dtype=np.float64), 1)
    assert err == UDEF_VARIANCE
    assert np.isnan(stat)

    err, stat = lb_stat.py_func(np.array([1.0, 2.0], dtype=np.float64), 0)
    assert err == BAD_LAG
    assert BAD_LAG == TestStatus.BAD_LAG
    assert np.isnan(stat)

    err, stat = lb_stat.py_func(np.ones(3, dtype=np.float64), 1)
    assert err == UDEF_VARIANCE
    assert np.isnan(stat)


def test_ljung_box_builds_chi_square_test_result() -> None:
    x = np.array([1.0, 2.0, 0.0, 4.0, 3.0, -1.0], dtype=np.float64)
    L = 3

    out = ljung_box(x, L=L, alpha=0.1)
    expected_stat = _manual_ljung_box(x, L)

    assert out.test_name == "Ljung-Box (L=3)"
    assert out.dist is ReferenceDistribution.CHI2
    assert out.pval_method is PvalMethod.SF
    assert out.df == np.float64(3.0)
    assert out.alpha == np.float64(0.1)
    assert out.status is TestStatus.OK
    assert out.statistic == pytest.approx(expected_stat)
    assert out.pval == pytest.approx(chi2(df=3).sf(expected_stat))


def test_ljung_box_reports_effective_lag_in_df_and_name() -> None:
    x = np.array([1.0, 2.0, 0.0, 4.0], dtype=np.float64)

    out = ljung_box(x, L=20)

    assert out.test_name == "Ljung-Box (L=3)"
    assert out.df == np.float64(3.0)
    assert out.statistic == pytest.approx(_manual_ljung_box(x, 3))
    assert out.pval == pytest.approx(chi2(df=3).sf(out.statistic))


def test_ljung_box_returns_status_for_invalid_lag_without_raising() -> None:
    out = ljung_box(np.array([1.0, 2.0], dtype=np.float64), L=0)

    assert out.status is TestStatus.BAD_LAG
    assert np.isnan(out.statistic)
    assert np.isnan(out.pval)
