# type: ignore
import numpy as np
import pandas as pd

from SymbolicDSGE.utils import math_utils


def test_hp_two_sided_reconstructs_input_for_numpy():
    x = np.array([1.0, 2.0, 4.0, 3.0, 5.0], dtype=float)
    trend, cycle = math_utils.HP_two_sided(x, lamb=1600.0)

    assert isinstance(trend, np.ndarray)
    assert isinstance(cycle, np.ndarray)
    assert trend.shape == x.shape
    assert cycle.shape == x.shape
    assert np.allclose(trend + cycle, x)


def test_hp_two_sided_preserves_series_index():
    idx = pd.date_range("2000-01-01", periods=6, freq="QE")
    s = pd.Series([1.0, 2.0, 3.5, 4.0, 5.5, 6.0], index=idx)
    trend, cycle = math_utils.HP_two_sided(s, lamb=1600.0)

    assert isinstance(trend, pd.Series)
    assert isinstance(cycle, pd.Series)
    assert trend.index.equals(idx)
    assert cycle.index.equals(idx)
    assert np.allclose((trend + cycle).values, s.values)


def test_hp_one_sided_reconstructs_input_for_numpy():
    x = np.array([1.0, 2.0, 4.0, 3.0, 5.0], dtype=float)
    trend, cycle = math_utils.HP_one_sided(x, lamb=1600.0)

    assert isinstance(trend, np.ndarray)
    assert isinstance(cycle, np.ndarray)
    assert trend.shape == x.shape
    assert cycle.shape == x.shape
    assert np.allclose(trend + cycle, x)


def test_hp_one_sided_preserves_series_index():
    idx = pd.date_range("2000-01-01", periods=6, freq="QE")
    s = pd.Series([2.0, 3.0, 2.0, 5.0, 6.0, 8.0], index=idx)
    trend, cycle = math_utils.HP_one_sided(s, lamb=1600.0)

    assert isinstance(trend, pd.Series)
    assert isinstance(cycle, pd.Series)
    assert trend.index.equals(idx)
    assert cycle.index.equals(idx)
    assert np.allclose((trend + cycle).values, s.values)


def test_annualized_log_percent_numpy_matches_formula():
    x = np.exp(np.array([0.0, 0.01, 0.03], dtype=float))
    out = math_utils.annualized_log_percent(x, periods_per_year=4)
    expected = (np.exp(np.array([0.01, 0.02]) * 4.0) - 1.0) * 100.0

    assert isinstance(out, np.ndarray)
    assert np.allclose(out, expected)


def test_annualized_log_percent_series_preserves_shifted_index():
    idx = pd.period_range("2000Q1", periods=4, freq="Q").to_timestamp()
    s = pd.Series(np.exp(np.array([0.0, 0.02, 0.03, 0.07], dtype=float)), index=idx)
    out = math_utils.annualized_log_percent(s, periods_per_year=4)

    assert isinstance(out, pd.Series)
    assert out.index.equals(idx[1:])


def test_demean_numpy_has_zero_mean():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    out = math_utils.demean(x)

    assert isinstance(out, np.ndarray)
    assert np.isclose(np.mean(out), 0.0)


def test_demean_series_preserves_index_and_zero_mean():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    s = pd.Series([2.0, 4.0, 8.0, 10.0], index=idx)
    out = math_utils.demean(s)

    assert isinstance(out, pd.Series)
    assert out.index.equals(idx)
    assert np.isclose(out.mean(), 0.0)


def test_detrend_numpy_removes_linear_trend():
    x = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=float)
    out = math_utils.detrend(x)

    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape
    assert np.allclose(out, np.zeros_like(x), atol=1e-10)


def test_detrend_series_preserves_index():
    idx = pd.date_range("2010-01-01", periods=5, freq="YE")
    s = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0], index=idx)
    out = math_utils.detrend(s)

    assert isinstance(out, pd.Series)
    assert out.index.equals(idx)
    assert np.allclose(out.values, np.zeros(len(s)), atol=1e-10)
