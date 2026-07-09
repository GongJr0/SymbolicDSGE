import numpy as np
from numpy import float64
from scipy.stats import chi2
from scipy.stats import jarque_bera as scipy_jarque_bera

from SymbolicDSGE._diag_tests.distributions import PvalMethod, ReferenceDistribution
from SymbolicDSGE._diag_tests.jarque_bera import jarque_bera, jb_stat
from SymbolicDSGE._diag_tests.jb_lookup import (
    JB_N_GRID,
    JB_PVAL_GRID,
    JB_SMALL_N_CRITICAL_VALUES,
    JarqueBeraDist,
)
from SymbolicDSGE._diag_tests.status import TestStatus


def test_jb_stat_matches_scipy_and_reports_small_sample_status() -> None:
    x = np.array([0.2, -1.0, 0.4, 1.8, -0.3, 0.9], dtype=np.float64)

    status, statistic = jb_stat(x)
    bad_status, bad_statistic = jb_stat(x.reshape(2, 3))

    assert status == TestStatus.INSUFFICIENT_SAMPLES
    np.testing.assert_allclose(statistic, scipy_jarque_bera(x).statistic)
    assert bad_status == TestStatus.BAD_SHAPE
    assert np.isnan(bad_statistic)


def test_jb_stat_handles_empty_input() -> None:
    status, statistic = jb_stat(np.array([], dtype=np.float64))

    assert status == TestStatus.INSUFFICIENT_SAMPLES
    assert np.isnan(statistic)


def test_jb_stat_rejects_non_finite_input() -> None:
    # A NaN/inf in the series (e.g. log of a non-positive value upstream) makes
    # the variance non-finite. `m2 <= 0.0` alone can't catch it -- every NaN
    # comparison is False -- so the kernel must guard isfinite rather than fall
    # through to an OK result carrying a NaN statistic.
    base = np.linspace(-2.0, 2.0, 50, dtype=np.float64)
    for bad in (np.nan, np.inf, -np.inf):
        x = base.copy()
        x[3] = bad
        status, statistic = jb_stat(x)
        assert status == TestStatus.UDEF_VARIANCE
        assert np.isnan(statistic)


def test_jarque_bera_flags_non_finite_series_instead_of_ok() -> None:
    x = np.linspace(-2.0, 2.0, 100, dtype=np.float64)
    x[10] = np.nan

    out = jarque_bera(x)

    assert out.status is TestStatus.UDEF_VARIANCE
    assert np.isnan(out.statistic)


def test_jarque_bera_returns_lookup_backed_test_result() -> None:
    x = np.linspace(-2.0, 2.0, 100, dtype=np.float64)

    out = jarque_bera(x)

    assert out.test_name == "jarque_bera"
    assert out.dist is ReferenceDistribution.JB_LOOKUP
    assert out.df == x.size
    assert out.pval_method is PvalMethod.SF
    assert out.status is TestStatus.OK
    assert out.pval == JarqueBeraDist(x.size).sf(out.statistic)


def test_small_n_distribution_matches_lookup_and_inverse_semantics() -> None:
    n_index = 5
    p_index = 5
    n = int(JB_N_GRID[n_index])
    upper_tail_probability = JB_PVAL_GRID[p_index]
    critical_value = JB_SMALL_N_CRITICAL_VALUES[p_index, n_index]
    dist = JarqueBeraDist(n)

    np.testing.assert_allclose(dist.sf(critical_value), upper_tail_probability)
    np.testing.assert_allclose(dist.cdf(critical_value), 1 - upper_tail_probability)
    np.testing.assert_allclose(dist.isf(upper_tail_probability), critical_value)
    np.testing.assert_allclose(dist.ppf(1 - upper_tail_probability), critical_value)


def test_small_n_distribution_interpolates_between_sample_sizes() -> None:
    probability_index = 5
    n_lo_index = 5
    n_hi_index = 6
    n = int((JB_N_GRID[n_lo_index] + JB_N_GRID[n_hi_index]) // 2)
    probability = JB_PVAL_GRID[probability_index]
    expected = (
        JB_SMALL_N_CRITICAL_VALUES[probability_index, n_lo_index]
        + JB_SMALL_N_CRITICAL_VALUES[probability_index, n_hi_index]
    ) / 2.0

    np.testing.assert_allclose(JarqueBeraDist(n).isf(probability), expected)


def test_small_n_distribution_supports_arrays_and_distribution_boundaries() -> None:
    dist = JarqueBeraDist(100)
    values = np.array([0.0, 1.0, 5.0, np.inf, np.nan], dtype=np.float64)
    probabilities = np.array([0.0, 0.05, 0.5, 1.0, np.nan], dtype=np.float64)

    cdf = np.asarray(dist.cdf(values), dtype=np.float64)
    sf = np.asarray(dist.sf(values), dtype=np.float64)
    isf = np.asarray(dist.isf(probabilities), dtype=np.float64)
    ppf = np.asarray(dist.ppf(probabilities), dtype=np.float64)

    assert cdf.shape == values.shape
    assert sf.shape == values.shape
    np.testing.assert_allclose(cdf[:-1] + sf[:-1], np.ones(values.size - 1))
    assert np.isnan(cdf[-1])
    assert np.isnan(sf[-1])
    assert np.isinf(isf[0])
    assert isf[3] == 0.0
    assert ppf[0] == 0.0
    assert np.isinf(ppf[3])
    assert np.isnan(isf[-1])
    assert np.isnan(ppf[-1])


def test_large_n_distribution_uses_initialized_chi2_frozen_distribution() -> None:
    dist = JarqueBeraDist(int(JB_N_GRID[-1]) + 1)
    values = np.array([0.5, 2.0, 5.0], dtype=np.float64)
    probabilities = np.array([0.1, 0.5, 0.9], dtype=np.float64)

    np.testing.assert_allclose(dist.cdf(values), chi2(df=2).cdf(values))
    np.testing.assert_allclose(dist.sf(values), chi2(df=2).sf(values))
    np.testing.assert_allclose(dist.ppf(probabilities), chi2(df=2).ppf(probabilities))
    np.testing.assert_allclose(dist.isf(probabilities), chi2(df=2).isf(probabilities))
    assert dist.mean() == 2.0
