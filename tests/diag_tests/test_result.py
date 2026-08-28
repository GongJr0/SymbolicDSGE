from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2, f

from SymbolicDSGE._diag_tests.distributions import PvalMethod, ReferenceDistribution
from SymbolicDSGE._diag_tests.jb_lookup import JarqueBeraDist
from SymbolicDSGE._diag_tests.result import MCTestResult, TestResult as DiagTestResult
from SymbolicDSGE._diag_tests.status import TestStatus


def _mc_result(
    statistic_trace: np.ndarray,
    statuses: tuple[TestStatus, ...] | None = None,
    *,
    n_rep: int | None = None,
    retained_reps: np.ndarray | None = None,
    **metadata: object,
) -> MCTestResult:
    n_retained = statistic_trace.shape[0]
    if statuses is None:
        statuses = (TestStatus.OK,) * n_retained
    if n_rep is None:
        n_rep = n_retained
    if retained_reps is None:
        retained_reps = np.arange(n_retained, dtype=np.int_)
    return MCTestResult(
        statistic_trace=statistic_trace,
        n_retained=n_retained,
        retained_reps=retained_reps,
        n_rep=n_rep,
        _raw_status=np.asarray(statuses, dtype=np.int_),
        **metadata,
    )


def test_test_result_computes_p_value_from_reference_distribution() -> None:
    out = DiagTestResult(
        test_name="wald",
        dist=ReferenceDistribution.CHI2,
        df=np.float64(2.0),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.05),
        statistic=np.float64(10.0),
        status=TestStatus.OK,
    )

    assert out.pval == pytest.approx(chi2(df=2).sf(10.0))
    assert out.is_significant()
    assert out.status is TestStatus.OK


def test_test_result_can_defer_p_value_until_requested() -> None:
    out = DiagTestResult(
        test_name="wald",
        dist=ReferenceDistribution.CHI2,
        df=np.float64(2.0),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.05),
        statistic=np.float64(10.0),
        status=TestStatus.OK,
        _auto_pval=False,
    )

    assert out._pval is None
    assert "frozen_dist" not in out.__dict__

    assert out.compute_pval() == pytest.approx(chi2(df=2).sf(10.0))
    assert out._pval == pytest.approx(chi2(df=2).sf(10.0))
    assert "frozen_dist" in out.__dict__
    assert out.pval == pytest.approx(chi2(df=2).sf(10.0))


def test_test_result_supports_multi_df_reference_distribution() -> None:
    out = DiagTestResult(
        test_name="f_test",
        dist=ReferenceDistribution.F,
        df=[np.float64(2.0), np.float64(10.0)],
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.05),
        statistic=np.float64(3.0),
        status=TestStatus.OK,
    )

    assert out.df == [np.float64(2.0), np.float64(10.0)]
    assert out.pval == pytest.approx(f(dfn=2.0, dfd=10.0).sf(3.0))


def test_test_result_to_dict_excludes_frozen_distribution() -> None:
    out = DiagTestResult(
        test_name="wald",
        dist=ReferenceDistribution.CHI2,
        df=np.float64(2.0),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.05),
        statistic=np.float64(10.0),
        status=TestStatus.OK,
    )

    assert out.to_dict() == {
        "test_name": "wald",
        "dist": "chi2",
        "df": np.float64(2.0),
        "pval_method": "sf",
        "alpha": np.float64(0.05),
        "statistic": np.float64(10.0),
        "status": TestStatus.OK,
        "pval": out.pval,
    }


def test_mc_result_exposes_retention_metadata() -> None:
    statistic_trace = np.array([1.0, 10.0, 20.0], dtype=np.float64)
    out = _mc_result(
        statistic_trace,
        (TestStatus.OK,) * statistic_trace.size,
        test_name="demo",
        dist=ReferenceDistribution.CHI2,
        df=np.float64(2.0),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.05),
        n_rep=10,
        retained_reps=np.array([0, 4, 9], dtype=np.int_),
    )

    assert out.n_rep == 10
    assert out.n_retained == 3
    np.testing.assert_array_equal(out.retained_reps, [0, 4, 9])
    np.testing.assert_allclose(out.pval_trace, chi2(df=2).sf(statistic_trace))
    assert out.rejection_rate == pytest.approx(2.0 / 3.0)
    assert out.status_trace == (TestStatus.OK,) * 3


def test_mc_result_supports_multi_df_reference_distribution() -> None:
    statistic_trace = np.array([1.0, 3.0, 5.0], dtype=np.float64)
    out = _mc_result(
        statistic_trace,
        (TestStatus.OK,) * statistic_trace.size,
        test_name="f_test",
        dist=ReferenceDistribution.F,
        df=(np.float64(2.0), np.float64(10.0)),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.05),
    )

    np.testing.assert_allclose(out.pval_trace, f(dfn=2.0, dfd=10.0).sf(statistic_trace))


def test_pval_method_enum_members_dispatch_to_frozen_distribution() -> None:
    dist = chi2(df=2)
    statistic = np.array([1.0, 2.0], dtype=np.float64)

    assert set(PvalMethod.__members__) == {"CDF", "SF"}
    np.testing.assert_allclose(PvalMethod.CDF(dist, statistic), dist.cdf(statistic))
    np.testing.assert_allclose(PvalMethod.SF(dist, statistic), dist.sf(statistic))


def test_reference_distribution_freezes_t_distribution() -> None:
    frozen = ReferenceDistribution.t.freeze(np.float64(5.0))

    assert frozen.mean() == pytest.approx(0.0)


def test_reference_distribution_freezes_jb_lookup_with_integer_sample_size() -> None:
    frozen = ReferenceDistribution.JB_LOOKUP.freeze(100)

    assert isinstance(frozen, JarqueBeraDist)
    assert frozen.n == 100

    with pytest.raises(TypeError, match="exactly one"):
        ReferenceDistribution.JB_LOOKUP.freeze()
    with pytest.raises(TypeError, match="integer"):
        ReferenceDistribution.JB_LOOKUP.freeze(np.float64(100.0))


def test_test_and_mc_results_preserve_integer_jb_sample_size() -> None:
    statistic = np.float64(5.0)
    test_result = DiagTestResult(
        test_name="jarque_bera",
        dist=ReferenceDistribution.JB_LOOKUP,
        df=100,
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.05),
        statistic=statistic,
        status=TestStatus.OK,
    )
    mc_result = _mc_result(
        np.array([1.0, 5.0], dtype=np.float64),
        (TestStatus.OK, TestStatus.INSUFFICIENT_SAMPLES),
        test_name="jarque_bera",
        dist=ReferenceDistribution.JB_LOOKUP,
        df=np.int64(100),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.05),
    )

    assert test_result.df == 100
    assert test_result.pval == pytest.approx(JarqueBeraDist(100).sf(statistic))
    assert mc_result.df == 100
    np.testing.assert_allclose(
        mc_result.pval_trace,
        JarqueBeraDist(100).sf(mc_result.statistic_trace),
    )


def test_test_result_lazy_distribution_and_repeated_pval_access() -> None:
    out = DiagTestResult(
        test_name="wald",
        dist=ReferenceDistribution.CHI2,
        df=np.float64(2.0),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.05),
        statistic=np.float64(1.0),
        status=TestStatus.OK,
        _auto_pval=False,
    )

    assert "frozen_dist" not in out.__dict__
    frozen_dist = out.frozen_dist
    assert out.frozen_dist is frozen_dist
    first = out.compute_pval()
    assert out.compute_pval() == first


def test_mc_result_confidence_intervals_cover_wilson_normal_and_t_paths() -> None:
    out = _mc_result(
        np.array([0.1, 1.0, 3.0, 5.0], dtype=np.float64),
        (TestStatus.OK,) * 4,
        test_name="demo",
        dist=ReferenceDistribution.CHI2,
        df=np.float64(2.0),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.5),
    )

    wilson = out.rejection_rate_confidence_interval(wilson=True)
    binomial = out.rejection_rate_confidence_interval(wilson=False)
    pval_z = out.pval_confidence_interval(t_interval=False)
    pval_t = out.pval_confidence_interval(t_interval=True)
    z_interval = out.statistic_confidence_interval(t_interval=False)
    t_interval = out.statistic_confidence_interval(t_interval=True)

    assert 0.0 <= wilson[0] <= wilson[1] <= 1.0
    assert 0.0 <= binomial[0] <= binomial[1] <= 1.0
    assert wilson != binomial
    assert 0.0 <= pval_z[0] <= pval_z[1] <= 1.0
    assert pval_t[0] <= pval_z[0] and pval_z[1] <= pval_t[1]
    assert z_interval[0] <= z_interval[1]
    assert t_interval[0] <= t_interval[1]


def test_mc_result_pval_members_describe_the_pval_trace() -> None:
    out = _mc_result(
        np.array([0.1, 1.0, 3.0, 5.0], dtype=np.float64),
        (TestStatus.OK,) * 4,
        test_name="demo",
        dist=ReferenceDistribution.CHI2,
        df=np.float64(2.0),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.5),
    )

    assert out.pval_se == pytest.approx(
        out.pval_trace.std(ddof=1) / np.sqrt(out.n_retained)
    )
    p = out.rejection_rate
    assert out.rejection_rate_se == pytest.approx(np.sqrt(p * (1 - p) / out.n_retained))


def test_mc_result_summary_and_intervals() -> None:
    out = _mc_result(
        np.array([0.1, 1.0, 3.0, 5.0], dtype=np.float64),
        (TestStatus.OK,) * 4,
        test_name="demo",
        dist=ReferenceDistribution.CHI2,
        df=np.float64(2.0),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.5),
    )

    summary = out.summary()
    assert list(summary.index) == ["demo"]
    assert summary.index.name == "test"
    assert list(summary.columns) == [
        "statistic",
        "statistic_se",
        "pval",
        "reject_rate",
    ]
    assert summary.loc["demo", "statistic"] == pytest.approx(out.mean_statistic)
    assert summary.loc["demo", "reject_rate"] == pytest.approx(out.rejection_rate)

    intervals = out.intervals()
    assert list(intervals.index) == ["statistic", "pval", "reject_rate"]
    assert intervals.index.name == "quantity"
    assert list(intervals.columns) == ["ci_low", "ci_high"]
    for quantity in intervals.index:
        low, high = intervals.loc[quantity]
        column = "statistic" if quantity == "statistic" else quantity
        assert low <= summary.loc["demo", column] <= high


def test_mc_result_mc_se_needs_two_replications() -> None:
    out = _mc_result(
        np.array([1.0], dtype=np.float64),
        (TestStatus.OK,),
        test_name="demo",
        dist=ReferenceDistribution.CHI2,
        df=np.float64(2.0),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.5),
    )

    assert np.isnan(out.statistic_se)
    assert np.isnan(out.pval_se)
