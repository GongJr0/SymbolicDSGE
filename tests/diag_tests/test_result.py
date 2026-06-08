from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2, f

from SymbolicDSGE._diag_tests.distributions import PvalMethod, ReferenceDistribution
from SymbolicDSGE._diag_tests.jb_lookup import JarqueBeraDist
from SymbolicDSGE._diag_tests.result import MCResult, TestResult as DiagTestResult
from SymbolicDSGE._diag_tests.result import _df_args, _normalize_df
from SymbolicDSGE._diag_tests.status import TestStatus


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
    assert out._frozen_dist is None

    assert out.compute_pval() == pytest.approx(chi2(df=2).sf(10.0))
    assert out._pval == pytest.approx(chi2(df=2).sf(10.0))
    assert out._frozen_dist is not None
    assert out.pval == pytest.approx(chi2(df=2).sf(10.0))


def test_test_result_supports_multi_df_reference_distribution() -> None:
    out = DiagTestResult(
        test_name="f_test",
        dist=ReferenceDistribution.F,
        df=(np.float64(2.0), np.float64(10.0)),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.05),
        statistic=np.float64(3.0),
        status=TestStatus.OK,
    )

    assert out.df == (np.float64(2.0), np.float64(10.0))
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


def test_mc_result_derives_n_from_trace_length() -> None:
    statistic_trace = np.array([1.0, 10.0, 20.0], dtype=np.float64)
    out = MCResult(
        test_name="demo",
        dist=ReferenceDistribution.CHI2,
        df=np.float64(2.0),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.05),
        statistic_trace=statistic_trace,
    )

    assert out.n == 3
    np.testing.assert_allclose(out.pval_trace, chi2(df=2).sf(statistic_trace))
    assert out.rejection_rate == pytest.approx(2.0 / 3.0)


def test_mc_result_supports_multi_df_reference_distribution() -> None:
    statistic_trace = np.array([1.0, 3.0, 5.0], dtype=np.float64)
    out = MCResult(
        test_name="f_test",
        dist=ReferenceDistribution.F,
        df=[np.float64(2.0), np.float64(10.0)],
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.05),
        statistic_trace=statistic_trace,
    )

    assert out.df == (np.float64(2.0), np.float64(10.0))
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
    mc_result = MCResult(
        test_name="jarque_bera",
        dist=ReferenceDistribution.JB_LOOKUP,
        df=np.int64(100),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.05),
        statistic_trace=np.array([1.0, 5.0], dtype=np.float64),
    )

    assert test_result.df == 100
    assert test_result.pval == pytest.approx(JarqueBeraDist(100).sf(statistic))
    assert mc_result.df == 100
    np.testing.assert_allclose(
        mc_result.pval_trace,
        JarqueBeraDist(100).sf(mc_result.statistic_trace),
    )


def test_mc_result_raises_on_empty_traces() -> None:
    with pytest.raises(ValueError, match="statistic_trace must be non-empty"):
        MCResult(
            test_name="demo",
            dist=ReferenceDistribution.CHI2,
            df=np.float64(2.0),
            pval_method=PvalMethod.SF,
            alpha=np.float64(0.05),
            statistic_trace=np.array([], dtype=np.float64),
        )


def test_mc_result_raises_on_non_1d_statistic_trace() -> None:
    with pytest.raises(ValueError, match="1D array"):
        MCResult(
            test_name="demo",
            dist=ReferenceDistribution.CHI2,
            df=np.float64(2.0),
            pval_method=PvalMethod.SF,
            alpha=np.float64(0.05),
            statistic_trace=np.ones((2, 2), dtype=np.float64),
        )


def test_mc_result_raises_on_unsupported_reference_distribution() -> None:
    with pytest.raises(ValueError, match="Unsupported reference distribution"):
        MCResult(
            test_name="demo",
            dist="not_a_distribution",
            df=np.float64(2.0),
            pval_method=PvalMethod.SF,
            alpha=np.float64(0.05),
            statistic_trace=np.array([1.0], dtype=np.float64),
        )


def test_df_normalization_accepts_scalars_and_rejects_bad_sequences() -> None:
    assert _normalize_df(np.array(2.0, dtype=np.float64)) == np.float64(2.0)
    assert _df_args(np.float64(2.0)) == (np.float64(2.0),)
    assert _normalize_df(np.int64(2)) == 2
    assert _df_args(2) == (2,)

    with pytest.raises(ValueError, match="1D"):
        _normalize_df(np.ones((1, 1), dtype=np.float64))
    with pytest.raises(ValueError, match="non-empty"):
        _normalize_df(np.array([], dtype=np.float64))
    with pytest.raises(TypeError, match="numeric"):
        _normalize_df("2")
    with pytest.raises(ValueError, match="non-empty"):
        _normalize_df([])


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

    assert out._frozen_dist is None
    assert out.frozen_dist is out._frozen_dist
    first = out.compute_pval()
    assert out.compute_pval() == first


def test_mc_result_confidence_intervals_cover_wilson_normal_and_t_paths() -> None:
    out = MCResult(
        test_name="demo",
        dist=ReferenceDistribution.CHI2,
        df=np.float64(2.0),
        pval_method=PvalMethod.SF,
        alpha=np.float64(0.5),
        statistic_trace=np.array([0.1, 1.0, 3.0, 5.0], dtype=np.float64),
    )

    wilson = out.pval_confidence_interval(wilson=True)
    normal = out.pval_confidence_interval(wilson=False)
    z_interval = out.statistic_confidence_interval(t_interval=False)
    t_interval = out.statistic_confidence_interval(t_interval=True)

    assert 0.0 <= wilson[0] <= wilson[1] <= 1.0
    assert normal[0] <= normal[1]
    assert z_interval[0] <= z_interval[1]
    assert t_interval[0] <= t_interval[1]
