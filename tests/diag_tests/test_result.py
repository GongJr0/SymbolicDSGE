from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from SymbolicDSGE._diag_tests.distributions import PvalMethod, ReferenceDistribution
from SymbolicDSGE._diag_tests.result import MCResult


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


def test_pval_method_enum_members_dispatch_to_frozen_distribution() -> None:
    dist = chi2(df=2)
    statistic = np.array([1.0, 2.0], dtype=np.float64)

    assert set(PvalMethod.__members__) == {"CDF", "SF"}
    np.testing.assert_allclose(PvalMethod.CDF(dist, statistic), dist.cdf(statistic))
    np.testing.assert_allclose(PvalMethod.SF(dist, statistic), dist.sf(statistic))


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
