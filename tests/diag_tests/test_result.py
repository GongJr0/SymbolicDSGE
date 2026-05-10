from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._diag_tests.result import MCResult


def test_mc_result_derives_n_from_trace_length() -> None:
    out = MCResult(
        test_name="demo",
        alpha=np.float64(0.05),
        statistic_trace=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        pval_trace=np.array([0.01, 0.20, 0.03], dtype=np.float64),
    )

    assert out.n == 3
    assert out.rejection_rate == pytest.approx(2.0 / 3.0)


def test_mc_result_raises_on_mismatched_trace_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        MCResult(
            test_name="demo",
            alpha=np.float64(0.05),
            statistic_trace=np.array([1.0, 2.0], dtype=np.float64),
            pval_trace=np.array([0.01], dtype=np.float64),
        )


def test_mc_result_raises_on_empty_traces() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MCResult(
            test_name="demo",
            alpha=np.float64(0.05),
            statistic_trace=np.array([], dtype=np.float64),
            pval_trace=np.array([], dtype=np.float64),
        )
