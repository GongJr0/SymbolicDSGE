from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._diag_tests.hac_covariance import (
    andrews_bandwidth,
    andrews_bandwidth_matrix,
    hac_covariance,
    wooldridge_bandwidth,
)


GOLDEN_R = np.array(
    [
        [1.0, 2.0],
        [2.0, -1.0],
        [0.0, 1.0],
        [3.0, 0.0],
        [-1.0, 2.0],
    ],
    dtype=np.float64,
)

GOLDEN_HAC_CENTERED = {
    "bartlett": np.array(
        [
            [0.6666666666666667, -0.5599999999999998],
            [-0.5599999999999998, 0.6453333333333331],
        ],
        dtype=np.float64,
    ),
    "parzen": np.array(
        [
            [0.5629629629629630, -0.3733333333333333],
            [-0.3733333333333333, 0.6079999999999999],
        ],
        dtype=np.float64,
    ),
    "qs": np.array(
        [
            [0.4104387018488457, -0.4840134757411693],
            [-0.4840134757411693, 0.5017280910104844],
        ],
        dtype=np.float64,
    ),
}


@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "qs"])
@pytest.mark.parametrize("nopython", [False, True])
def test_hac_covariance_matches_golden_output_for_all_kernels_and_backends(
    kernel: str,
    nopython: bool,
) -> None:
    out = hac_covariance(
        GOLDEN_R,
        kernel=kernel,
        bandwidth=2,
        center=True,
        nopython=nopython,
    )

    np.testing.assert_allclose(out, GOLDEN_HAC_CENTERED[kernel], rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    "series",
    [
        np.zeros(20, dtype=np.float64),
        np.ones(20, dtype=np.float64),
        np.array([(-1.0) ** i for i in range(20)], dtype=np.float64),
    ],
)
def test_andrews_bandwidth_returns_minimum_for_degenerate_or_negative_rhat(
    series: np.ndarray,
) -> None:
    assert andrews_bandwidth(series) == 1


def test_andrews_bandwidth_matrix_handles_negative_rhat_columns() -> None:
    series = np.array([(-1.0) ** i for i in range(20)], dtype=np.float64)
    r = np.column_stack([series])

    assert andrews_bandwidth_matrix(r, kernel="qs") == 1
    out = hac_covariance(r, kernel="qs", bandwidth="auto", nopython=False)

    assert out.shape == (1, 1)
    assert np.isfinite(out).all()


def test_wooldridge_bandwidth_matches_textbook_rule() -> None:
    r = np.zeros((200, 3), dtype=np.float64)

    assert wooldridge_bandwidth(r) == 4


def test_hac_covariance_raises_on_bad_shape() -> None:
    with pytest.raises(ValueError, match="2D"):
        hac_covariance(np.arange(5.0))


def test_hac_covariance_raises_on_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least 2 observations"):
        hac_covariance(np.ones((1, 2), dtype=np.float64))


def test_hac_covariance_raises_on_bad_bandwidth() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        hac_covariance(GOLDEN_R, bandwidth=-1)

    with pytest.raises(ValueError, match="Unsupported bandwidth"):
        hac_covariance(GOLDEN_R, bandwidth="bad")
