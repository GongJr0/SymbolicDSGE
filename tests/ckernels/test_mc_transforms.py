"""Parity coverage for native Monte Carlo sample transform kernels.

The production transform wrappers call this extension, so these tests compare
it only with the independent NumPy implementations in ``_oracles``.
"""

from __future__ import annotations

import numpy as np
import pytest

native = pytest.importorskip("SymbolicDSGE._ckernels.monte_carlo._transforms")

from _oracles import mc_transforms as oracle

RTOL = 1e-11
ATOL = 1e-11


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260727)


def _sample(
    rng: np.random.Generator, n_rows: int, n_columns: int, *, positive: bool = False
) -> np.ndarray:
    sample = rng.normal(10.0, 2.0, size=(n_rows, n_columns))
    if positive:
        sample = np.abs(sample) + 0.5
    return np.ascontiguousarray(sample, dtype=np.float64)


@pytest.mark.parametrize("shape", [(64, 3), (17, 1), (200, 5)])
@pytest.mark.parametrize("ddof", [0, 1])
def test_standardize_matches_numpy_oracle(
    rng: np.random.Generator, shape: tuple[int, int], ddof: int
) -> None:
    sample = _sample(rng, *shape)

    np.testing.assert_allclose(
        native.standardize_ax0(sample, ddof),
        oracle.standardize(sample, ddof),
        rtol=RTOL,
        atol=ATOL,
    )


def test_standardize_sets_constant_columns_to_zero(rng: np.random.Generator) -> None:
    sample = np.ascontiguousarray(
        np.column_stack([np.full(32, 3.0), rng.normal(size=32)]), dtype=np.float64
    )

    actual = native.standardize_ax0(sample, 0)

    np.testing.assert_array_equal(actual[:, 0], np.zeros(32))
    np.testing.assert_allclose(actual, oracle.standardize(sample), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("shape", [(64, 3), (17, 1), (200, 5)])
def test_log_and_log_diff_match_numpy_oracle(
    rng: np.random.Generator, shape: tuple[int, int]
) -> None:
    sample = _sample(rng, *shape, positive=True)

    np.testing.assert_allclose(
        native.log_transform(sample, 0.25),
        oracle.log(sample, 0.25),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        native.log_diff_transform(sample, 0.25),
        oracle.log_diff(sample, 0.25),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("shape", [(64, 3), (17, 1), (200, 5)])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_diff_matches_numpy_oracle_bitwise(
    rng: np.random.Generator, shape: tuple[int, int], order: int
) -> None:
    sample = _sample(rng, *shape)

    np.testing.assert_array_equal(
        native.diff_transform(sample, order), oracle.diff(sample, order)
    )


@pytest.mark.parametrize("shape", [(64, 3), (17, 1), (200, 5)])
@pytest.mark.parametrize("window", [1, 2, 5, 17, 64])
@pytest.mark.parametrize("ddof", [0, 1])
def test_rolling_transforms_match_numpy_oracle(
    rng: np.random.Generator, shape: tuple[int, int], window: int, ddof: int
) -> None:
    n_rows, _ = shape
    if window > n_rows or ddof >= window:
        pytest.skip("outside the native rolling-transform domain")
    sample = _sample(rng, *shape)

    np.testing.assert_allclose(
        native.rolling_mean(sample, window),
        oracle.rolling_mean(sample, window),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        native.rolling_var(sample, window, ddof),
        oracle.rolling_var(sample, window, ddof),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        native.rolling_std(sample, window, ddof),
        oracle.rolling_std(sample, window, ddof),
        rtol=RTOL,
        atol=ATOL,
    )


def test_single_period_and_full_width_windows(rng: np.random.Generator) -> None:
    sample = _sample(rng, 64, 3)

    np.testing.assert_array_equal(native.rolling_mean(sample, 1), sample)
    np.testing.assert_array_equal(
        native.rolling_var(sample, 1, 0), np.zeros_like(sample)
    )
    np.testing.assert_array_equal(
        native.rolling_std(sample, 1, 0), np.zeros_like(sample)
    )
    np.testing.assert_allclose(
        native.rolling_var(sample, 64, 1)[0],
        sample.var(axis=0, ddof=1),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda sample: native.rolling_mean(sample, 0), "window must be at least 1"),
        (
            lambda sample: native.rolling_mean(sample, 99),
            r"window \(99\) exceeds input length \(8\)",
        ),
        (
            lambda sample: native.rolling_var(sample, 3, 3),
            r"ddof \(3\) must be smaller than the window \(3\)",
        ),
        (lambda sample: native.diff_transform(sample, 0), "order must be at least 1"),
        (
            lambda sample: native.standardize_ax0(sample, 8),
            r"ddof \(8\) must be smaller than the sample length \(8\)",
        ),
    ],
)
def test_invalid_arguments_raise_contract_errors(call: object, match: str) -> None:
    sample = np.ascontiguousarray(np.arange(16.0).reshape(8, 2))

    with pytest.raises(ValueError, match=match):
        call(sample)  # type: ignore[operator]


def test_diff_and_log_diff_can_return_no_rows() -> None:
    sample = np.ascontiguousarray(np.arange(16.0).reshape(8, 2))

    assert native.diff_transform(sample, 8).shape == (0, 2)
    assert native.diff_transform(sample, 9).shape == (0, 2)
    assert native.log_diff_transform(sample[:1] + 1.0).shape == (0, 2)


def test_kernel_shim_coerces_input_to_contiguous_float64() -> None:
    source = np.arange(48.0).reshape(8, 6)
    strided = source[:, ::2]
    integers = np.arange(16).reshape(8, 2)

    np.testing.assert_allclose(
        native.rolling_mean(strided, 3), oracle.rolling_mean(strided, 3)
    )
    np.testing.assert_array_equal(
        native.diff_transform(integers, 1),
        oracle.diff(integers.astype(np.float64), 1),
    )
