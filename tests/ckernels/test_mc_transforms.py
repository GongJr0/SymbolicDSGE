"""Parity tests for the native Monte Carlo transform kernels
(``_ckernels.monte_carlo``).

Each kernel is checked against the Python op in
``monte_carlo.operations.transforms.ops``, which stays the library's
implementation: nothing imports these kernels yet.

``diff`` and ``log_diff`` agree bitwise because they perform the same
subtractions in the same order as ``np.diff``. Everything else agrees to a
tolerance, for two reasons worth keeping in mind when a case here is tightened:

- ``log`` calls libm while NumPy switches to a SIMD implementation above a size
  threshold, so a large sample differs on a few elements by 1 ulp,
- the rolling kernels slide the window by removing the leaving observation from
  the Welford state, where NumPy recomputes each window from scratch.
"""

from __future__ import annotations

import numpy as np
import pytest

native = pytest.importorskip("SymbolicDSGE._ckernels.monte_carlo")

from SymbolicDSGE.monte_carlo.operations.transforms import ops

RTOL = 1e-11
ATOL = 1e-11

# The Python ops take the executor's keyword contract; none of them read it.
_OP_KW = dict(context=None, reference=None, dgp=None, rep_idx=0)

_SHAPES = [(64, 3), (17, 1), (200, 5)]


def _sample(rng, n, p, *, positive=False):
    """A sample with a mean far from zero, where cancellation actually bites."""
    x = rng.normal(10.0, 2.0, size=(n, p))
    if positive:
        x = np.abs(x) + 0.5
    return np.ascontiguousarray(x)


@pytest.fixture
def rng():
    return np.random.default_rng(20260727)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("ddof", [0, 1])
def test_standardize_matches_python_op(rng, shape, ddof) -> None:
    x = _sample(rng, *shape)

    np.testing.assert_allclose(
        native.standardize_ax0(x, ddof),
        ops.run_standardize(sample=x, ddof=ddof, **_OP_KW),
        rtol=RTOL,
        atol=ATOL,
    )


def test_standardize_zeros_a_constant_column(rng) -> None:
    # A zero-variance column scales to zeros rather than dividing through.
    x = np.ascontiguousarray(np.column_stack([np.full(32, 3.0), rng.normal(size=32)]))

    out = native.standardize_ax0(x, 0)

    np.testing.assert_array_equal(out[:, 0], np.zeros(32))
    np.testing.assert_allclose(
        out, ops.run_standardize(sample=x, ddof=0, **_OP_KW), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("shape", _SHAPES)
def test_log_matches_python_op(rng, shape) -> None:
    x = _sample(rng, *shape, positive=True)

    np.testing.assert_allclose(
        native.log(x, 0.25),
        ops.run_log(sample=x, offset=0.25, **_OP_KW),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("shape", _SHAPES)
def test_log_diff_matches_python_op(rng, shape) -> None:
    x = _sample(rng, *shape, positive=True)
    want = ops.run_log_diff(sample=x, offset=0.25, **_OP_KW)

    got = native.log_diff(x, 0.25)

    assert got.shape == want.shape == (shape[0] - 1, shape[1])
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("order", [1, 2, 3])
def test_diff_matches_python_op_bitwise(rng, shape, order) -> None:
    x = _sample(rng, *shape)
    want = ops.run_diff(sample=x, order=order, **_OP_KW)

    got = native.diff(x, order)

    assert got.shape == want.shape == (shape[0] - order, shape[1])
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("window", [1, 5, 64])
def test_rolling_mean_matches_python_op(rng, shape, window) -> None:
    n, p = shape
    if window > n:
        pytest.skip("window wider than the sample is rejected, not compared")
    x = _sample(rng, n, p)
    want = ops.run_rolling_mean(sample=x, window=window, **_OP_KW)

    got = native.rolling_mean(x, window)

    assert got.shape == want.shape == (n - window + 1, p)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("window", [2, 5, 64])
@pytest.mark.parametrize("ddof", [0, 1])
def test_rolling_var_and_std_match_python_op(rng, shape, window, ddof) -> None:
    n, p = shape
    if window > n:
        pytest.skip("window wider than the sample is rejected, not compared")
    x = _sample(rng, n, p)

    np.testing.assert_allclose(
        native.rolling_var(x, window, ddof),
        ops.run_rolling_var(sample=x, window=window, ddof=ddof, **_OP_KW),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        native.rolling_std(x, window, ddof),
        ops.run_rolling_std(sample=x, window=window, ddof=ddof, **_OP_KW),
        rtol=RTOL,
        atol=ATOL,
    )


def test_rolling_var_is_the_square_of_rolling_std(rng) -> None:
    x = _sample(rng, 128, 4)

    np.testing.assert_allclose(
        native.rolling_std(x, 12, 1) ** 2,
        native.rolling_var(x, 12, 1),
        rtol=1e-12,
        atol=1e-12,
    )


def test_single_period_window_reproduces_the_sample(rng) -> None:
    # A one-wide window is the degenerate slide: mean is the observation and
    # the spread is zero, which exercises the reduced-window branch.
    x = _sample(rng, 32, 2)

    np.testing.assert_array_equal(native.rolling_mean(x, 1), x)
    np.testing.assert_array_equal(native.rolling_var(x, 1, 0), np.zeros_like(x))
    np.testing.assert_array_equal(native.rolling_std(x, 1, 0), np.zeros_like(x))


def test_full_width_window_reproduces_the_whole_sample_moment(rng) -> None:
    x = _sample(rng, 48, 3)

    np.testing.assert_allclose(
        native.rolling_var(x, 48, 1)[0], x.var(axis=0, ddof=1), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda x: native.rolling_mean(x, 99), "rejected its arguments"),
        (lambda x: native.rolling_mean(x, 0), "rejected its arguments"),
        (lambda x: native.rolling_var(x, 3, 3), "rejected its arguments"),
        (lambda x: native.rolling_std(x, 3, 4), "rejected its arguments"),
        (lambda x: native.diff(x, 0), "order must be at least 1"),
        (lambda x: native.standardize_ax0(x, 8), "rejected its arguments"),
    ],
)
def test_kernels_reject_arguments_they_are_not_defined_on(call, match) -> None:
    x = np.ascontiguousarray(np.arange(16.0).reshape(8, 2))

    with pytest.raises(ValueError, match=match):
        call(x)


def test_transforms_that_consume_every_row_return_no_rows() -> None:
    x = np.ascontiguousarray(np.arange(16.0).reshape(8, 2))

    assert native.diff(x, 8).shape == (0, 2)
    assert native.diff(x, 9).shape == (0, 2)
    assert native.log_diff(x[:1] + 1.0).shape == (0, 2)


def test_kernels_accept_non_contiguous_and_non_float_input() -> None:
    # The shim copies through ``ascontiguousarray``, so a strided view or an
    # integer array is marshalled rather than rejected.
    source = np.arange(48.0).reshape(8, 6)
    strided = source[:, ::2]
    integers = np.arange(16).reshape(8, 2)

    np.testing.assert_array_equal(
        native.diff(strided, 1), np.diff(np.ascontiguousarray(strided), axis=0)
    )
    np.testing.assert_array_equal(
        native.diff(integers, 1), np.diff(integers.astype(np.float64), axis=0)
    )
