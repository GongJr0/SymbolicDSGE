"""Parity tests: native ``_ckernels.rng`` draws vs numpy's own Generator (#328).

The bridge borrows a numpy ``Generator``'s ``bitgen_t*`` and calls numpy's own
ziggurat / uniform fill (linked from ``npyrandom``), so native draws must be
**bit-identical** to ``rng.standard_normal()`` / ``rng.random()`` on the same
generator state, and must advance that shared state so numpy and native calls can
be interleaved or continued on one generator. numpy is its own oracle here; there
is no separate reference kernel.
"""

from __future__ import annotations

import numpy as np
import pytest

native = pytest.importorskip("SymbolicDSGE._ckernels.rng")

_SEEDS = [0, 1, 42, 12345, 2**31, 20260724]
# Sizes span the ziggurat's variable per-draw consumption (the tail rejection
# path only triggers on longer runs), and include 1 and a large batch.
_SIZES = [1, 2, 7, 64, 1000, 100_000]


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("n", _SIZES)
def test_standard_normal_bit_parity(seed, n):
    got = native.standard_normal(np.random.default_rng(seed), n)
    want = np.random.default_rng(seed).standard_normal(n)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("n", _SIZES)
def test_standard_uniform_bit_parity(seed, n):
    got = native.standard_uniform(np.random.default_rng(seed), n)
    want = np.random.default_rng(seed).random(n)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, want)


def test_interleaved_draws_share_one_stream():
    """Alternating native and numpy draws on the *same* generator match a single
    numpy generator drawing the same sequence: the native call advances numpy's
    own PCG64 state, it does not fork a private one."""
    r_native = np.random.default_rng(7)
    r_ref = np.random.default_rng(7)

    got = np.concatenate(
        [
            native.standard_normal(r_native, 3),
            native.standard_uniform(r_native, 5),
            native.standard_normal(r_native, 2),
        ]
    )
    want = np.concatenate(
        [
            r_ref.standard_normal(3),
            r_ref.random(5),
            r_ref.standard_normal(2),
        ]
    )
    np.testing.assert_array_equal(got, want)


def test_native_then_numpy_continues_state():
    """After a native draw, numpy's own methods continue the stream in parity: a
    native draw of n then a numpy draw of m equals one numpy draw of n+m."""
    rng = np.random.default_rng(99)
    ref = np.random.default_rng(99)

    head = native.standard_normal(rng, 4)
    tail = rng.standard_normal(6)  # numpy continues from the advanced state

    full = ref.standard_normal(10)
    np.testing.assert_array_equal(np.concatenate([head, tail]), full)


def test_zero_length_returns_empty_and_does_not_advance():
    rng = np.random.default_rng(3)
    ref = np.random.default_rng(3)

    empty = native.standard_normal(rng, 0)
    assert empty.shape == (0,)
    assert empty.dtype == np.float64
    # A zero-length draw must not consume any randomness.
    np.testing.assert_array_equal(rng.standard_normal(5), ref.standard_normal(5))


@pytest.mark.parametrize("fn", [native.standard_normal, native.standard_uniform])
def test_negative_length_raises(fn):
    with pytest.raises(ValueError):
        fn(np.random.default_rng(0), -1)


@pytest.mark.parametrize("fn", [native.standard_normal, native.standard_uniform])
def test_non_generator_raises(fn):
    with pytest.raises((ValueError, AttributeError)):
        fn(object(), 3)
