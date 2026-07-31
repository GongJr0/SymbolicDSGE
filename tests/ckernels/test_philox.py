"""The owned counter-based engine behind the MC shock draw (#374).

Unlike the borrowed-PCG64 bridge in ``test_rng_parity``, this engine is ours:
the MC runner parallelizes over replications and hands a step only its
``rep_idx``, and a borrowed ``bitgen_t`` exposes neither a reseed nor a jump, so
per-replication streams have to come from a counter rather than a seeding pass.

We own the *engine*, but the algorithm is standard Philox4x64-10 and numpy ships
it, so ``numpy.random.Philox`` is still an exact oracle. The one deliberate
divergence is the counter origin: this engine advances the block counter before
emitting, so its stream from ``stream=(s0, s1)`` equals numpy's from
``counter = s0 << 128 | s1 << 192``. The transforms are numpy's own (linked from
``npyrandom``), which is why the normal and uniform fills land on parity too.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator, Philox

native = pytest.importorskip("SymbolicDSGE._ckernels.rng")

_KEYS = [
    (0, 0),
    (1, 0),
    (0, 1),
    (0xDEADBEEF12345678, 0x0BADC0DE55AA33CC),
    (2**64 - 1, 2**64 - 1),
]
# Sizes span the ziggurat's variable per-draw consumption (its tail rejection
# path only triggers on longer runs) and straddle the engine's 4-word block.
_SIZES = [1, 2, 4, 5, 7, 64, 1000, 100_000]


def _numpy_philox(key0: int, key1: int, stream0: int = 0, stream1: int = 0):
    """The numpy generator this engine's ``(key, stream)`` selects."""
    return Generator(
        Philox(counter=(stream0 << 128) | (stream1 << 192), key=key0 | (key1 << 64))
    )


@pytest.mark.parametrize("key", _KEYS)
@pytest.mark.parametrize("n", _SIZES)
def test_raw_matches_numpy_philox(key, n):
    got = native.philox_raw(*key, 0, 0, n)
    want = _numpy_philox(*key).bit_generator.random_raw(n)
    assert got.dtype == np.uint64
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("key", _KEYS)
@pytest.mark.parametrize("n", _SIZES)
def test_standard_normal_matches_numpy_philox(key, n):
    got = native.philox_standard_normal(*key, 0, 0, n)
    want = _numpy_philox(*key).standard_normal(n)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("key", _KEYS)
@pytest.mark.parametrize("n", _SIZES)
def test_standard_uniform_matches_numpy_philox(key, n):
    got = native.philox_standard_uniform(*key, 0, 0, n)
    want = _numpy_philox(*key).random(n)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("stream", [(0, 0), (1, 0), (37, 0), (0, 1), (2**63, 5)])
def test_stream_words_select_the_counter_high_half(stream):
    """The stream words are what ``rep_idx`` will ride on, so their mapping onto
    numpy's counter has to hold across the whole 128-bit range, not just small
    replication counts."""
    got = native.philox_standard_normal(11, 22, *stream, 64)
    want = _numpy_philox(11, 22, *stream).standard_normal(64)
    np.testing.assert_array_equal(got, want)


def test_same_key_and_stream_replays_exactly():
    first = native.philox_raw(7, 3, 5, 0, 32)
    second = native.philox_raw(7, 3, 5, 0, 32)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "a,b",
    [
        ((7, 3, 5, 0), (8, 3, 5, 0)),  # key0
        ((7, 3, 5, 0), (7, 4, 5, 0)),  # key1
        ((7, 3, 5, 0), (7, 3, 6, 0)),  # stream0, the replication axis
        ((7, 3, 5, 0), (7, 3, 5, 1)),  # stream1
    ],
)
def test_distinct_selectors_give_distinct_streams(a, b):
    assert not np.array_equal(native.philox_raw(*a, 64), native.philox_raw(*b, 64))


def test_consecutive_replication_indices_are_independent():
    """``rep_idx`` maps straight onto ``stream0``, so adjacent replications must
    not share a prefix the way adjacent seeds of a hashed seeder can."""
    draws = [native.philox_standard_normal(0, 0, rep, 0, 16) for rep in range(8)]
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not np.array_equal(draws[i], draws[j])


@pytest.mark.parametrize("n", [1, 4, 5, 64])
def test_draws_are_a_prefix_of_a_longer_draw(n):
    """A short fill is the head of a long one, so the number of shocks a
    replication asks for cannot change the values it gets."""
    long = native.philox_standard_normal(3, 1, 42, 0, 1000)
    np.testing.assert_array_equal(
        native.philox_standard_normal(3, 1, 42, 0, n), long[:n]
    )


@pytest.mark.parametrize(
    "fn",
    [
        native.philox_raw,
        native.philox_standard_normal,
        native.philox_standard_uniform,
    ],
)
def test_zero_length_returns_empty(fn):
    out = fn(1, 2, 3, 4, 0)
    assert out.shape == (0,)


@pytest.mark.parametrize(
    "fn",
    [
        native.philox_raw,
        native.philox_standard_normal,
        native.philox_standard_uniform,
    ],
)
def test_negative_length_raises(fn):
    with pytest.raises(ValueError):
        fn(1, 2, 3, 4, -1)


def test_uniform_stays_in_the_unit_interval():
    u = native.philox_standard_uniform(0, 0, 0, 0, 200_000)
    assert u.min() >= 0.0
    assert u.max() < 1.0
