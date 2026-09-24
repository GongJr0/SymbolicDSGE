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


# Gamma's below-one, exponential, and Marsaglia-Tsang paths, plus Beta's
# Johnk, tiny-shape, and Gamma-ratio paths. Unequal Beta shapes catch swaps.
_ADDED_SAMPLERS = [
    pytest.param("standard_exponential", "standard_exponential", (), id="exp"),
    pytest.param("standard_gamma", "standard_gamma", (0.3,), id="gamma-small"),
    pytest.param("standard_gamma", "standard_gamma", (1.0,), id="gamma-unit"),
    pytest.param("standard_gamma", "standard_gamma", (7.5,), id="gamma-large"),
    pytest.param("chi2", "chisquare", (0.6,), id="chi2-small"),
    pytest.param("chi2", "chisquare", (9.0,), id="chi2-large"),
    pytest.param("beta", "beta", (0.3, 0.8), id="beta-johnk"),
    pytest.param("beta", "beta", (1e-110, 2e-110), id="beta-tiny"),
    pytest.param("beta", "beta", (0.3, 4.0), id="beta-mixed"),
    pytest.param("beta", "beta", (2.0, 7.0), id="beta-ratio"),
]


@pytest.mark.parametrize("name, numpy_name, params", _ADDED_SAMPLERS)
@pytest.mark.parametrize("seed", [0, 42, 20260724])
@pytest.mark.parametrize("n", [0, 1, 1000, 100_000])
def test_added_sampler_matches_numpy_and_advances_state(
    name, numpy_name, params, seed, n
):
    rng = np.random.default_rng(seed)
    reference = np.random.default_rng(seed)
    # Start from an already-used generator to catch accidental reseeding.
    rng.random(7)
    reference.random(7)
    got = getattr(native, name)(rng, n, *params)
    want = getattr(reference, numpy_name)(*params, size=n)
    assert got.shape == (n,)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got.view(np.uint64), want.view(np.uint64))
    np.testing.assert_array_equal(
        rng.bit_generator.random_raw(16), reference.bit_generator.random_raw(16)
    )


@pytest.mark.parametrize("name, numpy_name, params", _ADDED_SAMPLERS)
def test_added_philox_sampler_replays_and_preserves_prefix(name, numpy_name, params):
    draw = getattr(native, "philox_" + name)
    address = (17, 29, 41, 53)
    whole = draw(*address, 1000, *params)
    replay = draw(*address, 1000, *params)
    prefix = draw(*address, 13, *params)
    assert whole.shape == (1000,)
    assert whole.dtype == np.float64
    assert np.isfinite(whole).all()
    np.testing.assert_array_equal(whole.view(np.uint64), replay.view(np.uint64))
    np.testing.assert_array_equal(whole[:13].view(np.uint64), prefix.view(np.uint64))
    empty = draw(*address, 0, *params)
    assert empty.shape == (0,)
    assert empty.dtype == np.float64


@pytest.mark.parametrize("name, numpy_name, params", _ADDED_SAMPLERS)
def test_added_samplers_reject_negative_lengths_and_invalid_generators(
    name, numpy_name, params
):
    with pytest.raises(ValueError, match="n must be non-negative"):
        getattr(native, name)(np.random.default_rng(0), -1, *params)
    with pytest.raises(ValueError, match="n must be non-negative"):
        getattr(native, "philox_" + name)(0, 0, 0, 0, -1, *params)
    with pytest.raises((ValueError, AttributeError)):
        getattr(native, name)(object(), 1, *params)


@pytest.mark.parametrize(
    "name, params, index, parameter",
    [
        ("standard_gamma", (2.0,), 0, "a"),
        ("chi2", (2.0,), 0, "df"),
        ("beta", (2.0, 3.0), 0, "a"),
        ("beta", (2.0, 3.0), 1, "b"),
    ],
)
@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_sampler_parameters_are_validated_before_drawing(
    name, params, index, parameter, invalid
):
    params = list(params)
    params[index] = invalid
    rng = np.random.default_rng(42)
    reference = np.random.default_rng(42)
    with pytest.raises(ValueError, match=f"{parameter} must be finite and positive"):
        getattr(native, name)(rng, 5, *params)
    np.testing.assert_array_equal(
        rng.bit_generator.random_raw(16), reference.bit_generator.random_raw(16)
    )
    with pytest.raises(ValueError, match=f"{parameter} must be finite and positive"):
        getattr(native, "philox_" + name)(0, 0, 0, 0, 5, *params)
