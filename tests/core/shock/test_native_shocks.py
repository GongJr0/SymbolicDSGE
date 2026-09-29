"""Native shock sampling, family dispatch, and stream addressing."""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE import Shock
from SymbolicDSGE._ckernels.core._shocks import ShockCode, native_shock_plan
from SymbolicDSGE._ckernels.rng import (
    philox_beta,
    philox_chi2,
    philox_standard_exponential,
    philox_standard_gamma,
    philox_standard_normal,
    philox_standard_uniform,
)
from SymbolicDSGE.core.shock.generators import ShockPath
from SymbolicDSGE.core.shock.plan import ShockEntry, get_native_shock_plan
from SymbolicDSGE.core.shock.spec import resolve_shock_plan

from scipy.stats import beta, expon, gamma, t, uniform

T = 16

#: Both sides share the engine fill, and differ from there: the kernel scales by
#: a reciprocal it computes once, the reconstruction divides. That is a few ulps
#: apart, and orders of magnitude tighter than any wrong constant.
RTOL = 1e-12
ATOL = 1e-14


def _plan(solved_test_model, shocks, shock_scale=1.0):
    resolved = resolve_shock_plan(solved_test_model.compiled, shocks, T)
    return get_native_shock_plan(resolved, T, shock_scale)


def _entries(solved_test_model, shocks):
    return resolve_shock_plan(solved_test_model.compiled, shocks, T).entries


def test_a_spec_with_no_entries_lowers_to_no_plan(solved_test_model) -> None:
    assert _plan(solved_test_model, {}) is None


def test_a_mixed_spec_splits_by_family(solved_test_model) -> None:
    resolved = resolve_shock_plan(
        solved_test_model.compiled,
        {
            ("e_u",): Shock("norm", seed=0),
            ("e_v",): Shock(t, seed=1, dist_kwargs={"df": 5}),
        },
        T,
    )
    native = {entry.key for entry in resolved.native.entries}
    python = {entry.key for entry in resolved.python.entries}

    assert native == {("e_u",)}
    assert python == {("e_v",)}
    assert native.isdisjoint(python)
    assert native | python == {entry.key for entry in resolved.entries}


# --- the draw itself --------------------------------------------------------


def test_univariate_normal_draw_is_the_scaled_engine_stream(solved_test_model) -> None:
    shocks = {("e_u",): Shock("norm", seed=7)}
    (entry,) = _entries(solved_test_model, shocks)
    block = _plan(solved_test_model, shocks).draw(3)

    z = philox_standard_normal(entry._native_seed_key, 0, 3, 0, T)
    expected = np.zeros((T, solved_test_model.compiled.n_exog))
    expected[:, entry.indices[0]] = z * entry.factor[0, 0]

    np.testing.assert_array_equal(block, expected)


def test_multivariate_normal_draw_applies_the_covariance_factor(
    solved_test_model,
) -> None:
    shocks = {("e_u", "e_v"): Shock("norm", seed=11)}
    (entry,) = _entries(solved_test_model, shocks)
    block = _plan(solved_test_model, shocks).draw(2)

    width = len(entry.indices)
    z = philox_standard_normal(entry._native_seed_key, 0, 2, 0, T * width).reshape(
        T, width
    )
    expected = np.zeros((T, solved_test_model.compiled.n_exog))
    expected[:, entry.indices] = z @ entry.factor.T

    np.testing.assert_array_equal(block, expected)
    # The factor is the thing under test, so it must not be the identity.
    assert not np.allclose(entry.factor, np.eye(width))


def test_normal_draw_applies_the_location_shift(solved_test_model) -> None:
    shocks = {("e_u",): Shock("norm", seed=7, dist_kwargs={"loc": 2.5})}
    (entry,) = _entries(solved_test_model, shocks)
    block = _plan(solved_test_model, shocks).draw(0)

    z = philox_standard_normal(entry._native_seed_key, 0, 0, 0, T)
    np.testing.assert_array_equal(
        block[:, entry.indices[0]], z * entry.factor[0, 0] + 2.5
    )


def test_shock_scale_multiplies_the_whole_block(solved_test_model) -> None:
    shocks = {("e_u",): Shock("norm", seed=7)}
    plain = _plan(solved_test_model, shocks, shock_scale=1.0).draw(1)
    scaled = _plan(solved_test_model, shocks, shock_scale=2.5).draw(1)

    np.testing.assert_array_equal(scaled, 2.5 * plain)


def test_untargeted_columns_stay_zero(solved_test_model) -> None:
    shocks = {("e_u",): Shock("norm", seed=7)}
    (entry,) = _entries(solved_test_model, shocks)
    block = _plan(solved_test_model, shocks).draw(0)

    untargeted = [
        i
        for i in range(solved_test_model.compiled.n_exog)
        if i not in set(entry.indices)
    ]
    assert untargeted
    np.testing.assert_array_equal(block[:, untargeted], 0.0)


# --- draw integrity, per family ---------------------------------------------
#
# Each family's variate reaches the affine map standardized, so the kernel's job
# is to center and scale a raw stream by that family's own moments. Both sides
# call the same engine fill, which is not what is under test and has its own
# oracle in ``tests/ckernels/test_philox.py``; the standardization and the affine
# map are written twice, once in ``shocks.c`` from the shape parameters and once
# here from scipy's moments. A wrong constant fails on the difference.


def _address(entry, rep_idx: int, stream: int) -> tuple[int, int, int, int]:
    """The Philox key and stream words the kernel seeds this entry with."""
    return (entry._native_seed_key, entry.indices[0], rep_idx, stream)


def _reference(dist: str, kwargs: dict):
    """The family as scipy parameterizes it, for its mean and standard deviation."""
    match dist:
        case "uni":
            return uniform()
        case "exp":
            return expon()
        case "gamma":
            return gamma(kwargs["a"])
        case "beta":
            return beta(kwargs["a"], kwargs["b"])
    raise AssertionError(f"no scipy reference for {dist!r}")


def _raw_variate(dist: str, kwargs: dict, address, n: int):
    """The family's stream before standardization, at the kernel's own address."""
    match dist:
        case "uni":
            return philox_standard_uniform(*address, n)
        case "exp":
            return philox_standard_exponential(*address, n)
        case "gamma":
            return philox_standard_gamma(*address, n, kwargs["a"])
        case "beta":
            return philox_beta(*address, n, kwargs["a"], kwargs["b"])
    raise AssertionError(f"no engine fill for {dist!r}")


def _scattered(model, entry, v, shock_scale: float = 1.0):
    """``shock_scale * (loc + factor @ v)`` in the entry's columns, zero elsewhere."""
    block = np.zeros((T, model.compiled.n_exog))
    block[:, entry.indices] = shock_scale * (v @ entry.factor.T + entry.loc)
    return block


#: One spec per family and per sampler branch worth separating: gamma below and
#: above unit shape, beta symmetric and skewed. The lopsided beta is numerical
#: rather than economic: ``var = mean * (1 - mean) / (a + b + 1)`` loses the low
#: bits of ``1 - mean`` once ``a >> b``, and this is where that shows above ATOL.
SINGLE_STREAM_FAMILIES = [
    pytest.param("uni", {}, id="uni"),
    pytest.param("exp", {}, id="exp"),
    pytest.param("gamma", {"a": 0.3}, id="gamma-small"),
    pytest.param("gamma", {"a": 2.5}, id="gamma-large"),
    pytest.param("beta", {"a": 2.0, "b": 5.0}, id="beta"),
    pytest.param("beta", {"a": 0.3, "b": 4.0}, id="beta-skewed"),
    pytest.param("beta", {"a": 1e6, "b": 1.0}, id="beta-lopsided"),
]


@pytest.mark.parametrize("loc, shock_scale", [(0.0, 1.0), (1.5, 2.5)])
@pytest.mark.parametrize("dist, kwargs", SINGLE_STREAM_FAMILIES)
def test_a_standardized_family_is_its_centered_stream(
    solved_test_model, dist, kwargs, loc, shock_scale
) -> None:
    shocks = {("e_u",): Shock(dist, seed=7, dist_kwargs={**kwargs, "loc": loc})}
    (entry,) = _entries(solved_test_model, shocks)
    block = _plan(solved_test_model, shocks, shock_scale=shock_scale).draw(3)

    reference = _reference(dist, kwargs)
    raw = _raw_variate(dist, kwargs, _address(entry, 3, 0), T)
    v = (raw - reference.mean()) / reference.std()

    np.testing.assert_allclose(
        block,
        _scattered(solved_test_model, entry, v[:, None], shock_scale),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("targets, df", [(("e_u",), 5.0), (("e_u", "e_v"), 6.0)])
def test_student_t_scales_a_gaussian_by_its_own_chi_square(
    solved_test_model, targets, df
) -> None:
    shocks = {targets: Shock("t", seed=11, dist_kwargs={"df": df})}
    (entry,) = _entries(solved_test_model, shocks)
    block = _plan(solved_test_model, shocks).draw(3)

    # The Gaussian core rides stream 0 and the chi-square it is divided by rides
    # stream 1, one chi-square per period whatever the width.
    z = philox_standard_normal(*_address(entry, 3, 0), T * entry.width).reshape(
        T, entry.width
    )
    g = philox_chi2(*_address(entry, 3, 1), T, df)
    v = (z / np.sqrt(g / df)[:, None]) / t(df).std()

    np.testing.assert_allclose(
        block, _scattered(solved_test_model, entry, v), rtol=RTOL, atol=ATOL
    )


#: Dense factors, which no calibration produces: a model's is Cholesky and so
#: lower-triangular. The 2x2 is what ``_gaussian_factor`` returns for
#: [[1, 1], [1, 1]] through its eigh fallback; the 3x3 carries both halves of
#: every row at a width the test model's pair cannot express. Neither starts its
#: columns contiguously, and the 3x3 does not start at column zero.
DENSE_FACTORS = [
    pytest.param(np.array([[0.0, 1.0], [0.0, 1.0]]), (0, 2), 3, id="eigh-2x2"),
    pytest.param(
        np.array(
            [
                [0.50, 0.25, -0.125],
                [-0.75, 0.40, 0.20],
                [0.25, -0.50, 0.60],
            ]
        ),
        (1, 2, 4),
        5,
        id="dense-3x3",
    ),
]


@pytest.mark.parametrize("factor, indices, n_exog", DENSE_FACTORS)
def test_the_affine_map_sums_the_whole_factor_row(factor, indices, n_exog) -> None:
    """The ``j > i`` half of the row sum, which no calibration can reach.

    Building the entry rather than resolving one is what makes a dense factor
    reachable without a model whose shock covariance is degenerate.
    """
    width = len(indices)
    entry = ShockEntry(
        key=tuple(f"s{column}" for column in indices),
        indices=indices,
        family=ShockCode.NORMAL,
        loc=np.linspace(0.25, -0.5, width),
        factor=factor,
        base_seed=7,
    )
    block = native_shock_plan([entry], T, n_exog, 1.5).draw(2)

    z = philox_standard_normal(7, indices[0], 2, 0, T * width).reshape(T, width)
    expected = np.zeros((T, n_exog))
    expected[:, indices] = 1.5 * (z @ factor.T + entry.loc)
    np.testing.assert_allclose(block, expected, rtol=RTOL, atol=ATOL)

    # The row sum is the thing under test, so a kernel stopping at j <= i has to
    # give a different answer. A lower-triangular factor cannot state that.
    lower = np.tril(factor)
    truncated = np.zeros((T, n_exog))
    truncated[:, indices] = 1.5 * (z @ lower.T + entry.loc)
    assert not np.allclose(block, truncated)

    untargeted = [column for column in range(n_exog) if column not in set(indices)]
    np.testing.assert_array_equal(block[:, untargeted], 0.0)


def test_a_correlated_pair_draws_through_the_off_diagonal(solved_post82) -> None:
    """The same row sum on the resolved route, with a factor that is not diagonal.

    The test model declares no shock correlation, so its factor is diagonal and
    every cross term above is multiplied by zero. POST82 declares ``rho_gz``.
    """
    shocks = {("e_g", "e_z"): Shock("norm", seed=11)}
    (entry,) = _entries(solved_post82, shocks)
    assert not np.allclose(entry.factor, np.diag(np.diag(entry.factor)))

    block = _plan(solved_post82, shocks).draw(2)
    z = philox_standard_normal(*_address(entry, 2, 0), T * entry.width).reshape(
        T, entry.width
    )
    expected = np.zeros((T, solved_post82.compiled.n_exog))
    expected[:, entry.indices] = z @ entry.factor.T + entry.loc

    np.testing.assert_allclose(block, expected, rtol=RTOL, atol=ATOL)


def test_a_path_entry_is_scattered_verbatim(solved_test_model) -> None:
    values = np.arange(T, dtype=np.float64) * 0.5
    shocks = {("e_u",): values}
    (entry,) = _entries(solved_test_model, shocks)
    block = _plan(solved_test_model, shocks, shock_scale=2.5).draw(0)

    expected = np.zeros((T, solved_test_model.compiled.n_exog))
    expected[:, entry.indices[0]] = 2.5 * values
    np.testing.assert_array_equal(block, expected)


def test_a_multi_column_path_keeps_each_column_with_its_own_shock(
    solved_test_model,
) -> None:
    # Width 1 cannot separate a correct scatter from a transposed one.
    values = np.arange(T * 2, dtype=np.float64).reshape(T, 2)
    shocks = [ShockPath(values, "e_u", "e_v")]
    (entry,) = _entries(solved_test_model, shocks)
    block = _plan(solved_test_model, shocks).draw(0)

    expected = np.zeros((T, solved_test_model.compiled.n_exog))
    expected[:, entry.indices] = values
    np.testing.assert_array_equal(block, expected)


# --- addressing -------------------------------------------------------------


def test_the_kernel_addresses_a_draw_by_the_declared_seed_and_column(
    solved_test_model,
) -> None:
    """The Philox address is literal here, not read back off the entry.

    Every reconstruction above takes the key from ``_native_seed_key``, so a
    changed derivation moves both sides together and passes. This one does not.
    """
    assert solved_test_model.compiled.shock_idx["e_v"] == 1
    shocks = {("e_v",): Shock("norm", seed=7)}
    (entry,) = _entries(solved_test_model, shocks)
    block = _plan(solved_test_model, shocks).draw(3)

    # seed 7, e_v's canonical column, replication 3, the family's first stream.
    z = philox_standard_normal(7, 1, 3, 0, T)
    expected = np.zeros((T, solved_test_model.compiled.n_exog))
    expected[:, 1] = z * entry.factor[0, 0]

    np.testing.assert_array_equal(block, expected)


def test_a_seeded_spec_replays_across_plans(solved_test_model) -> None:
    shocks = {("e_u",): Shock("norm", seed=1), ("e_v",): Shock("uni", seed=2)}
    first = _plan(solved_test_model, shocks)
    second = _plan(solved_test_model, shocks)

    for rep_idx in (0, 1, 97):
        np.testing.assert_array_equal(first.draw(rep_idx), second.draw(rep_idx))


def test_replications_do_not_share_a_stream(solved_test_model) -> None:
    plan = _plan(solved_test_model, {("e_u", "e_v"): Shock("norm", seed=1)})
    blocks = [plan.draw(rep_idx) for rep_idx in range(4)]

    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            assert not np.array_equal(blocks[i], blocks[j])


def test_entries_sharing_a_seed_stay_independent(solved_test_model) -> None:
    shocks = {("e_u",): Shock("norm", seed=5), ("e_v",): Shock("norm", seed=5)}
    entries = _entries(solved_test_model, shocks)
    block = _plan(solved_test_model, shocks).draw(0)

    assert entries[0]._native_seed_key == entries[1]._native_seed_key
    left = block[:, entries[0].indices[0]]
    right = block[:, entries[1].indices[0]]
    assert not np.array_equal(left, right)


def test_an_unseeded_spec_redraws_each_run(solved_test_model) -> None:
    shocks = {("e_u",): Shock("norm", seed=None)}
    assert not np.array_equal(
        _plan(solved_test_model, shocks).draw(0),
        _plan(solved_test_model, shocks).draw(0),
    )


def test_negative_replication_index_is_rejected(solved_test_model) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _plan(solved_test_model, {("e_u",): Shock("norm", seed=0)}).draw(-1)


# --- the caller-owned buffer -------------------------------------------------


def test_fill_writes_in_place_and_leaves_untargeted_columns_alone(
    solved_test_model,
) -> None:
    """``fill`` writes the spec's columns and nothing else.

    The caller owns zeroing, which is what lets a simulation step hand the kernel
    a slice of its own arena instead of a fresh block. ``draw`` only ever passes
    zeros, so nothing else can state this.
    """
    shocks = {("e_u",): Shock("norm", seed=7)}
    (entry,) = _entries(solved_test_model, shocks)
    plan = _plan(solved_test_model, shocks)

    out = np.full((T, solved_test_model.compiled.n_exog), 9.5)
    plan.fill(out, 3)

    untargeted = [
        column
        for column in range(solved_test_model.compiled.n_exog)
        if column not in set(entry.indices)
    ]
    assert untargeted
    np.testing.assert_array_equal(out[:, untargeted], 9.5)
    np.testing.assert_array_equal(
        out[:, entry.indices[0]], plan.draw(3)[:, entry.indices[0]]
    )


@pytest.mark.parametrize("d_rows, d_cols", [(1, 0), (0, 1), (-1, 2)])
def test_fill_rejects_a_block_of_the_wrong_shape(
    solved_test_model, d_rows, d_cols
) -> None:
    n_exog = solved_test_model.compiled.n_exog
    plan = _plan(solved_test_model, {("e_u",): Shock("norm", seed=7)})

    with pytest.raises(ValueError, match=rf"must have shape \({T}, {n_exog}\)"):
        plan.fill(np.zeros((T + d_rows, n_exog + d_cols)), 0)


@pytest.mark.parametrize(
    "build, message",
    [
        pytest.param(
            lambda rows, cols: np.zeros((rows, cols), dtype=np.float32),
            "Buffer dtype mismatch",
            id="float32",
        ),
        pytest.param(
            lambda rows, cols: np.asfortranarray(np.zeros((rows, cols))),
            "not C-contiguous",
            id="f-order",
        ),
        pytest.param(
            lambda rows, cols: np.zeros((rows, cols * 2))[:, ::2],
            "not C-contiguous",
            id="strided",
        ),
        pytest.param(
            lambda rows, cols: np.zeros(rows * cols),
            "wrong number of dimensions",
            id="flat",
        ),
    ],
)
def test_fill_rejects_a_block_it_cannot_write_through(
    solved_test_model, build, message
) -> None:
    # The buffer protocol raises these, not the shape check above, and they are
    # what stands between a caller's wrong layout and a corrupt write.
    n_exog = solved_test_model.compiled.n_exog
    plan = _plan(solved_test_model, {("e_u",): Shock("norm", seed=7)})

    with pytest.raises(ValueError, match=message):
        plan.fill(build(T, n_exog), 0)


@pytest.mark.parametrize("n_periods", [T + 3, T - 3])
def test_a_path_whose_length_is_not_the_horizon_is_rejected(
    solved_test_model, n_periods
) -> None:
    # Resolution accepts any length; the horizon is only known where the plan is
    # built, which is the one place that can compare them.
    with pytest.raises(ValueError, match=f"Path period length {n_periods}"):
        _plan(solved_test_model, {("e_u",): np.zeros(n_periods)})


# --- entry invariance, on both routes (#507) --------------------------------
#
# An entry's stream must be a function of the entry alone. The two routes key
# differently -- Philox on ``(seed, columns[0], rep_idx)`` in C, a mixed
# ``SeedSequence`` on the same triple in Python -- so they never produce equal
# numbers, but they owe the same invariances. Parameterizing on the family is
# what selects the route: ``norm`` lowers to the kernel, ``t`` does not, and
# ``_native_draw`` asserts the lowering actually happened rather than trusting it.


def _draw(solved_test_model, spec, rep_idx):
    plan = _plan(solved_test_model, spec)
    assert plan is not None, "spec was expected to lower to the native draw"
    return plan.draw(rep_idx)


ROUTES = [
    pytest.param("t", _draw, id="t"),
    pytest.param("norm", _draw, id="norm"),
]


def _entry(family, seed, target):
    kwargs = {"df": 5} if family == "t" else {}
    return Shock(family, seed=seed, dist_kwargs=kwargs).independent(target)[0]


@pytest.mark.parametrize("family, draw", ROUTES)
def test_entry_draw_does_not_depend_on_its_position_in_the_spec(
    solved_test_model, family, draw
) -> None:
    """Reordering a spec must move nothing.

    This is the native half of #507: the Philox key carried the entry's position
    in the spec list, so every entry that moved drew differently even though the
    columns it writes are canonical. The Python route always held this.
    """
    spec = [_entry(family, 11, "e_u"), _entry(family, 12, "e_v")]

    np.testing.assert_array_equal(
        draw(solved_test_model, spec, 3),
        draw(solved_test_model, list(reversed(spec)), 3),
    )


@pytest.mark.parametrize("family, draw", ROUTES)
def test_entry_draw_does_not_depend_on_the_rest_of_the_spec(
    solved_test_model, family, draw
) -> None:
    """Adding a second seeded entry leaves the first untouched.

    The Python half of #507: the per-replication shift was the count of seeded
    entries, so adding one moved every other entry at every replication past
    zero. Native keyed per entry and already held this.
    """
    alone = [_entry(family, 11, "e_u")]
    joined = alone + [_entry(family, 12, "e_v")]
    col = solved_test_model.compiled.shock_idx["e_u"]

    np.testing.assert_array_equal(
        draw(solved_test_model, alone, 3)[:, col],
        draw(solved_test_model, joined, 3)[:, col],
    )


@pytest.mark.parametrize("family, draw", ROUTES)
def test_identically_seeded_copies_are_not_degenerate(
    solved_test_model, family, draw
) -> None:
    """Copies sharing one seed still draw independently.

    ``offset_seeds=False`` hands every copy the template's seed. On the Python
    route that used to give one standardized variate scaled per column, a
    rank-deficient block whatever ``shock_corr`` declared. Rank is scale-free,
    so this reads the same on either route.
    """
    kwargs = {"df": 5} if family == "t" else {}
    spec = Shock(family, seed=5, dist_kwargs=kwargs).independent(
        "e_u", "e_v", offset_seeds=False
    )
    assert [shock.seed for shock in spec] == [5, 5]

    assert np.linalg.matrix_rank(draw(solved_test_model, spec, 0)) == len(spec)
