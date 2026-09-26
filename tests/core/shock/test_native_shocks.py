"""Native shock sampling, eligibility, and stream addressing."""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE import Shock
from SymbolicDSGE._ckernels.rng import philox_standard_normal
from SymbolicDSGE.core.shock.plan import get_native_shock_plan, is_native_spec_eligible
from SymbolicDSGE.core.shock.spec import _normalized_spec, resolve_shock_plan

T = 16


def _plan(solved_test_model, shocks, shock_scale=1.0):
    resolved = resolve_shock_plan(solved_test_model.compiled, shocks, T)
    return get_native_shock_plan(resolved, T, resolved.n_exog, shock_scale)


def _entries(solved_test_model, shocks):
    return resolve_shock_plan(solved_test_model.compiled, shocks, T).entries


@pytest.mark.parametrize(
    "dist, targets", [("norm", ("e_u",)), ("norm", ("e_u", "e_v")), ("uni", ("e_u",))]
)
def test_native_spec_accepts_supported_families(dist, targets):
    assert is_native_spec_eligible(_normalized_spec({targets: Shock(dist, seed=0)}))


@pytest.mark.parametrize(
    "shocks",
    [
        {},
        {("e_u",): Shock("t", seed=0, dist_kwargs={"df": 5})},
        {("e_u", "e_v"): Shock("uni", seed=0)},
        {("e_u",): np.zeros(T)},
        # One ineligible entry sends the whole specification back.
        {
            ("e_u",): Shock("norm", seed=0),
            ("e_v",): Shock("t", seed=1, dist_kwargs={"df": 5}),
        },
    ],
)
def test_native_families_rejects_unported_specs(shocks) -> None:
    spec = _normalized_spec(shocks)
    assert not is_native_spec_eligible(spec)


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


# --- addressing -------------------------------------------------------------


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


# --- entry invariance, on both routes (#507) --------------------------------
#
# An entry's stream must be a function of the entry alone. The two routes key
# differently -- Philox on ``(seed, columns[0], rep_idx)`` in C, a mixed
# ``SeedSequence`` on the same triple in Python -- so they never produce equal
# numbers, but they owe the same invariances. Parameterizing on the family is
# what selects the route: ``norm`` lowers to the kernel, ``t`` does not, and
# ``_native_draw`` asserts the lowering actually happened rather than trusting it.


def _python_draw(solved_test_model, spec, rep_idx):
    return resolve_shock_plan(solved_test_model.compiled, spec, T).matrix(
        T, 1.0, rep_idx
    )


def _native_draw(solved_test_model, spec, rep_idx):
    plan = _plan(solved_test_model, spec)
    assert plan is not None, "spec was expected to lower to the native draw"
    return plan.draw(rep_idx)


ROUTES = [
    pytest.param("t", _python_draw, id="python"),
    pytest.param("norm", _native_draw, id="native"),
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
