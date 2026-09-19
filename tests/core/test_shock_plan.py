# type: ignore
"""A resolved ShockPlan redraws per replication without re-resolving.

A plan resolves the calibration, the covariance and its factor once, then varies
only the replication index per draw. Each index must select one reproducible
draw, and distinct indices must select distinct ones, since that is what lets a
single plan serve every replication of a run.

The seed an entry draws under is derived from its declared seed, its canonical
column, and the replication index (#507), so it is not the declared seed itself
and a spec authored with a shifted seed is a different spec, not a later draw.

That derivation carries a second property, covered at the end of this file: an
entry's stream is a function of the entry alone. Where it sits in the spec, what
else the spec contains, and whether another entry happens to share or neighbour
its seed must none of them move it. The additive scheme this replaced held the
first and broke the other two.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.core.shock.generators import Shock
import SymbolicDSGE.core.shock.spec as shocks_mod
from SymbolicDSGE.core.shock.spec import (
    resolve_shock_plan,
    simulation_shock_matrix,
)

T = 12


#: One seeded spec per drawn family and arity, so the two replication properties
#: below are checked against every route :meth:`Shock.draw_fn` resolves onto.
FAMILY_SPECS = [
    {("e_u",): Shock(dist="norm", seed=3)},
    {("e_u",): Shock(dist="t", seed=5, dist_kwargs={"df": 4})},
    {("e_u",): Shock(dist="uni", seed=7)},
    {("e_u", "e_v"): Shock(dist="norm", seed=11)},
    {("e_u", "e_v"): Shock(dist="t", seed=13, dist_kwargs={"df": 6})},
]


@pytest.mark.parametrize("spec", FAMILY_SPECS)
@pytest.mark.parametrize("rep_idx", [0, 1, 37])
def test_plan_draw_is_reproducible_per_replication(solved_test, spec, rep_idx):
    """Every family draws one fixed path per replication index."""
    plan = resolve_shock_plan(solved_test.compiled, spec, T)

    np.testing.assert_array_equal(
        plan.matrix(T, 2.5, rep_idx),
        plan.matrix(T, 2.5, rep_idx),
    )


@pytest.mark.parametrize("spec", FAMILY_SPECS)
@pytest.mark.parametrize("rep_idx", [0, 1, 37])
def test_plan_draw_differs_across_replications(solved_test, spec, rep_idx):
    """And a different index is a different path, for every family."""
    plan = resolve_shock_plan(solved_test.compiled, spec, T)

    assert not np.array_equal(
        plan.matrix(T, 2.5, rep_idx),
        plan.matrix(T, 2.5, rep_idx + 1),
    )


def test_plan_reseeds_independently_across_draws(solved_test):
    spec = {("e_u", "e_v"): Shock(dist="norm", seed=11)}
    plan = resolve_shock_plan(solved_test.compiled, spec, T)

    first = plan.matrix(T, 1.0, 0)
    second = plan.matrix(T, 1.0, 1)
    again = plan.matrix(T, 1.0, 0)

    # Redrawing is a pure function of the index: same index, same path.
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, second)


def test_unseeded_spec_redraws_each_time(solved_test):
    plan = resolve_shock_plan(
        solved_test.compiled, {("e_u",): Shock(dist="norm", seed=None)}, T
    )

    first = plan.matrix(T, 1.0, 0)
    second = plan.matrix(T, 1.0, 0)

    # A seedless spec draws fresh entropy per call; the offset cannot pin it.
    assert not np.array_equal(first, second)


def test_plan_resolution_is_reused_not_recomputed(solved_test, monkeypatch):
    spec = {("e_u", "e_v"): Shock(dist="norm", seed=11)}

    calls = {"n": 0}
    original = shocks_mod.make_Q

    def counting_make_Q(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(shocks_mod, "make_Q", counting_make_Q)

    plan = resolve_shock_plan(solved_test.compiled, spec, T)
    resolved = calls["n"]
    assert resolved > 0

    for offset in range(25):
        plan.matrix(T, 1.0, offset)

    # The covariance is spec-level, so redrawing must not rebuild it.
    assert calls["n"] == resolved


def test_passthrough_entries_ignore_the_replication_index(solved_test):
    values = np.arange(T, dtype=np.float64)
    plan = resolve_shock_plan(solved_test.compiled, {("e_u",): values}, T)

    np.testing.assert_array_equal(plan.matrix(T, 1.0, 0), plan.matrix(T, 1.0, 9))


def test_live_shock_requires_a_horizon(solved_test):
    with pytest.raises(ValueError, match="needs a horizon T"):
        resolve_shock_plan(solved_test.compiled, {("e_u",): Shock(dist="norm", seed=1)})


# ---- entry invariance (#507) ---------------------------------------------
#
# POST82 rather than the two-shock test model: these need three shocks to say
# anything about "the rest of the spec". ``sig_g`` and ``sig_r`` are both 0.18,
# so two entries that shared a stream drew byte-identical columns rather than
# proportional ones, which is what lets the assertions below be exact.


def _cols(model):
    return model.compiled.shock_idx


def _t_shock(seed, target):
    """A Student-t entry, which keeps the spec on the Python draw route."""
    return Shock(dist="t", seed=seed, dist_kwargs={"df": 5}).independent(target)[0]


def test_entry_draw_does_not_depend_on_its_position_in_the_spec(solved_post82):
    spec = [_t_shock(11, "e_g"), _t_shock(12, "e_z"), _t_shock(13, "e_r")]

    forward = resolve_shock_plan(solved_post82.compiled, spec, T).matrix(T, 1.0, 3)
    backward = resolve_shock_plan(
        solved_post82.compiled, list(reversed(spec)), T
    ).matrix(T, 1.0, 3)

    # Columns are addressed by shock, so reordering the spec must move nothing.
    # Unlike the three below, this one also held under the additive scheme: the
    # Python route was always position-invariant and it was the native keying
    # that was not. Pinned here so the derivation cannot regress into using
    # anything positional.
    np.testing.assert_array_equal(forward, backward)


def test_entry_draw_does_not_depend_on_the_rest_of_the_spec(solved_post82):
    col = _cols(solved_post82)
    pair = [_t_shock(11, "e_g"), _t_shock(12, "e_z")]
    trio = pair + [_t_shock(13, "e_r")]

    without = resolve_shock_plan(solved_post82.compiled, pair, T).matrix(T, 1.0, 3)
    with_extra = resolve_shock_plan(solved_post82.compiled, trio, T).matrix(T, 1.0, 3)

    # Adding a third seeded entry leaves the first two untouched. Under the old
    # per-replication stride it shifted both at every replication past zero.
    for name in ("e_g", "e_z"):
        np.testing.assert_array_equal(without[:, col[name]], with_extra[:, col[name]])


def test_entries_with_congruent_seeds_do_not_share_a_stream(solved_post82):
    col = _cols(solved_post82)
    # Seeds 0 and 2 over two seeded entries: the old offset was
    # ``base_seed + rep_idx * seeded_count``, so e_g at replication 1 and e_r at
    # replication 0 both resolved to seed 2. Equal standard deviations made the
    # two columns identical, not merely correlated.
    spec = [_t_shock(0, "e_g"), _t_shock(2, "e_r")]
    plan = resolve_shock_plan(solved_post82.compiled, spec, T)

    e_g_at_1 = plan.matrix(T, 1.0, 1)[:, col["e_g"]]
    e_r_at_0 = plan.matrix(T, 1.0, 0)[:, col["e_r"]]

    assert not np.array_equal(e_g_at_1, e_r_at_0)


def test_identically_seeded_independent_copies_are_not_degenerate(solved_post82):
    # ``offset_seeds=False`` hands every copy the template's own seed. That used
    # to give one standardized variate scaled three ways: a rank-1 shock block,
    # perfectly correlated across all pairs, whatever ``shock_corr`` declared.
    spec = Shock(dist="norm", seed=5).independent(
        "e_g", "e_z", "e_r", offset_seeds=False
    )
    assert [shock.seed for shock in spec] == [5, 5, 5]

    drawn = simulation_shock_matrix(solved_post82.compiled, T, spec)

    assert np.linalg.matrix_rank(drawn) == len(spec)


def test_every_entry_and_replication_pair_draws_its_own_stream(solved_post82):
    col = _cols(solved_post82)
    # Seeds spaced by the seeded-entry count, which is the pattern the old
    # stride collided on: every pairwise difference is a multiple of three.
    spec = [_t_shock(0, "e_g"), _t_shock(3, "e_z"), _t_shock(6, "e_r")]
    plan = resolve_shock_plan(solved_post82.compiled, spec, T)

    seen: dict[bytes, tuple[str, int]] = {}
    for rep_idx in range(6):
        block = plan.matrix(T, 1.0, rep_idx)
        for name in ("e_g", "e_z", "e_r"):
            drawn = block[:, col[name]].tobytes()
            assert (
                drawn not in seen
            ), f"{name} at replication {rep_idx} repeats {seen.get(drawn)}"
            seen[drawn] = (name, rep_idx)
