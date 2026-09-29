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

from SymbolicDSGE._ckernels.core._shocks import ShockCode
from SymbolicDSGE.core.shock.generators import Shock
from SymbolicDSGE.core.shock.plan import ShockEntry, get_native_shock_plan
import SymbolicDSGE.core.shock.spec as shocks_mod
from SymbolicDSGE.core.shock.spec import (
    resolve_shock_plan,
    simulation_shock_matrix,
)

T = 12


#: One seeded spec per drawn family and arity, so the two replication properties
#: below are checked against every family the kernel draws. A family that failed
#: to lower would leave ``get_native_shock_plan`` returning None and fail here,
#: which is what pins the dispatch for the families nothing else reaches.
FAMILY_SPECS = [
    {("e_u",): Shock(dist="norm", seed=3)},
    {("e_u",): Shock(dist="t", seed=5, dist_kwargs={"df": 4})},
    {("e_u",): Shock(dist="uni", seed=7)},
    {("e_u",): Shock(dist="exp", seed=17)},
    {("e_u",): Shock(dist="gamma", seed=19, dist_kwargs={"a": 2.5})},
    {("e_u",): Shock(dist="beta", seed=23, dist_kwargs={"a": 2.0, "b": 5.0})},
    {("e_u", "e_v"): Shock(dist="norm", seed=11)},
    {("e_u", "e_v"): Shock(dist="t", seed=13, dist_kwargs={"df": 6})},
]


@pytest.mark.parametrize("spec", FAMILY_SPECS)
@pytest.mark.parametrize("rep_idx", [0, 1, 37])
def test_plan_draw_is_reproducible_per_replication(solved_test, spec, rep_idx):
    """Every family draws one fixed path per replication index."""
    plan = resolve_shock_plan(solved_test.compiled, spec, T)
    native = get_native_shock_plan(plan, T, 2.5)
    np.testing.assert_array_equal(
        native.draw(rep_idx),
        native.draw(rep_idx),
    )


@pytest.mark.parametrize("spec", FAMILY_SPECS)
@pytest.mark.parametrize("rep_idx", [0, 1, 37])
def test_plan_draw_differs_across_replications(solved_test, spec, rep_idx):
    """And a different index is a different path, for every family."""
    plan = resolve_shock_plan(solved_test.compiled, spec, T)
    native = get_native_shock_plan(plan, T, 2.5)
    assert not np.array_equal(
        native.draw(rep_idx),
        native.draw(rep_idx + 1),
    )


def test_plan_reseeds_independently_across_draws(solved_test):
    spec = {("e_u", "e_v"): Shock(dist="norm", seed=11)}
    plan = resolve_shock_plan(solved_test.compiled, spec, T)
    native = get_native_shock_plan(plan, T, 1.0)

    first = native.draw(0)
    second = native.draw(1)
    again = native.draw(0)

    # Redrawing is a pure function of the index: same index, same path.
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, second)


def test_unseeded_spec_reproduces_with_the_same_plan(solved_test):
    plan = resolve_shock_plan(
        solved_test.compiled, {("e_u",): Shock(dist="norm", seed=None)}, T
    )
    native = get_native_shock_plan(plan, T, 1.0)

    first = native.draw(0)
    second = native.draw(0)

    assert np.array_equal(first, second)


def test_plan_resolution_is_reused_not_recomputed(solved_test, monkeypatch):
    spec = {("e_u", "e_v"): Shock(dist="norm", seed=11)}

    calls = {"n": 0}
    original = shocks_mod.make_Q

    def counting_make_Q(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(shocks_mod, "make_Q", counting_make_Q)

    plan = resolve_shock_plan(solved_test.compiled, spec, T)
    native = get_native_shock_plan(plan, T, 1.0)
    resolved = calls["n"]
    assert resolved > 0

    for offset in range(25):
        native.draw(offset)

    # The covariance is spec-level, so redrawing must not rebuild it.
    assert calls["n"] == resolved


def test_passthrough_entries_ignore_the_replication_index(solved_test):
    values = np.arange(T, dtype=np.float64)
    plan = resolve_shock_plan(solved_test.compiled, {("e_u",): values}, T)
    native = get_native_shock_plan(plan, T, 1.0)
    np.testing.assert_array_equal(native.draw(0), native.draw(9))


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

    forward = resolve_shock_plan(solved_post82.compiled, spec, T)
    fnative = get_native_shock_plan(forward, T, 1.0).draw(3)

    backward = resolve_shock_plan(solved_post82.compiled, list(reversed(spec)), T)
    bnative = get_native_shock_plan(backward, T, 1.0).draw(3)

    # Columns are addressed by shock, so reordering the spec must move nothing.
    # Unlike the three below, this one also held under the additive scheme: the
    # Python route was always position-invariant and it was the native keying
    # that was not. Pinned here so the derivation cannot regress into using
    # anything positional.
    np.testing.assert_array_equal(fnative, bnative)


def test_entry_draw_does_not_depend_on_the_rest_of_the_spec(solved_post82):
    col = _cols(solved_post82)
    pair = [_t_shock(11, "e_g"), _t_shock(12, "e_z")]
    trio = pair + [_t_shock(13, "e_r")]

    without = resolve_shock_plan(solved_post82.compiled, pair, T)
    wnative = get_native_shock_plan(without, T, 1.0).draw(3)
    with_extra = resolve_shock_plan(solved_post82.compiled, trio, T)
    wenative = get_native_shock_plan(with_extra, T, 1.0).draw(3)
    # Adding a third seeded entry leaves the first two untouched. Under the old
    # per-replication stride it shifted both at every replication past zero.
    for name in ("e_g", "e_z"):
        np.testing.assert_array_equal(wnative[:, col[name]], wenative[:, col[name]])


def test_entries_with_congruent_seeds_do_not_share_a_stream(solved_post82):
    col = _cols(solved_post82)
    # Seeds 0 and 2 over two seeded entries: the old offset was
    # ``base_seed + rep_idx * seeded_count``, so e_g at replication 1 and e_r at
    # replication 0 both resolved to seed 2. Equal standard deviations made the
    # two columns identical, not merely correlated.
    spec = [_t_shock(0, "e_g"), _t_shock(2, "e_r")]
    plan = resolve_shock_plan(solved_post82.compiled, spec, T)
    native = get_native_shock_plan(plan, T, 1.0)
    e_g_at_1 = native.draw(1)[:, col["e_g"]]
    e_r_at_0 = native.draw(0)[:, col["e_r"]]

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
    native = get_native_shock_plan(plan, T, 1.0)
    seen: dict[bytes, tuple[str, int]] = {}
    for rep_idx in range(6):
        block = native.draw(rep_idx)
        for name in ("e_g", "e_z", "e_r"):
            drawn = block[:, col[name]].tobytes()
            assert (
                drawn not in seen
            ), f"{name} at replication {rep_idx} repeats {seen.get(drawn)}"
            seen[drawn] = (name, rep_idx)


# ---- the seed derivation (#507) ------------------------------------------
#
# Both routes key off the triple (declared seed, canonical column, replication),
# and every draw test reads the result back off the entry, so only a literal can
# fail when the derivation moves. ``_seed`` mixes through numpy's
# ``SeedSequence``, which puts these values inside the drift surface of #527.


def _seed_entry(base_seed, column):
    return ShockEntry(
        key=("x",),
        indices=(column,),
        family=ShockCode.NORMAL,
        loc=np.zeros(1),
        factor=np.ones((1, 1)),
        base_seed=base_seed,
    )


@pytest.mark.parametrize(
    "base_seed, key",
    [
        (0, 0),
        (7, 7),
        (2**63, 2**63),
        (2**64 - 1, 2**64 - 1),
        # A declared seed is an unbounded Python int and the kernel key is a u64,
        # so both ends fold in: a negative seed is its two's complement, and one
        # past the range is congruent to a seed inside it.
        (-1, 2**64 - 1),
        (2**64, 0),
        (2**64 + 5, 5),
    ],
)
def test_native_seed_key_is_the_declared_seed_masked_to_64_bits(base_seed, key):
    assert _seed_entry(base_seed, 0)._native_seed_key == key


@pytest.mark.parametrize(
    "base_seed, column, rep_idx, seed",
    [
        (7, 0, 0, 13432090166537452992),
        (7, 0, 1, 15529291740490724314),
        (7, 1, 0, 23751027488930731),
        (0, 0, 0, 2635072618980576772),
        (5, 3, 41, 10328991043790977369),
    ],
)
def test_replication_seed_is_the_mixed_triple(base_seed, column, rep_idx, seed):
    assert _seed_entry(base_seed, column)._seed(rep_idx) == seed


def test_an_unseeded_entry_has_no_replication_seed_and_a_fresh_key():
    # The key is drawn per access, which is why a plan reads it once at build and
    # a draw is reproducible within that plan but not across two of them.
    entry = _seed_entry(None, 0)
    assert entry._seed(0) is None
    assert len({entry._native_seed_key for _ in range(4)}) == 4
