# type: ignore
"""Reseeding a resolved ShockPlan equals cloning the spec with a shifted seed.

A plan resolves the calibration, the covariance and its factor once, then varies
only the seed per draw. Shifting a plan's seed by ``k`` must therefore produce
exactly what a spec authored with ``seed + k`` produces on its first draw, since
that equality is what lets one plan serve every replication of a run.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.core.shock.generators import Shock
import SymbolicDSGE.core.shock.spec as shocks_mod
from SymbolicDSGE.core.shock.spec import resolve_shock_plan

T = 12


def _cloned_matrix(model, shocks, T, shock_scale, seed_offset):
    """Author the same spec with the shifted seed and draw it at offset zero."""
    cloned = {}
    for name, shock in shocks.items():
        if isinstance(shock, Shock):
            seed = None if shock.seed is None else int(shock.seed) + seed_offset
            cloned[name] = Shock(
                dist=shock.dist,
                seed=seed,
                dist_kwargs=shock.dist_kwargs.copy(),
            )
        else:
            cloned[name] = shock

    return resolve_shock_plan(model.compiled, cloned, T).matrix(T, shock_scale, 0)


@pytest.mark.parametrize(
    "spec",
    [
        {("e_u",): Shock(dist="norm", seed=3)},
        {("e_u",): Shock(dist="t", seed=5, dist_kwargs={"df": 4})},
        {("e_u",): Shock(dist="uni", seed=7)},
        {("e_u", "e_v"): Shock(dist="norm", seed=11)},
        {("e_u", "e_v"): Shock(dist="t", seed=13, dist_kwargs={"df": 6})},
    ],
)
@pytest.mark.parametrize("seed_offset", [0, 1, 37])
def test_plan_draw_matches_clone_per_draw(solved_test, spec, seed_offset):
    plan = resolve_shock_plan(solved_test.compiled, spec, T)

    got = plan.matrix(T, 2.5, seed_offset)
    want = _cloned_matrix(solved_test, spec, T, 2.5, seed_offset)

    np.testing.assert_array_equal(got, want)


def test_plan_reseeds_independently_across_draws(solved_test):
    spec = {("e_u", "e_v"): Shock(dist="norm", seed=11)}
    plan = resolve_shock_plan(solved_test.compiled, spec, T)

    first = plan.matrix(T, 1.0, 0)
    second = plan.matrix(T, 1.0, 1)
    again = plan.matrix(T, 1.0, 0)

    # Redrawing is a pure function of the offset: same offset, same path.
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


def test_passthrough_entries_ignore_the_seed_offset(solved_test):
    values = np.arange(T, dtype=np.float64)
    plan = resolve_shock_plan(solved_test.compiled, {("e_u",): values}, T)

    np.testing.assert_array_equal(plan.matrix(T, 1.0, 0), plan.matrix(T, 1.0, 9))


def test_seeded_count_counts_seeded_entries(solved_test):
    spec = {
        ("e_u", "e_v"): Shock(dist="norm", seed=0),
    }
    plan = resolve_shock_plan(solved_test.compiled, spec, T)

    assert plan.seeded_count == 1


def test_unseeded_specs_do_not_count(solved_test):
    plan = resolve_shock_plan(
        solved_test.compiled, {("e_u",): Shock(dist="norm", seed=None)}, T
    )

    assert plan.seeded_count == 0


def test_live_shock_requires_a_horizon(solved_test):
    with pytest.raises(ValueError, match="needs a horizon T"):
        resolve_shock_plan(solved_test.compiled, {("e_u",): Shock(dist="norm", seed=1)})
