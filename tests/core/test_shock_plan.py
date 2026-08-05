# type: ignore
"""Parity between a resolved ShockPlan and the clone-per-draw path it replaces.

The Monte Carlo lowering used to rebuild a ``Shock``, its draw closure, and the
whole calibration resolution once per replication. A plan resolves all of that
once and reseeds per draw, so every path it produces must be bit-identical to
what the old route produced for the same seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.core.shock_generators import Shock

T = 12


def _legacy_matrix(model, shocks, T, shock_scale, seed_offset):
    """The pre-plan route: clone each Shock with its offset seed, then unpack."""
    materialized = {}
    for name, shock in shocks.items():
        if isinstance(shock, Shock):
            seed = None if shock.seed is None else int(shock.seed) + seed_offset
            materialized[name] = Shock(
                dist=shock.dist,
                multivar=shock.multivar,
                seed=seed,
                dist_args=shock.dist_args,
                dist_kwargs=shock.dist_kwargs.copy(),
            ).shock_generator(T)
        else:
            materialized[name] = shock

    out = np.zeros((T, model.compiled.n_exog), dtype=np.float64)
    for idx, values in model._shock_unpack(materialized):
        out[:, idx] = shock_scale * values
    return out


@pytest.mark.parametrize(
    "spec",
    [
        {"e_u": Shock(dist="norm", seed=3)},
        {"e_u": Shock(dist="t", seed=5, dist_kwargs={"df": 4})},
        {"e_u": Shock(dist="uni", seed=7)},
        {"e_u,e_v": Shock(dist="norm", multivar=True, seed=11)},
        {"e_u,e_v": Shock(dist="t", multivar=True, seed=13, dist_kwargs={"df": 6})},
    ],
)
@pytest.mark.parametrize("seed_offset", [0, 1, 37])
def test_plan_draw_matches_clone_per_draw(solved_test, spec, seed_offset):
    plan = solved_test._resolve_shock_plan(spec, T)

    got = plan.matrix(T, 2.5, seed_offset)
    want = _legacy_matrix(solved_test, spec, T, 2.5, seed_offset)

    np.testing.assert_array_equal(got, want)


def test_plan_reseeds_independently_across_draws(solved_test):
    spec = {"e_u,e_v": Shock(dist="norm", multivar=True, seed=11)}
    plan = solved_test._resolve_shock_plan(spec, T)

    first = plan.matrix(T, 1.0, 0)
    second = plan.matrix(T, 1.0, 1)
    again = plan.matrix(T, 1.0, 0)

    # Redrawing is a pure function of the offset: same offset, same path.
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, second)


def test_unseeded_spec_redraws_each_time(solved_test):
    plan = solved_test._resolve_shock_plan({"e_u": Shock(dist="norm", seed=None)}, T)

    first = plan.matrix(T, 1.0, 0)
    second = plan.matrix(T, 1.0, 0)

    # A seedless spec draws fresh entropy per call; the offset cannot pin it.
    assert not np.array_equal(first, second)


def test_plan_factor_matches_unfactored_covariance(solved_test):
    spec = {"e_u,e_v": Shock(dist="norm", multivar=True, seed=11)}
    plan = solved_test._resolve_shock_plan(spec, T)
    entry = plan.entries[0]

    assert entry.factor is not None
    np.testing.assert_allclose(entry.factor @ entry.factor.T, entry.scale, atol=1e-12)

    with_factor = entry.draw(entry.scale, 11, entry.factor)
    without_factor = entry.draw(entry.scale, 11, None)
    np.testing.assert_array_equal(with_factor, without_factor)


def test_plan_resolution_is_reused_not_recomputed(solved_test, monkeypatch):
    spec = {"e_u,e_v": Shock(dist="norm", multivar=True, seed=11)}

    calls = {"n": 0}
    original = type(solved_test)._get_rho

    def counting_get_rho(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(solved_test), "_get_rho", counting_get_rho)

    plan = solved_test._resolve_shock_plan(spec, T)
    resolved = calls["n"]
    assert resolved > 0

    for offset in range(25):
        plan.matrix(T, 1.0, offset)

    # Correlations are spec-level, so redrawing must not touch them again.
    assert calls["n"] == resolved


def test_passthrough_entries_ignore_the_seed_offset(solved_test):
    values = np.arange(T, dtype=np.float64)
    plan = solved_test._resolve_shock_plan({"e_u": values}, T)

    np.testing.assert_array_equal(plan.matrix(T, 1.0, 0), plan.matrix(T, 1.0, 9))


def test_seeded_count_counts_seeded_entries(solved_test):
    spec = {
        "e_u,e_v": Shock(dist="norm", multivar=True, seed=0),
    }
    plan = solved_test._resolve_shock_plan(spec, T)

    assert plan.seeded_count == 1


def test_unseeded_specs_do_not_count(solved_test):
    plan = solved_test._resolve_shock_plan({"e_u": Shock(dist="norm", seed=None)}, T)

    assert plan.seeded_count == 0


def test_live_shock_requires_a_horizon(solved_test):
    with pytest.raises(ValueError, match="needs a horizon T"):
        solved_test._resolve_shock_plan({"e_u": Shock(dist="norm", seed=1)})
