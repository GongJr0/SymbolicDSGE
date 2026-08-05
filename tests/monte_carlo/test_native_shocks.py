# type: ignore
"""The shock block the Monte Carlo loop draws for itself (#374).

The native draw replaces a prematerialized ``(n_rep, T, n_exog)`` slab, so the
two things worth pinning are that the deterministic half is exact and that the
stochastic half is addressable. Every draw here is recomputed independently
from :mod:`SymbolicDSGE._ckernels.rng`, so a covariance factorization, a
location shift, a uniform rescaling, or a column scatter that drifts shows up as
an exact mismatch rather than as a distributional one.

Specifications the kernel cannot reproduce stay on the Python route, and the
last tests here check that route is still chosen and still bit-identical to a
resolved :class:`ShockPlan`.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE import DSGESolver, ModelParser, Shock
from SymbolicDSGE._ckernels.monte_carlo._runner import run as run_native
from SymbolicDSGE._ckernels.rng import (
    philox_standard_normal,
    philox_standard_uniform,
)
from SymbolicDSGE.monte_carlo import MCPipeline, replication_shocks
from SymbolicDSGE.monte_carlo.shock_native import (
    SHOCK_NORMAL,
    SHOCK_UNIFORM,
    build_native_plan,
    native_shock_entries,
    native_shock_families,
    native_shock_scratch,
)
from SymbolicDSGE.monte_carlo.step_factories import simulation_step

T = 16


@pytest.fixture(scope="module")
def solved():
    model, kalman = ModelParser("MODELS/test.yaml").get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile())


def _plan(solved, shocks, shock_scale=1.0):
    step = simulation_step(T=T, shocks=shocks, shock_scale=shock_scale)
    return build_native_plan(solved, step, T)


def _entries(solved, shocks):
    families = native_shock_families(shocks)
    resolved = solved._resolve_shock_plan(shocks, T)
    return native_shock_entries(resolved, families)


# --- eligibility ------------------------------------------------------------


def test_native_families_accepts_normal_and_univariate_uniform() -> None:
    assert native_shock_families({"e_u": Shock("norm", seed=0)}) == {
        "e_u": SHOCK_NORMAL
    }
    assert native_shock_families({"e_u,e_v": Shock("norm", multivar=True, seed=0)}) == {
        "e_u,e_v": SHOCK_NORMAL
    }
    assert native_shock_families({"e_u": Shock("uni", seed=0)}) == {
        "e_u": SHOCK_UNIFORM
    }


@pytest.mark.parametrize(
    "shocks",
    [
        None,
        {},
        {"e_u": Shock("t", seed=0, dist_kwargs={"df": 5})},
        {"e_u,e_v": Shock("uni", multivar=True, seed=0)},
        {"e_u": Shock("norm", seed=0, shock_arr=np.zeros(T))},
        {"e_u": np.zeros(T)},
        {"e_u": lambda scale: np.zeros(T)},
        # One ineligible entry sends the whole specification back.
        {
            "e_u": Shock("norm", seed=0),
            "e_v": Shock("t", seed=1, dist_kwargs={"df": 5}),
        },
    ],
)
def test_native_families_rejects_unported_specs(shocks) -> None:
    assert native_shock_families(shocks) is None
    assert native_shock_scratch(shocks, T) == 0


def test_native_scratch_sizes_on_the_widest_entry() -> None:
    shocks = {"e_u,e_v": Shock("norm", multivar=True, seed=0)}
    assert native_shock_scratch(shocks, T) == T * 2
    assert native_shock_scratch({"e_u": Shock("norm", seed=0)}, T) == T


# --- the draw itself --------------------------------------------------------


def test_univariate_normal_draw_is_the_scaled_engine_stream(solved) -> None:
    shocks = {"e_u": Shock("norm", seed=7)}
    (entry,) = _entries(solved, shocks)
    block = _plan(solved, shocks).draw(3)

    z = philox_standard_normal(entry.key, 0, 3, 0, T)
    expected = np.zeros((T, solved.compiled.n_exog))
    expected[:, entry.columns[0]] = z * entry.factor[0, 0]

    np.testing.assert_array_equal(block, expected)


def test_multivariate_normal_draw_applies_the_covariance_factor(solved) -> None:
    shocks = {"e_u,e_v": Shock("norm", multivar=True, seed=11)}
    (entry,) = _entries(solved, shocks)
    block = _plan(solved, shocks).draw(2)

    width = len(entry.columns)
    z = philox_standard_normal(entry.key, 0, 2, 0, T * width).reshape(T, width)
    expected = np.zeros((T, solved.compiled.n_exog))
    expected[:, entry.columns] = z @ entry.factor.T

    np.testing.assert_array_equal(block, expected)
    # The factor is the thing under test, so it must not be the identity.
    assert not np.allclose(entry.factor, np.eye(width))


def test_normal_draw_applies_the_location_shift(solved) -> None:
    shocks = {"e_u": Shock("norm", seed=7, dist_kwargs={"loc": 2.5})}
    (entry,) = _entries(solved, shocks)
    block = _plan(solved, shocks).draw(0)

    z = philox_standard_normal(entry.key, 0, 0, 0, T)
    np.testing.assert_array_equal(
        block[:, entry.columns[0]], z * entry.factor[0, 0] + 2.5
    )


def test_uniform_draw_maps_the_unit_interval_onto_the_support(solved) -> None:
    shocks = {"e_u": Shock("uni", seed=4, dist_kwargs={"loc": -1.0})}
    (entry,) = _entries(solved, shocks)
    block = _plan(solved, shocks).draw(5)

    u = philox_standard_uniform(entry.key, 0, 5, 0, T)
    np.testing.assert_array_equal(
        block[:, entry.columns[0]], entry.low + entry.span * u
    )


def test_shock_scale_multiplies_the_whole_block(solved) -> None:
    shocks = {"e_u": Shock("norm", seed=7)}
    plain = _plan(solved, shocks, shock_scale=1.0).draw(1)
    scaled = _plan(solved, shocks, shock_scale=2.5).draw(1)

    np.testing.assert_array_equal(scaled, 2.5 * plain)


def test_untargeted_columns_stay_zero(solved) -> None:
    shocks = {"e_u": Shock("norm", seed=7)}
    (entry,) = _entries(solved, shocks)
    block = _plan(solved, shocks).draw(0)

    untargeted = [
        i for i in range(solved.compiled.n_exog) if i not in set(entry.columns)
    ]
    assert untargeted
    np.testing.assert_array_equal(block[:, untargeted], 0.0)


# --- addressing -------------------------------------------------------------


def test_a_seeded_spec_replays_across_plans(solved) -> None:
    shocks = {"e_u": Shock("norm", seed=1), "e_v": Shock("uni", seed=2)}
    first = _plan(solved, shocks)
    second = _plan(solved, shocks)

    for rep_idx in (0, 1, 97):
        np.testing.assert_array_equal(first.draw(rep_idx), second.draw(rep_idx))


def test_replications_do_not_share_a_stream(solved) -> None:
    plan = _plan(solved, {"e_u,e_v": Shock("norm", multivar=True, seed=1)})
    blocks = [plan.draw(rep_idx) for rep_idx in range(4)]

    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            assert not np.array_equal(blocks[i], blocks[j])


def test_entries_sharing_a_seed_stay_independent(solved) -> None:
    shocks = {"e_u": Shock("norm", seed=5), "e_v": Shock("norm", seed=5)}
    entries = _entries(solved, shocks)
    block = _plan(solved, shocks).draw(0)

    assert entries[0].key == entries[1].key
    left = block[:, entries[0].columns[0]]
    right = block[:, entries[1].columns[0]]
    assert not np.array_equal(left, right)


def test_an_unseeded_spec_redraws_each_run(solved) -> None:
    shocks = {"e_u": Shock("norm", seed=None)}
    assert not np.array_equal(
        _plan(solved, shocks).draw(0), _plan(solved, shocks).draw(0)
    )


def test_negative_replication_index_is_rejected(solved) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _plan(solved, {"e_u": Shock("norm", seed=0)}).draw(-1)


# --- the run reads the same blocks -----------------------------------------


def _run_states(solved, shocks, n_rep, n_jobs):
    """Simulate ``n_rep`` replications and return their retained state paths."""
    pipeline = MCPipeline(
        [
            simulation_step(
                "sim", target="reference", T=T, shocks=shocks, observables=False
            )
        ]
    )
    lowered = pipeline.lower_native(reference=solved, n_rep=n_rep, n_jobs=n_jobs)
    assert (
        run_native(lowered.allocation, lowered.steps, lowered.input_bindings).status
        == 0
    )

    layout = lowered.plan["sim"].out_fields["states"]
    retained = lowered.allocation.steps["sim"].float_retained
    return retained[:, layout.offset : layout.offset + layout.flat_count].reshape(
        n_rep, *layout.shape
    )


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("n_rep", [3, 8])
def test_run_states_match_the_addressed_blocks(solved, n_rep, n_jobs) -> None:
    shocks = {"e_u,e_v": Shock("norm", multivar=True, seed=1)}
    states = _run_states(solved, shocks, n_rep, n_jobs)

    plan = _plan(solved, shocks)
    (entry,) = _entries(solved, shocks)
    for rep_idx in range(n_rep):
        block = plan.draw(rep_idx)[:, entry.columns]
        expected = solved.sim(T, shocks={"e_u,e_v": block})["_X"]
        np.testing.assert_allclose(states[rep_idx], expected, rtol=1e-12, atol=1e-12)


# --- reproducing one replication --------------------------------------------


@pytest.mark.parametrize(
    "shocks",
    [
        {"e_u,e_v": Shock("norm", multivar=True, seed=1)},
        {"e_u": Shock("norm", seed=1), "e_v": Shock("uni", seed=2)},
        # The Python fallback route.
        {"e_u": Shock("t", seed=3, dist_kwargs={"df": 5})},
    ],
)
def test_replication_shocks_reproduce_a_single_replication(solved, shocks) -> None:
    step = simulation_step("sim", target="reference", T=T, shocks=shocks)
    states = _run_states(solved, shocks, 5, 1)

    for rep_idx in (0, 2, 4):
        drawn = replication_shocks(solved, step, rep_idx)
        expected = solved.sim(T, shocks=drawn, shock_scale=1.0)["_X"]
        np.testing.assert_allclose(states[rep_idx], expected, rtol=1e-12, atol=1e-12)


def test_replication_shocks_rejects_a_deterministic_step(solved) -> None:
    step = simulation_step("sim", target="reference", T=T, shocks=None)
    with pytest.raises(ValueError, match="draws no shocks"):
        replication_shocks(solved, step, 0)


# --- the fallback route ------------------------------------------------------


def test_unported_spec_still_runs_off_the_python_slab(solved) -> None:
    shocks = {"e_u": Shock("t", seed=3, dist_kwargs={"df": 5})}
    step = simulation_step(T=T, target="reference", shocks=shocks, observables=False)
    assert build_native_plan(solved, step, T) is None

    states = _run_states(solved, shocks, 3, 1)
    resolved = solved._resolve_shock_plan(shocks, T)
    for rep_idx in range(3):
        drawn = resolved.matrix(T, 1.0, rep_idx * resolved.seeded_count)
        expected = solved.sim(T, shocks={"e_u": drawn[:, 0]})["_X"]
        np.testing.assert_allclose(states[rep_idx], expected, rtol=1e-12, atol=1e-12)
