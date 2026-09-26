"""Replication shock reconstruction and simulation fallback execution."""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE import Shock
from SymbolicDSGE._ckernels.monte_carlo._runner import run as run_native
from SymbolicDSGE.monte_carlo import MCPipeline, replication_shocks
from SymbolicDSGE.monte_carlo.native_lowering import lower_native_run
from SymbolicDSGE.monte_carlo.step_factories import simulation_step
from SymbolicDSGE.core.shock.plan import get_native_shock_plan
from SymbolicDSGE.core.shock.spec import resolve_shock_plan

T = 16


def _plan(solved_test_model, shocks, shock_scale=1.0):
    resolved = resolve_shock_plan(solved_test_model.compiled, shocks, T)
    return get_native_shock_plan(resolved, T, resolved.n_exog, shock_scale)


def _run_states(solved_test_model, shocks, n_rep, n_jobs):
    """Simulate ``n_rep`` replications and return their retained state paths."""
    pipeline = MCPipeline(
        [
            simulation_step(
                "sim", target="reference", T=T, shocks=shocks, observables=False
            )
        ]
    )
    lowered = lower_native_run(
        pipeline, models={"reference": solved_test_model}, n_rep=n_rep, n_jobs=n_jobs
    )
    assert (
        run_native(lowered.allocation, lowered.steps, lowered.input_bindings).status
        == 0
    )

    layout = lowered.plan["sim"].out_fields["states"]
    retained = lowered.allocation.steps["sim"].float_retained
    return retained[:, layout.offset : layout.offset + layout.flat_count].reshape(
        n_rep, *layout.shape
    )


# --- reproducing one replication --------------------------------------------


@pytest.mark.parametrize(
    "shocks",
    [
        {("e_u", "e_v"): Shock("norm", seed=1)},
        {("e_u",): Shock("norm", seed=1), ("e_v",): Shock("uni", seed=2)},
        # The Python fallback route.
        {("e_u",): Shock("t", seed=3, dist_kwargs={"df": 5})},
    ],
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_replication_shocks_reproduce_a_single_replication(
    solved_test_model, shocks, n_jobs
) -> None:
    step = simulation_step("sim", target="reference", T=T, shocks=shocks)
    states = _run_states(solved_test_model, shocks, 5, n_jobs)

    for rep_idx in (0, 2, 4):
        drawn = replication_shocks(solved_test_model, step, rep_idx)
        expected = solved_test_model.sim(T, shocks=drawn, shock_scale=1.0).X
        np.testing.assert_allclose(states[rep_idx], expected, rtol=1e-12, atol=1e-12)


def test_replication_shocks_rejects_a_deterministic_step(solved_test_model) -> None:
    step = simulation_step("sim", target="reference", T=T, shocks=None)
    with pytest.raises(ValueError, match="draws no shocks"):
        replication_shocks(solved_test_model, step, 0)


# --- the fallback route ------------------------------------------------------


def test_unported_spec_still_runs_off_the_python_slab(solved_test_model) -> None:
    shocks = {("e_u",): Shock("t", seed=3, dist_kwargs={"df": 5})}
    assert _plan(solved_test_model, shocks) is None

    states = _run_states(solved_test_model, shocks, 3, 1)
    resolved = resolve_shock_plan(solved_test_model.compiled, shocks, T)
    for rep_idx in range(3):
        drawn = resolved.matrix(T, 1.0, rep_idx)
        expected = solved_test_model.sim(T, shocks={("e_u",): drawn[:, 0]}).X
        np.testing.assert_allclose(states[rep_idx], expected, rtol=1e-12, atol=1e-12)
