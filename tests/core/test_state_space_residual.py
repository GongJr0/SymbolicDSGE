"""The assembled state space against the equations it was built from.

``(A, B)`` is only ever checked for internal consistency. ``A`` comes out of the
Klein solve, but ``B`` is written by ``sdsge_assemble_state_space`` as
``[I(n_exog); 0]`` stacked under ``f @ B_state``, and innovations are substituted
to zero before the solve, so nothing constrains ``B`` to the model. These
simulate the state space and feed the path back through the user's own
equations.

See https://github.com/GongJr0/SymbolicDSGE/issues/390.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE.core import DSGESolver, ModelParser

from _oracles.state_space import simulate, worst_residual

POST82 = "tests/fixtures/models/POST82.yaml"
RBC_RELABELLED = "tests/fixtures/models/rbc_second_order.yaml"
RBC_LAGGED = "tests/fixtures/models/rbc_lagged.yaml"


def _solve(path):
    model, kalman = ModelParser(path).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    return solver, compiled, solver.solve(compiled, order=1)


def _steady_state(compiled):
    calib = compiled.config.calibration.parameters
    seeds = {"c": "c_ss", "k": "k_ss"}
    return np.array(
        [
            float(calib[sp.Symbol(seeds[name])]) if name in seeds else 0.0
            for name in compiled.var_names
        ]
    )


def _shocks(n_step, n_exog, scale, seed=0):
    draws = np.random.default_rng(seed).normal(size=(n_step, n_exog)) * scale
    draws[0] = 0.0
    return draws


def test_transition_alone_satisfies_the_equations():
    """Control. With no innovations the path is driven by ``A`` only, so this
    isolates ``B`` from the rest of the solve."""
    solver, compiled, sol = _solve(POST82)
    shocks = np.zeros((12, compiled.n_exog))

    state = np.array([0.4, -0.3, 0.2])
    path = np.zeros((shocks.shape[0], len(compiled.var_names)))
    path[0] = np.concatenate([state, np.real(sol.policy.f) @ state])
    for k in range(1, path.shape[0]):
        path[k] = sol.A @ path[k - 1]

    worst = worst_residual(compiled, solver.t, sol.A, path, shocks)
    assert max(worst.values()) < 1e-10, worst


def test_linear_model_state_space_satisfies_the_equations():
    """POST82 is linear, so a correct state space leaves no residual at all.

    ``e_r`` targets a Taylor rule carrying ``Pi(t)`` and ``x(t)``, so the true
    impact of the innovation on ``r`` is a fixed point rather than one.
    """
    solver, compiled, sol = _solve(POST82)
    shocks = _shocks(12, compiled.n_exog, scale=1.0)
    path = simulate(sol.A, sol.B, shocks)

    worst = worst_residual(compiled, solver.t, sol.A, path, shocks)
    assert max(worst.values()) < 1e-10, worst


@pytest.mark.parametrize("path_", [RBC_RELABELLED, RBC_LAGGED])
def test_nonlinear_model_state_space_is_first_order_accurate(path_):
    """A first-order state space leaves a residual of order ``scale ** 2``, so
    halving the innovations must cut it by about four. A residual linear in the
    scale halves instead, which is what a wrong ``B`` produces."""
    solver, compiled, sol = _solve(path_)
    ss = _steady_state(compiled)

    def resid(scale):
        shocks = _shocks(12, compiled.n_exog, scale=scale)
        path = simulate(sol.A, sol.B, shocks)
        return max(worst_residual(compiled, solver.t, sol.A, path, shocks, ss).values())

    coarse, fine = resid(0.02), resid(0.01)
    assert coarse > 0.0
    assert (
        coarse / fine > 3.5
    ), f"residual scales linearly, not quadratically: {coarse / fine:.3f}"
