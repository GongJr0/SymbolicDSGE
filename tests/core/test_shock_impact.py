"""The shock impact block, on the models its shape has to survive.

Lifting every shock into a state of its own leaves the raw symbol in exactly one
equation, so the state impact is a selection matrix rather than anything read off
the model. These pin the structure on a plain model, then run the three shapes
that would break an impact block built from the shock jacobian instead: a loading
that is not one, a shock reaching several equations, and an equation not
normalized on the variable its shock drives.

The check on each is a residual: simulate, substitute the paths back into the
equations the author wrote, and require zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sympy as sp
import yaml
from sympy.core.function import AppliedUndef

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.desugar import shock_state_name

t = sp.Symbol("t", integer=True)

TEST_MODEL_PATH = Path(__file__).resolve().parents[2] / "MODELS" / "test.yaml"

#: A fixed draw for both shocks. Pinned so a residual is the solve's, not a seed's.
EPS = np.random.default_rng(20260805).normal(0.0, 0.3, size=(24, 2))


@pytest.fixture
def test_model_path():
    return TEST_MODEL_PATH


@pytest.fixture
def variant(tmp_path):
    """Solve ``MODELS/test.yaml`` with some of its equations replaced."""

    def build(**equations: str):
        data = yaml.safe_load(TEST_MODEL_PATH.read_text(encoding="utf-8"))
        data["equations"]["model"].update(equations)
        path = tmp_path / f"model_{len(list(tmp_path.iterdir()))}.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

        model, kalman = ModelParser(path).get_all()
        solver = DSGESolver(model, kalman)
        compiled = solver.compile()
        return model, compiled, solver.solve(compiled=compiled)

    return build


def _impulse(T: int, n_shock: int, shock: int) -> np.ndarray:
    out = np.zeros((T, n_shock), dtype=np.float64)
    out[1, shock] = 1.0
    return out


def _sim(compiled, solved, eps: np.ndarray) -> dict[str, np.ndarray]:
    return solved.sim(
        T=eps.shape[0],
        shocks={name: eps[:, j] for j, name in enumerate(compiled.shock_names)},
    )


def _worst_residual(model, compiled, solved, eps: np.ndarray) -> float:
    """Largest |residual| of the authored equations on a simulated path.

    A lead is an expectation, so ``E_t x(t+1) = A x(t)`` stands in for the
    realized one. Every equation then holds exactly, which is what makes a
    nonzero here a solve error rather than a forecast error.
    """
    sim = _sim(compiled, solved, eps)
    names = list(compiled.var_names)
    path = np.column_stack([sim.states[name] for name in names])
    A = np.asarray(solved.policy.A)
    params = {sym: float(value) for sym, value in model.calibration.parameters.items()}

    worst = 0.0
    for equation in model.equations.model.values():
        expr = (equation.lhs - equation.rhs).subs(params)
        for i in range(2, eps.shape[0] - 2):
            expected = A @ path[i]
            subs: dict[sp.Basic, float] = {}
            for call in expr.atoms(AppliedUndef):
                name = call.func.__name__
                offset = int(sp.simplify(call.args[0] - t))
                subs[call] = (
                    float(expected[names.index(name)])
                    if offset == 1
                    else float(sim.states[name][i + offset])
                )
            for j, shock in enumerate(compiled.shock_names):
                subs[sp.Symbol(shock)] = float(eps[i, j])
            worst = max(worst, abs(float(expr.xreplace(subs))))
    return worst


def _assert_selection(compiled, solved) -> None:
    n_exog = compiled.n_exog
    n_state = compiled.layout.n_state

    np.testing.assert_array_equal(solved.policy.B[:n_exog], np.eye(n_exog))
    np.testing.assert_array_equal(solved.policy.B[n_exog:n_state], 0.0)


def test_impact_is_a_selection_on_the_shock_states(compiled_test, solved_test):
    # Column j is shock j and row j is the state minted for it, so the identity
    # lines up by construction rather than by coincidence of ordering.
    assert compiled_test.layout.exo_state_names == tuple(
        shock_state_name(name) for name in compiled_test.shock_names
    )
    _assert_selection(compiled_test, solved_test)

    # Controls are where a shock's contemporaneous effect shows up.
    assert np.any(solved_test.policy.B[compiled_test.layout.n_state :] != 0.0)


@pytest.mark.parametrize(
    "path_fixture",
    [
        "test_model_path",
        "post82_test_model_path",
        "dense_lkj_test_model_path",
        "rbc_second_order_test_model_path",
    ],
)
def test_no_raw_shock_survives_outside_its_own_state_equation(request, path_fixture):
    # What licenses the hard-coded identity. Were a shock to reach any other row,
    # the impact block would owe that row an entry it does not have.
    model, kalman = ModelParser(request.getfixturevalue(path_fixture)).get_all()
    compiled = DSGESolver(model, kalman).compile()

    for shock in compiled.shock_names:
        carriers = [
            name
            for name, equation in compiled.config.equations.model.items()
            if sp.Symbol(shock) in (equation.lhs - equation.rhs).free_symbols
        ]
        assert carriers == [shock_state_name(shock)]


def test_simulated_paths_satisfy_the_authored_equations(
    parsed_test, compiled_test, solved_test
):
    model, _ = parsed_test

    assert _worst_residual(model, compiled_test, solved_test, EPS) < 1e-12


def test_non_unit_loading_scales_the_response(compiled_test, solved_test, variant):
    model, compiled, solved = variant(u_process="u(t) = rho_u*u(t-1) + 2.5*e_u")

    shocks = _impulse(12, 2, 0)
    keys = {name: shocks[:, j] for j, name in enumerate(compiled.shock_names)}
    base = solved_test.sim(T=12, shocks=keys)
    scaled = solved.sim(T=12, shocks=keys)

    for name in ("u", "x", "r", "Pi"):
        np.testing.assert_allclose(
            scaled.states[name], 2.5 * base.states[name], rtol=0, atol=1e-13
        )
    _assert_selection(compiled, solved)
    assert _worst_residual(model, compiled, solved, EPS) < 1e-12


def test_one_shock_reaches_several_equations_contemporaneously(variant):
    # e_u now drives its own process, a forward-looking equation, and a static
    # one. None of the three is the variable shock_map names it against.
    model, compiled, solved = variant(
        euler="x(t) = x(t+1) - sigma*(r(t) - Pi(t+1)) + u(t) + 0.3*e_u",
        taylor="r_star(t) = rbar + phi_pi*Pi(t) + phi_x*x(t) + v(t) - 0.4*e_u",
    )

    shocks = _impulse(8, 2, 0)
    sim = solved.sim(
        T=8, shocks={name: shocks[:, j] for j, name in enumerate(compiled.shock_names)}
    )

    assert sim.states["x"][0] == 0.0
    assert sim.states["x"][1] != 0.0  # the shock lands on the date it is dated
    _assert_selection(compiled, solved)
    assert _worst_residual(model, compiled, solved, EPS) < 1e-12


def test_unnormalized_equation_keeps_its_effective_loading(
    compiled_test, solved_test, variant
):
    # Scaled through, so u carries 2.5/2 of e_u. Reading the impact off the shock
    # jacobian would take the 2.5 and miss the 2.
    model, compiled, solved = variant(u_process="2*u(t) = 2*rho_u*u(t-1) + 2.5*e_u")

    shocks = _impulse(12, 2, 0)
    keys = {name: shocks[:, j] for j, name in enumerate(compiled.shock_names)}
    base = solved_test.sim(T=12, shocks=keys)
    scaled = solved.sim(T=12, shocks=keys)

    for name in ("u", "x", "r", "Pi"):
        np.testing.assert_allclose(
            scaled.states[name], 1.25 * base.states[name], rtol=0, atol=1e-13
        )
    _assert_selection(compiled, solved)
    assert _worst_residual(model, compiled, solved, EPS) < 1e-12
