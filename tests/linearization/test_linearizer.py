# type: ignore
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

import SymbolicDSGE.estimation.backend as est_backend
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.linearization import Linearizer, linearize_model


def _write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _nonlinear_model_yaml() -> str:
    return textwrap.dedent(
        """
        name: "NONLINEAR_LINEARIZATION_TEST"
        variables:
          a:
            linearization: log
            steady_state: a_ss
          k:
            linearization: taylor
            steady_state: k_ss
        constrained:
          a: false
          k: false
        parameters: [rho_a, rho_k, gamma, a_ss, k_ss, sig_a]
        shock_map:
          e_a: a
        observables: [AObs]
        equations:
          model:
            - a(t+1) = rho_a*a(t) + (1-rho_a)*a_ss + e_a
            - k(t+1) = rho_k*k(t) + (1-rho_k)*k_ss + gamma*(a(t) - a_ss)
          constraint: {}
          observables:
            AObs: a(t)
        calibration:
          parameters:
            rho_a: 0.8
            rho_k: 0.5
            gamma: 0.2
            a_ss: 2.0
            k_ss: 1.0
            sig_a: 0.1
          shocks:
            std:
              e_a: sig_a
            corr: {}
        """
    )


def test_linearizer_taylor_linearizes_quadratic_equation():
    t = sp.Symbol("t", integer=True)
    x = sp.Function("x")

    linearizer = Linearizer(
        method_dict={x: "taylor"},
        steady_state={x: 1.0},
        equations=[sp.Eq(x(t + 1), x(t) ** 2)],
        time_symbol=t,
        variable_order=[x],
    )

    eq = linearizer.linearize_equations()[0]
    expected = x(t + 1) - 2 * x(t)

    assert sp.simplify(eq.lhs - expected) == 0
    assert eq.rhs == 0


def test_linearizer_log_linearizes_power_equation():
    t = sp.Symbol("t", integer=True)
    alpha, k_ss = sp.symbols("alpha k_ss", positive=True)
    k = sp.Function("k")

    linearizer = Linearizer(
        method_dict={k: "log"},
        steady_state={k: k_ss},
        equations=[sp.Eq(k(t + 1), k_ss ** (1 - alpha) * k(t) ** alpha)],
        time_symbol=t,
        variable_order=[k],
    )

    eq = linearizer.linearize_equations()[0]
    expected = k_ss * (k(t + 1) - alpha * k(t))

    assert sp.simplify(eq.lhs - expected) == 0
    assert eq.rhs == 0


def test_linearizer_mixed_methods_handle_lagged_and_leaded_calls():
    t = sp.Symbol("t", integer=True)
    beta, gamma, k_ss = sp.symbols("beta gamma k_ss")
    x = sp.Function("x")
    k = sp.Function("k")
    z = sp.Function("z")

    linearizer = Linearizer(
        method_dict={x: "taylor", k: "log", z: "none"},
        steady_state={x: 0.0, k: k_ss, z: None},
        equations=[sp.Eq(x(t + 1), beta * x(t) + gamma * (k(t) - k_ss) + z(t - 1))],
        time_symbol=t,
        variable_order=[x, k, z],
    )

    eq = linearizer.linearize_equations()[0]
    expected = x(t + 1) - beta * x(t) - gamma * k_ss * k(t) - z(t - 1)

    assert sp.simplify(eq.lhs - expected) == 0
    assert eq.rhs == 0


def test_linearizer_tracks_missing_steady_states_before_linearization():
    t = sp.Symbol("t", integer=True)
    x = sp.Function("x")

    linearizer = Linearizer(
        method_dict={x: "taylor"},
        steady_state={x: None},
        equations=[sp.Eq(x(t + 1), x(t))],
        time_symbol=t,
        variable_order=[x],
    )

    assert linearizer.missing_steady_states == (x,)
    with pytest.raises(ValueError, match="missing a steady state"):
        linearizer.linearize_equations()


def test_linearizer_rejects_nonpositive_numeric_log_steady_state():
    t = sp.Symbol("t", integer=True)
    x = sp.Function("x")

    linearizer = Linearizer(
        method_dict={x: "log"},
        steady_state={x: 0.0},
        equations=[sp.Eq(x(t + 1), x(t))],
        time_symbol=t,
        variable_order=[x],
    )

    with pytest.raises(ValueError, match="nonpositive steady state"):
        linearizer.linearize_equations()


def test_linearizer_rejects_nonzero_residual_at_expansion_point():
    t = sp.Symbol("t", integer=True)
    x = sp.Function("x")

    linearizer = Linearizer(
        method_dict={x: "taylor"},
        steady_state={x: 1.0},
        equations=[sp.Eq(x(t + 1), x(t) ** 2 + 1)],
        time_symbol=t,
        variable_order=[x],
    )

    with pytest.raises(
        ValueError, match="does not vanish at the supplied steady state"
    ):
        linearizer.linearize_equations()


def test_linearize_model_marks_copy_and_solver_compiles_and_solves(tmp_path):
    path = _write_yaml(
        tmp_path / "nonlinear_linearization.yaml", _nonlinear_model_yaml()
    )

    model, kalman = ModelParser(path).get_all()
    linearized = linearize_model(model)

    assert model.symbolically_linearized is False
    assert linearized.symbolically_linearized is True
    assert linearized is not model
    assert [v.__name__ for v in linearized.variables.variables] == ["a", "k"]

    solver = DSGESolver(linearized, kalman)
    compiled = solver.compile(n_state=2, n_exog=1)
    solved = solver.solve(compiled)

    assert solved.policy.stab == 0
    assert solved.A.shape == (2, 2)
    assert solved.B.shape == (2, 1)


def test_linearized_model_supports_likelihood_evaluation(tmp_path):
    path = _write_yaml(tmp_path / "nonlinear_loglik.yaml", _nonlinear_model_yaml())

    model, kalman = ModelParser(path).get_all()
    linearized = linearize_model(model)
    solver = DSGESolver(linearized, kalman)
    compiled = solver.compile(n_state=2, n_exog=1)

    params = {
        p.name: float(linearized.calibration.parameters[p])
        for p in linearized.parameters
    }
    loglik = est_backend.evaluate_loglik(
        solver=solver,
        compiled=compiled,
        y=np.zeros((6, 1), dtype=np.float64),
        params=params,
        filter_mode=None,
        observables=["AObs"],
        steady_state=np.zeros((2,), dtype=np.float64),
        x0=None,
        p0_mode=None,
        p0_scale=None,
        jitter=None,
        symmetrize=None,
        R=np.eye(1, dtype=np.float64),
    )

    assert np.isfinite(loglik)


def test_linearize_model_rejects_double_linearization(tmp_path):
    path = _write_yaml(tmp_path / "double_linearization.yaml", _nonlinear_model_yaml())

    model = ModelParser(path).get()
    linearized = linearize_model(model)

    with pytest.raises(ValueError, match="already symbolically linearized"):
        linearize_model(linearized)
