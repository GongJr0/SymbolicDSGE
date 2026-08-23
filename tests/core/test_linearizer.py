# type: ignore
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE.core import DSGESolver, ModelParser, linearize_model
from SymbolicDSGE.core.linearization import Linearizer
from SymbolicDSGE.estimation import Estimator
from SymbolicDSGE.kalman.config import KalmanConfig


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
            ss_seed: a_ss
          k:
            linearization: taylor
            ss_seed: k_ss
        shocks:
          - e_a
        observables: [AObs]
        equations:
          model:
            a_process: "a(t) = rho_a*a(t-1) + (1-rho_a)*a_ss + e_a"
            k_process: "k(t) = rho_k*k(t-1) + (1-rho_k)*k_ss + gamma*(a(t-1) - a_ss)"
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


def _mixed_methods_nonlinear_yaml() -> str:
    return textwrap.dedent(
        """
        name: "NONLINEAR_EQUIVALENCE_TEST"
        variables:
          a:
            linearization: log
            ss_seed: a_ss
          k:
            linearization: taylor
            ss_seed: k_ss
          z: {}
        shocks:
          - e_a
          - e_z
        observables: [ZObs]
        equations:
          model:
            a_process: "a(t) = rho_a*a(t-1) + (1-rho_a)*a_ss + gamma*z(t-1) + e_a"
            k_process: "k(t) = rho_k*k(t-1) + (1-rho_k)*k_ss + z(t-1)"
            z_process: "z(t) = rho_z*z(t-1) + e_z"
          constraint: {}
          observables:
            ZObs: z(t)
        calibration:
          parameters:
            rho_a: 0.8
            rho_k: 0.5
            rho_z: 0.3
            gamma: 0.2
            a_ss: 2.0
            k_ss: 1.0
            sig_a: 0.1
            sig_z: 0.05
            meas_z: 1.0
          shocks:
            std:
              e_a: sig_a
              e_z: sig_z
            corr: {}
        kalman:
          R:
            std:
              ZObs: meas_z
            corr: {}
        """
    )


def _mixed_methods_hand_linearized_yaml() -> str:
    return textwrap.dedent(
        """
        name: "HAND_LINEARIZED_EQUIVALENCE_TEST"
        variables: [a, k, z]
        shocks:
          - e_a
          - e_z
        observables: [ZObs]
        equations:
          model:
            a_process: "a_ss*a(t) = rho_a*a_ss*a(t-1) + gamma*z(t-1) + e_a"
            k_process: "k(t) = rho_k*k(t-1) + z(t-1)"
            z_process: "z(t) = rho_z*z(t-1) + e_z"
          constraint: {}
          observables:
            ZObs: z(t)
        calibration:
          parameters:
            rho_a: 0.8
            rho_k: 0.5
            rho_z: 0.3
            gamma: 0.2
            a_ss: 2.0
            k_ss: 1.0
            sig_a: 0.1
            sig_z: 0.05
            meas_z: 1.0
          shocks:
            std:
              e_a: sig_a
              e_z: sig_z
            corr: {}
        kalman:
          R:
            std:
              ZObs: meas_z
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
    expected = sp.Eq(x(t + 1), 2 * x(t))

    assert sp.simplify((eq.lhs - eq.rhs) - (expected.lhs - expected.rhs)) == 0


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
    expected = sp.Eq(k_ss * k(t + 1), alpha * k_ss * k(t))

    assert sp.simplify((eq.lhs - eq.rhs) - (expected.lhs - expected.rhs)) == 0


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
    expected = sp.Eq(x(t + 1), beta * x(t) + gamma * k_ss * k(t) + z(t - 1))

    assert sp.simplify((eq.lhs - eq.rhs) - (expected.lhs - expected.rhs)) == 0


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
    compiled = solver.compile()
    solved = solver.solve(compiled)

    assert solved.policy.stab == 0
    # a and k both occur at t-1, so the model is all states and no controls.
    assert compiled.n_state == 2
    assert solved.policy.A.shape == (2, 2)
    assert solved.policy.B.shape == (2, 1)


def test_linearized_model_supports_likelihood_evaluation(tmp_path):
    path = _write_yaml(tmp_path / "nonlinear_loglik.yaml", _nonlinear_model_yaml())

    model, _ = ModelParser(path).get_all()
    linearized = linearize_model(model)
    # The nonlinear fixture has no `kalman:` block, so supply R directly.
    kalman = KalmanConfig(R=np.eye(1, dtype=np.float64))
    solver = DSGESolver(linearized, kalman)
    compiled = solver.compile()

    est = Estimator(
        compiled=compiled,
        y=np.zeros((6, 1), dtype=np.float64),
        observables=["AObs"],
        filter_mode="linear",
        estimated_params=[linearized.parameters[0].name],
        ss_seed=np.zeros((len(compiled.layout.declared_names),), dtype=np.float64),
        R=np.eye(1, dtype=np.float64),
    )

    assert np.isfinite(float(est.loglik(est.theta0())))


def test_linearizer_accepts_model_config_directly(tmp_path):
    path = _write_yaml(
        tmp_path / "linearizer_from_config.yaml", _nonlinear_model_yaml()
    )

    model = ModelParser(path).get()
    linearizer = Linearizer(model)

    assert [spec.original.__name__ for spec in linearizer.context.specs] == ["a", "k"]
    assert linearizer.method_dict[model.variables.variables[0]].value == "log"
    assert linearizer.method_dict[model.variables.variables[1]].value == "taylor"


def test_linearizer_matches_hand_linearized_solution_matrices(tmp_path):
    nonlinear_path = _write_yaml(
        tmp_path / "mixed_methods_nonlinear.yaml",
        _mixed_methods_nonlinear_yaml(),
    )
    hand_linearized_path = _write_yaml(
        tmp_path / "mixed_methods_hand_linearized.yaml",
        _mixed_methods_hand_linearized_yaml(),
    )

    nonlinear_model, nonlinear_kalman = ModelParser(nonlinear_path).get_all()
    hand_model, hand_kalman = ModelParser(hand_linearized_path).get_all()

    auto_linearized = linearize_model(nonlinear_model)

    nonlinear_solver = DSGESolver(auto_linearized, nonlinear_kalman)
    hand_solver = DSGESolver(hand_model, hand_kalman)

    nonlinear_compiled = nonlinear_solver.compile()
    hand_compiled = hand_solver.compile()

    nonlinear_solved = nonlinear_solver.solve(nonlinear_compiled)
    hand_solved = hand_solver.solve(hand_compiled)

    assert nonlinear_solved.policy.stab == 0
    assert hand_solved.policy.stab == 0
    assert np.allclose(
        nonlinear_solved.policy.A,
        hand_solved.policy.A,
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.allclose(
        nonlinear_solved.policy.B,
        hand_solved.policy.B,
        rtol=1e-10,
        atol=1e-10,
    )


def test_linearize_model_rejects_double_linearization(tmp_path):
    path = _write_yaml(tmp_path / "double_linearization.yaml", _nonlinear_model_yaml())

    model = ModelParser(path).get()
    linearized = linearize_model(model)

    with pytest.raises(ValueError, match="already symbolically linearized"):
        linearize_model(linearized)
