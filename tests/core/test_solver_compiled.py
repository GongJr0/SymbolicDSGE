# type: ignore
from __future__ import annotations

import copy
import random
import textwrap

import numpy as np
import pandas as pd
import sympy as sp
import yaml
from numpy import float64
import pytest

from SymbolicDSGE import _linearsolve as linearsolve
from SymbolicDSGE.core import DSGESolver, ModelParser


def test_compile_builds_expected_structures(compiled_test):
    c = compiled_test
    n_vars = len(c.config.variables)

    assert c.n_state == 3
    assert c.n_exog == 2
    assert len(c.var_names) == n_vars
    assert len(c.cur_syms) == n_vars
    assert len(c.objective_eqs) == len(c.config.equations.model)
    assert len(c.objective_funcs) == len(c.config.equations.model)
    assert set(c.idx.keys()) == set(c.var_names)
    assert set(c.observable_names) == {"Infl", "Rate"}


def test_compiled_equations_accept_dict_and_vector_parameters(compiled_test):
    c = compiled_test
    n = len(c.var_names)
    fwd = np.zeros(n, dtype=complex)
    cur = np.zeros(n, dtype=complex)

    par_dict = {
        p.name: float64(c.config.calibration.parameters[p]) for p in c.calib_params
    }
    out_from_dict = c.equations(fwd, cur, par_dict)

    par_vec = np.array([par_dict[p.name] for p in c.calib_params], dtype=complex)
    out_from_vec = c.equations(fwd, cur, par_vec)

    assert out_from_dict.shape == (len(c.objective_eqs),)
    assert list(map(str, out_from_dict)) == list(map(str, out_from_vec))


def test_compiled_equations_reject_bad_parameter_vector_length(compiled_test):
    c = compiled_test
    n = len(c.var_names)

    with pytest.raises(ValueError, match="Parameter vector length"):
        c.equations(np.zeros(n), np.zeros(n), np.zeros(len(c.calib_params) - 1))


def test_construct_measurement_vector_func_returns_expected_length(compiled_test):
    c = compiled_test
    f = c.construct_measurement_vector_func()
    n = len(c.cur_syms)
    p = len(c.calib_params)
    args = [float64(0.0)] * (n + p)

    out = f(*args)
    assert out.shape == (len(c.observable_names),)


def test_construct_measurement_vector_func_is_cached(compiled_test):
    c = compiled_test

    assert (
        c.construct_measurement_vector_func() is c.construct_measurement_vector_func()
    )


def test_construct_objective_vector_func_is_cached(compiled_test):
    c = compiled_test

    assert c.construct_objective_vector_func() is c.construct_objective_vector_func()


def test_objective_vector_func_matches_compiled_equations(compiled_test):
    c = compiled_test
    objective = c.construct_objective_vector_func()
    n = len(c.var_names)
    fwd = np.linspace(0.1, 0.1 * n, n, dtype=np.complex128)
    cur = np.linspace(-0.05, 0.05, n, dtype=np.complex128)
    params = np.array(
        [float64(c.config.calibration.parameters[p]) for p in c.calib_params],
        dtype=np.complex128,
    )

    expected = c.equations(fwd, cur, params)
    actual = objective(
        np.ascontiguousarray(fwd),
        np.ascontiguousarray(cur),
        np.ascontiguousarray(params),
    )
    assert np.allclose(actual, expected)

    fwd_step = fwd.copy()
    cur_step = cur.copy()
    fwd_step[0] += 1e-30j
    cur_step[-1] -= 2e-30j

    expected_step = c.equations(fwd_step, cur_step, params)
    actual_step = objective(
        np.ascontiguousarray(fwd_step),
        np.ascontiguousarray(cur_step),
        np.ascontiguousarray(params),
    )
    assert np.allclose(actual_step, expected_step)


def test_klein_helpers_use_numba_function_cache():
    assert type(linearsolve._to_complex._cache).__name__ == "FunctionCache"
    assert type(linearsolve._klein_postprocess._cache).__name__ == "FunctionCache"


def test_compile_rejects_unknown_variable_order(parsed_test):
    model, kalman = parsed_test
    solver = DSGESolver(model, kalman)

    with pytest.raises(ValueError, match="do not exist"):
        solver.compile(
            variable_order=[*model.variables, sp.Function("ghost")],
            n_state=3,
            n_exog=2,
        )


def test_compile_rejects_unknown_param_order(parsed_test):
    model, kalman = parsed_test
    solver = DSGESolver(model, kalman)

    with pytest.raises(ValueError, match="unknown parameters"):
        solver.compile(
            params_order=[*(p.name for p in model.parameters), "ghost_param"],
            n_state=3,
            n_exog=2,
        )


def test_compile_requires_n_state_and_n_exog(parsed_test):
    model, kalman = parsed_test
    solver = DSGESolver(model, kalman)

    with pytest.raises(ValueError, match="must provide n_state and n_exog"):
        solver.compile()


def test_compile_rejects_equations_with_time_offsets_beyond_one(parsed_test):
    model, kalman = parsed_test
    bad = copy.deepcopy(model)
    t = sp.Symbol("t", integer=True)
    u = bad.variables[0]
    e_u = next(iter(bad.shock_map.keys()))
    bad.equations.model[0] = sp.Eq(u(t + 2), bad.parameters[0] * u(t) + e_u)

    solver = DSGESolver(bad, kalman)
    with pytest.raises(ValueError, match="bad time offsets"):
        solver.compile(n_state=3, n_exog=2)


def test_post82_randomized_calibration_still_solves(tmp_path):
    rng = random.Random(42)
    base = yaml.safe_load(open("MODELS/POST82.yaml", "r", encoding="utf-8"))
    for k, v in list(base["calibration"]["parameters"].items()):
        if isinstance(v, (int, float)):
            base["calibration"]["parameters"][k] = float(v) * (
                1.0 + 0.02 * (2.0 * rng.random() - 1.0)
            )

    out = tmp_path / "post82_randomized.yaml"
    out.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    model, kalman = ModelParser(out).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile(n_state=3, n_exog=3)
    solved = solver.solve(compiled)

    assert solved.policy.stab == 0


def test_solver_log_linear_solves_positive_steady_model(tmp_path):
    config = textwrap.dedent(
        """
        name: "POSITIVE_TEST"
        variables: [a, k]
        constrained:
          a: false
          k: false
        parameters: [rho_a, rho_k, sig_a]
        shock_map:
          e_a: a
        observables: [AObs, KObs]
        equations:
          model:
            - a(t+1) = rho_a*a(t) + (1-rho_a) + e_a
            - k(t+1) = rho_k*k(t) + (1-rho_k)*a(t)
          constraint: {}
          observables:
            AObs: a(t)
            KObs: k(t)
        calibration:
          parameters:
            rho_a: 0.8
            rho_k: 0.5
            sig_a: 0.1
          shocks:
            std:
              e_a: sig_a
            corr: {}
        """
    )
    path = tmp_path / "positive_loglinear.yaml"
    path.write_text(config, encoding="utf-8")

    model, kalman = ModelParser(path).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile(n_state=2, n_exog=1)
    solved = solver.solve(
        compiled,
        steady_state=np.ones((2,), dtype=np.float64),
        log_linear=True,
    )

    assert solved.policy.stab == 0
    assert solved.A.shape == (2, 2)
    assert solved.B.shape == (2, 1)


def test_linearsolve_falls_back_without_numeric_dispatcher():
    params = pd.Series({"rho": 0.9}, dtype=float)

    def equations(fwd, cur, par):
        return np.array([fwd["x"] - par["rho"] * cur["x"]], dtype=complex)

    mdl = linearsolve.model(
        equations=equations,
        variables=["x"],
        parameters=params,
        n_states=1,
        n_exo_states=0,
    )
    mdl.set_ss(pd.Series({"x": 1.0}, dtype=float))
    mdl.linear_approximation()

    assert mdl.a.shape == (1, 1)
    assert mdl.b.shape == (1, 1)
    assert mdl.a[0, 0] == pytest.approx(1.0)
    assert mdl.b[0, 0] == pytest.approx(0.9)
