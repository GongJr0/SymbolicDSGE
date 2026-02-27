# type: ignore
from __future__ import annotations

import copy
import random

import numpy as np
import sympy as sp
import yaml
from numpy import float64
import pytest

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
