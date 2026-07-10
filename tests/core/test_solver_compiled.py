# type: ignore
from __future__ import annotations

import copy
import ctypes
import random
import textwrap

import numpy as np
import sympy as sp
import yaml
from numpy import float64
import pytest

from SymbolicDSGE.core import DSGESolver, ModelParser, linearize_model


def _nonlinear_compile_yaml() -> str:
    return textwrap.dedent(
        """
        name: "NONLINEAR_COMPILE_LINEARIZE_TEST"
        variables:
          a:
            linearization: log
            steady_state: a_ss
          k:
            linearization: taylor
            steady_state: k_ss
          z: {}
        parameters: [rho_a, rho_k, rho_z, gamma, a_ss, k_ss, sig_a, sig_z]
        shock_map:
          e_a: a
          e_z: z
        observables: []
        equations:
          model:
            - a(t+1) = rho_a*a(t) + (1-rho_a)*a_ss + gamma*z(t) + e_a
            - k(t+1) = rho_k*k(t) + (1-rho_k)*k_ss + z(t)
            - z(t+1) = rho_z*z(t) + e_z
          constraint: {}
          observables: {}
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
          shocks:
            std:
              e_a: sig_a
              e_z: sig_z
            corr: {}
        """
    )


def test_compile_builds_expected_structures(compiled_test):
    c = compiled_test
    n_vars = len(c.config.variables.variables)

    assert c.n_state == 3
    assert c.n_exog == 2
    assert len(c.var_names) == n_vars
    assert len(c.cur_syms) == n_vars
    assert len(c.objective_eqs) == len(c.config.equations.model)
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


def test_construct_measurement_array_dispatchers_are_cached(compiled_test):
    c = compiled_test
    obs = list(c.observable_names)

    assert c.construct_measurement_array_func(
        obs
    ) is c.construct_measurement_array_func(obs)
    assert c.construct_observable_jacobian_array_func(
        obs
    ) is c.construct_observable_jacobian_array_func(obs)


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


def test_measurement_array_dispatchers_match_scalar_dispatchers(compiled_test):
    c = compiled_test
    state = np.linspace(0.05, 0.05 * len(c.cur_syms), len(c.cur_syms), dtype=float64)
    params = np.array(
        [float64(c.config.calibration.parameters[p]) for p in c.calib_params],
        dtype=float64,
    )

    scalar_measure = np.asarray(
        c.construct_measurement_vector_func()(*state, *params), dtype=float64
    )
    array_measure_func = c.construct_measurement_array_func(c.observable_names)
    array_measure = array_measure_func(state, params)

    scalar_jac = np.asarray(c.observable_jacobian(*state, *params), dtype=float64)
    array_jac_func = c.construct_observable_jacobian_array_func(c.observable_names)
    array_jac = array_jac_func(state, params)

    assert getattr(array_measure_func, "_symbolicdsge_array_dispatch", False)
    assert getattr(array_jac_func, "_symbolicdsge_array_dispatch", False)
    assert np.allclose(array_measure, scalar_measure)
    assert np.allclose(array_jac, scalar_jac)


def test_measurement_cfunc_matches_array_dispatcher(compiled_test):
    c = compiled_test
    observables = list(reversed(c.observable_names))
    state = np.linspace(0.05, 0.05 * len(c.cur_syms), len(c.cur_syms), dtype=float64)
    params = np.array(
        [float64(c.config.calibration.parameters[p]) for p in c.calib_params],
        dtype=float64,
    )

    cf = c.construct_measurement_cfunc(observables)
    assert cf is c.construct_measurement_cfunc(observables)

    out = np.empty((len(c.observable_names),), dtype=float64)
    ptr = ctypes.POINTER(ctypes.c_double)
    cf.ctypes(
        state.ctypes.data_as(ptr),
        params.ctypes.data_as(ptr),
        out.ctypes.data_as(ptr),
    )

    expected = c.construct_measurement_array_func(observables)(state, params)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_compile_rejects_unknown_variable_order(parsed_test):
    model, kalman = parsed_test
    solver = DSGESolver(model, kalman)

    with pytest.raises(ValueError, match="do not exist"):
        solver.compile(
            variable_order=[*model.variables.variables, sp.Function("ghost")]
        )


def test_compile_rejects_unknown_param_order(parsed_test):
    model, kalman = parsed_test
    solver = DSGESolver(model, kalman)

    with pytest.raises(ValueError, match="unknown parameters"):
        solver.compile(
            params_order=[*(p.name for p in model.parameters), "ghost_param"]
        )


def test_compile_infers_n_state_and_n_exog(parsed_test):
    model, kalman = parsed_test
    solver = DSGESolver(model, kalman)

    compiled = solver.compile()

    assert compiled.n_state == 3
    assert compiled.n_exog == 2
    assert compiled.var_names[: compiled.n_state] == ["u", "v", "r"]


def test_compile_rejects_equations_with_time_offsets_beyond_one(parsed_test):
    model, kalman = parsed_test
    bad = copy.deepcopy(model)
    t = sp.Symbol("t", integer=True)
    u = bad.variables.variables[0]
    e_u = next(iter(bad.shock_map.keys()))
    bad.equations.model[0] = sp.Eq(u(t + 2), bad.parameters[0] * u(t) + e_u)

    solver = DSGESolver(bad, kalman)
    with pytest.raises(ValueError, match="bad time offsets"):
        solver.compile()


def test_compile_can_linearize_model_on_the_fly(tmp_path):
    path = tmp_path / "nonlinear_compile_linearize.yaml"
    path.write_text(_nonlinear_compile_yaml(), encoding="utf-8")

    model, kalman = ModelParser(path).get_all()
    solver = DSGESolver(model, kalman)

    compiled_from_flag = solver.compile(linearize=True)
    compiled_explicit = DSGESolver(
        linearize_model(model),
        kalman,
    ).compile()

    assert model.symbolically_linearized is False
    assert compiled_from_flag.config is not model
    assert compiled_from_flag.config.symbolically_linearized is True
    assert [
        sp.simplify(a - b)
        for a, b in zip(
            compiled_from_flag.objective_eqs, compiled_explicit.objective_eqs
        )
    ] == [0, 0, 0]

    solved_from_flag = solver.solve(compiled_from_flag)
    solved_explicit = DSGESolver(
        compiled_explicit.config,
        kalman,
    ).solve(compiled_explicit)

    assert solved_from_flag.policy.stab == 0
    assert solved_explicit.policy.stab == 0
    assert np.allclose(solved_from_flag.A, solved_explicit.A)
    assert np.allclose(solved_from_flag.B, solved_explicit.B)


def test_post82_randomized_calibration_still_solves(tmp_path, post82_test_model_path):
    rng = random.Random(42)
    base = yaml.safe_load(post82_test_model_path.read_text(encoding="utf-8"))
    for k, v in list(base["calibration"]["parameters"].items()):
        if isinstance(v, (int, float)):
            base["calibration"]["parameters"][k] = float(v) * (
                1.0 + 0.02 * (2.0 * rng.random() - 1.0)
            )

    out = tmp_path / "post82_randomized.yaml"
    out.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    model, kalman = ModelParser(out).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    solved = solver.solve(compiled)

    assert solved.policy.stab == 0
