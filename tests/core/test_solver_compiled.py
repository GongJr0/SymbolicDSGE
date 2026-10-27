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

from SymbolicDSGE.core import DSGESolver, parse_config
from SymbolicDSGE._ckernels.core import jacobian_eval, measurement_eval


def _nonlinear_compile_yaml() -> str:
    return textwrap.dedent(
        """
        name: "NONLINEAR_COMPILE_LINEARIZE_TEST"
        variables: [a, k, z]
        shocks:
          - e_a
          - e_z
        equations:
          model:
            a_process: "a(t) = rho_a*a(t-1) + (1-rho_a)*a_ss + gamma*z(t-1) + e_a"
            k_process: "k(t) = rho_k*k(t-1) + (1-rho_k)*k_ss + z(t-1)"
            z_process: "z(t) = rho_z*z(t-1) + e_z"
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


def test_compile_builds_expected_structures(parsed_test, compiled_test):
    c = compiled_test
    n_vars = len(parsed_test.variables)

    # test.yaml lags u, v and r, so those three are the states; its two shocks
    # are innovations rather than variables.
    assert c.n_state == 3
    assert c.n_exog == 2
    assert len(c.var_names) == n_vars
    assert len(c.cur_syms) == n_vars
    assert len(c.objective_eqs) == len(parsed_test.equations.model)
    assert set(c.idx.keys()) == set(c.var_names)
    assert set(c.observable_names) == {"Infl", "Rate"}


def test_compiled_parameters_accept_dict_and_vector_forms(parsed_test, compiled_test):
    c = compiled_test
    ss = np.zeros(len(c.var_names))

    par_dict = {
        p: float64(parsed_test.calibration.parameters[p]) for p in c.calib_params
    }
    par_vec = np.array([par_dict[p] for p in c.calib_params])

    C_dict, d_dict = c.build_affine_measurement_matrices(
        par_dict, c.observable_names, ss
    )
    C_vec, d_vec = c.build_affine_measurement_matrices(par_vec, c.observable_names, ss)

    assert np.array_equal(C_dict, C_vec)
    assert np.array_equal(d_dict, d_vec)


def test_compiled_parameters_reject_bad_vector_length(compiled_test):
    c = compiled_test
    ss = np.zeros(len(c.var_names))

    with pytest.raises(ValueError, match="Parameter vector length"):
        c.build_affine_measurement_matrices(
            np.zeros(len(c.calib_params) - 1), c.observable_names, ss
        )


def test_measurement_array_dispatchers_match_lambdify_reference(
    parsed_test, compiled_test
):
    # The native measurement/jacobian cfuncs must agree with an independent
    # sympy.lambdify evaluation of the stored observable exprs (the reference the
    # printer replaced).
    c = compiled_test
    state = np.linspace(0.05, 0.05 * len(c.cur_syms), len(c.cur_syms), dtype=float64)
    params = np.array(
        [float64(parsed_test.calibration.parameters[p]) for p in c.calib_params],
        dtype=float64,
    )
    n_var = len(c.cur_syms)
    args = [*c.cur_syms, *c.calib_params]

    meas_ref = [sp.lambdify(args, e, "numpy") for e in c.observable_eqs]
    scalar_measure = np.asarray([f(*state, *params) for f in meas_ref], dtype=float64)

    jac_ref = [sp.lambdify(args, e, "numpy") for e in c.measurement_jacobian_eqs]
    scalar_jac = np.asarray(
        [f(*state, *params) for f in jac_ref], dtype=float64
    ).reshape(len(c.observable_names), n_var)

    meas_addr = c.construct_measurement_cfunc(c.observable_names).address
    jac_addr = c.construct_measurement_jacobian_cfunc(c.observable_names).address
    n_obs = len(c.observable_names)
    array_measure = measurement_eval(meas_addr, state, params, n_obs)
    array_jac = jacobian_eval(jac_addr, state, params, n_obs, n_var)

    assert np.allclose(array_measure, scalar_measure)
    assert np.allclose(array_jac, scalar_jac)


def test_measurement_cfunc_matches_array_dispatcher(parsed_test, compiled_test):
    c = compiled_test
    observables = list(reversed(c.observable_names))
    state = np.linspace(0.05, 0.05 * len(c.cur_syms), len(c.cur_syms), dtype=float64)
    params = np.array(
        [float64(parsed_test.calibration.parameters[p]) for p in c.calib_params],
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

    expected = measurement_eval(
        c.construct_measurement_cfunc(observables).address,
        state,
        params,
        len(observables),
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_compile_rejects_unknown_variable_order(parsed_test):
    model = parsed_test
    solver = DSGESolver(model)

    with pytest.raises(ValueError, match="Unknown: \\['ghost'\\]"):
        solver.compile(variable_order=[*model.variables, sp.Function("ghost")])


def test_compile_rejects_unknown_param_order(parsed_test):
    model = parsed_test
    solver = DSGESolver(model)

    with pytest.raises(ValueError, match="params_order contains invalid entries"):
        solver.compile(
            params_order=[
                *(p.name for p in model.calibration.parameters),
                "ghost_param",
            ]
        )


def test_compile_infers_n_state_and_n_exog(parsed_test):
    model = parsed_test
    solver = DSGESolver(model)

    compiled = solver.compile()

    assert compiled.n_state == 3
    assert compiled.n_exog == 2
    assert compiled.var_names[: compiled.n_state] == ["u", "v", "r"]


def test_compile_lifts_a_lead_beyond_the_three_dates(parsed_test):
    model = parsed_test
    deep = copy.deepcopy(model)
    t = sp.Symbol("t", integer=True)
    u = deep.variables[0]
    e_u = next(iter(deep.shocks))
    first = next(iter(deep.equations.model))
    param = list(deep.calibration.parameters.keys())[0]
    deep.equations.model[first] = sp.Eq(u(t + 2), param * u(t) + e_u)

    compiled = DSGESolver(deep).compile()

    # A lead aux is read at t+1 and never at t-1, so it joins the controls
    # rather than widening the state block.
    assert "u_lead1" in compiled.layout.generated_names
    assert "u_lead1" in compiled.layout.control_names
    assert "u_lead1" not in compiled.layout.state_names


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

    model = parse_config(out)
    solver = DSGESolver(model)
    compiled = solver.compile()
    solved = solver.solve(compiled)

    assert solved.policy.stab == 0
