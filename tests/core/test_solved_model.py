# type: ignore
from __future__ import annotations

import builtins
from types import SimpleNamespace

import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import pytest
from sympy import Symbol

import SymbolicDSGE.core.solved_model as solved_model_module
from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.kalman.interface import KalmanInterface


@njit
def _obs_shift(x, alpha):
    return x + alpha


@njit
def _obs_scale(x, alpha):
    return x * alpha


def test_solved_model_sim_shapes_and_keys(solved_test):
    T = 12
    out = solved_test.sim(T)

    assert "_X" in out
    assert out["_X"].shape == (T + 1, solved_test.A.shape[0])
    for name in solved_test.compiled.var_names:
        assert name in out
        assert out[name].shape == (T + 1,)


def test_solved_model_sim_rejects_wrong_shock_length(solved_test):
    with pytest.raises(ValueError, match="must have length"):
        solved_test.sim(8, shocks={"u": np.ones(7)})


def test_solved_model_sim_with_observables_includes_measurements(solved_test):
    out = solved_test.sim(10, observables=True)
    for obs in solved_test.compiled.observable_names:
        assert obs in out
        assert out[obs].shape == (11,)


def test_solved_model_sim_uses_non_affine_measurement_branch(monkeypatch):
    compiled = SimpleNamespace(
        idx={"g": 0, "x": 1},
        var_names=["g", "x"],
        n_state=1,
        observable_names=["Obs"],
        config=SimpleNamespace(equations=SimpleNamespace(obs_is_affine={"Obs": False})),
    )
    solved = solved_model_module.SolvedModel(
        compiled=compiled,
        policy=SimpleNamespace(f=np.array([[0.0]], dtype=np.float64)),
        A=np.eye(2, dtype=np.float64),
        B=np.zeros((2, 1), dtype=np.float64),
    )

    def fake_non_affine(self, y_names, state):
        assert y_names == ["Obs"]
        return np.arange(state.shape[0], dtype=np.float64).reshape(-1, 1)

    monkeypatch.setattr(
        solved_model_module.SolvedModel,
        "_non_affine_measurement",
        fake_non_affine,
    )

    out = solved.sim(3, observables=True)

    assert np.array_equal(out["Obs"], np.array([0.0, 1.0, 2.0, 3.0]))


def test_solved_model_irf_validation_errors(solved_test):
    with pytest.raises(ValueError, match="At least one shock"):
        solved_test.irf(shocks=[], T=10)
    with pytest.raises(ValueError, match="not found in exogenous"):
        solved_test.irf(shocks=["Pi"], T=10)


def test_solved_model_irf_runs_for_exogenous_shock(solved_test):
    out = solved_test.irf(shocks=["u"], T=8, observables=True)
    assert out["u"].shape == (9,)
    assert "_X" in out
    assert "Infl" in out and "Rate" in out


def test_solved_model_transition_plot_renders_observables_and_shocks(
    solved_test, monkeypatch
):
    def fake_irf(self, shocks, T, scale=1.0, observables=False):
        return {
            "_X": np.zeros((T + 1, 3), dtype=np.float64),
            "Infl": np.linspace(0.0, 1.0, T + 1),
            "u": np.linspace(1.0, 0.0, T + 1),
            "x": np.linspace(-1.0, 0.0, T + 1),
        }

    monkeypatch.setattr(solved_model_module.SolvedModel, "irf", fake_irf)
    monkeypatch.setattr(plt, "show", lambda: None)

    solved_test.transition_plot(T=3, shocks=["u"], observables=True)

    assert plt.get_fignums()
    plt.close("all")


def test_solved_model_get_param_and_get_rho_helpers(solved_test):
    assert (
        solved_test._get_param("beta")
        == solved_test.config.calibration.parameters["beta"]
    )
    assert solved_test._get_rho("e_u", "e_u") == 1.0
    assert solved_test._get_rho("e_u", "e_v", default=0.0) == 0.0

    with pytest.raises(KeyError):
        solved_test._get_param("not_a_param")


def test_solved_model_get_param_default_and_configured_rho(solved_post82):
    assert solved_post82._get_param("missing_param", default=2.5) == pytest.approx(2.5)
    assert solved_post82._get_rho("e_g", "e_z") == pytest.approx(0.36)


def test_solved_model_build_measurement_matrices(solved_test):
    spec = {
        "Obs1": {"lin": {"Pi": 2.0, "x": -1.0}, "const": [1.5, "pi_mean"]},
        "Obs2": {"lin": {"r": 1.0}, "const": [0.0]},
    }
    C, d, names = solved_test._build_measurement(spec)

    assert C.shape == (2, solved_test.A.shape[0])
    assert d.shape == (2,)
    assert names == ["Obs1", "Obs2"]

    idx = solved_test.compiled.idx
    assert C[0, idx["Pi"]] == 2.0
    assert C[0, idx["x"]] == -1.0
    assert C[1, idx["r"]] == 1.0


def test_solved_model_build_measurement_rejects_unknown_variable(solved_test):
    with pytest.raises(KeyError, match="Variable 'ghost' not found"):
        solved_test._build_measurement({"Obs": {"lin": {"ghost": 1.0}, "const": []}})


def test_solved_model_build_C_d_from_observables(solved_test):
    C, d = solved_test._build_C_d_from_obs(solved_test.compiled.observable_names)
    m = len(solved_test.compiled.observable_names)
    n = solved_test.A.shape[0]

    assert C.shape == (m, n)
    assert d.shape == (m,)


def test_solved_model_shock_unpack_multivar_key_order_is_canonical(solved_test):
    T = 6

    def mv_shock(cov):
        # deterministic mapping from covariance -> shock matrix
        base = np.array([cov[0, 0], cov[1, 1]], dtype=float)
        return np.tile(base, (T, 1))

    unpack_1 = solved_test._shock_unpack({"u,v": mv_shock})
    unpack_2 = solved_test._shock_unpack({"v,u": mv_shock})

    idx_to_vec_1 = {idx: vec for idx, vec in unpack_1}
    idx_to_vec_2 = {idx: vec for idx, vec in unpack_2}

    assert idx_to_vec_1.keys() == idx_to_vec_2.keys()
    for k in idx_to_vec_1:
        assert np.array_equal(idx_to_vec_1[k], idx_to_vec_2[k])


def test_solved_model_shock_unpack_univariate_callable_and_errors(solved_test):
    out = solved_test._shock_unpack(
        {"u": lambda sig: np.full((4,), sig, dtype=np.float64)}
    )

    assert out[0][0] == solved_test.compiled.idx["u"]
    assert np.array_equal(out[0][1], np.full((4,), 0.50, dtype=np.float64))

    with pytest.raises(ValueError, match="not found in exogenous"):
        solved_test._shock_unpack({"Pi": np.ones((4,), dtype=np.float64)})

    with pytest.raises(TypeError, match="must be a callable or ndarray"):
        solved_test._shock_unpack({"u": "bad-shock"})


def test_solved_model_shock_unpack_multivariate_error_paths(solved_test):
    def bad_shape(_cov):
        return np.ones((3, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="must return array with shape"):
        solved_test._shock_unpack({"u,v": bad_shape})

    with pytest.raises(TypeError, match="must be a callable or ndarray"):
        solved_test._shock_unpack({"u,v": "bad-shock"})


def test_solved_model_kalman_smoke(solved_post82):
    sim = solved_post82.sim(20, observables=True)
    y = pd.DataFrame({"Infl": sim["Infl"][1:], "Rate": sim["Rate"][1:]})

    out = solved_post82.kalman(y, observables=["Infl", "Rate"])
    assert out is not None


def test_solved_model_non_affine_measurement_and_jit_cache():
    alpha = Symbol("alpha")
    compiled = SimpleNamespace(
        observable_funcs=[_obs_shift, _obs_scale],
        observable_names=["ObsShift", "ObsScale"],
        cur_syms=[Symbol("x")],
        config=SimpleNamespace(calibration=SimpleNamespace(parameters={alpha: 2.0})),
    )
    solved = solved_model_module.SolvedModel(
        compiled=compiled,
        policy=None,
        A=np.eye(1, dtype=np.float64),
        B=np.zeros((1, 1), dtype=np.float64),
    )
    state = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)

    f1 = solved._make_jit_measurement(2)
    f2 = solved._make_jit_measurement(2)
    out = solved._non_affine_measurement(["ObsScale", "ObsShift"], state)

    assert f1 is f2
    assert np.allclose(
        out,
        np.array([[2.0, 3.0], [4.0, 4.0], [6.0, 5.0]], dtype=np.float64),
    )


def test_solved_model_kalman_extended_uses_default_obs_and_debug(monkeypatch):
    alpha = Symbol("alpha")
    captured = {}

    class _FakeKalmanInterface:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self._debug_info = None

        def _ML_estimate_R_diag(self, scale_factor=1.0):
            captured["scale_factor"] = scale_factor

        def filter(self, x0=None, _debug=False):
            captured["filter"] = {"x0": x0, "_debug": _debug}
            self._debug_info = {"debug": True}
            return "kalman-result"

    compiled = SimpleNamespace(
        calib_params=[alpha],
        observable_names=["ObsA", "ObsB"],
        construct_measurement_array_func=lambda obs: ("h", tuple(obs)),
        construct_observable_jacobian_array_func=lambda obs: ("H", tuple(obs)),
        config=SimpleNamespace(calibration=SimpleNamespace(parameters={alpha: 1.5})),
        kalman=SimpleNamespace(y_names=["ObsB", "ObsA"]),
    )
    solved = solved_model_module.SolvedModel(
        compiled=compiled,
        policy=None,
        A=np.eye(1, dtype=np.float64),
        B=np.eye(1, dtype=np.float64),
    )
    printed = []

    monkeypatch.setattr(solved_model_module, "KalmanInterface", _FakeKalmanInterface)
    monkeypatch.setattr(builtins, "print", lambda *args: printed.append(args))

    out = solved.kalman(
        y=np.zeros((3, 2), dtype=np.float64),
        filter_mode="extended",
        observables=None,
        estimate_R_diag=True,
        R_scale=2.5,
        _debug=True,
    )

    assert out == "kalman-result"
    assert captured["init"]["h_func"] == ("h", ("ObsA", "ObsB"))
    assert captured["init"]["H_jac"] == ("H", ("ObsA", "ObsB"))
    assert np.array_equal(captured["init"]["calib_params"], np.array([1.5]))
    assert captured["init"]["estimate_R_diag"] is True
    assert captured["scale_factor"] == pytest.approx(2.5)
    assert captured["filter"] == {"x0": None, "_debug": True}
    assert printed == [({"debug": True},)]


def test_kalman_interface_rebuilds_symbolic_R_from_current_calibration(
    post82_test_model_path,
):
    model, kalman = ModelParser(post82_test_model_path).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile(n_state=3, n_exog=3)

    compiled.config.calibration.parameters[Symbol("meas_infl")] = 2.0
    compiled.config.calibration.parameters[Symbol("meas_rate")] = 3.0
    compiled.config.calibration.parameters[Symbol("meas_rho_ir")] = 0.1

    solved = solver.solve(compiled)
    y = pd.DataFrame({"Infl": [0.0, 0.0], "Rate": [0.0, 0.0]})
    ki = KalmanInterface(
        model=solved,
        filter_mode="linear",
        observables=["Infl", "Rate"],
        y=y,
    )

    assert np.allclose(
        ki.R,
        np.array([[4.0, 0.6], [0.6, 9.0]], dtype=np.float64),
    )
    assert np.allclose(
        solved.kalman_config.R,
        np.eye(3, dtype=np.float64),
    )


def test_solved_model_to_dict_contains_main_fields(solved_test):
    d = solved_test.to_dict()
    assert "compiled" in d
    assert "policy" in d
    assert "A" in d
    assert "B" in d
