# type: ignore
from __future__ import annotations

import builtins
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sympy as sp
from sympy import Symbol

import SymbolicDSGE.core.solved_model as solved_model_module
from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE._ckernels.core import (
    affine_observations_into,
    simulate_linear_states_into,
)
from _oracles.core import (
    _affine_observations_into_numba,
    _simulate_linear_states_into_numba,
)
from SymbolicDSGE.kalman.filter import FilterRawResult, UnscentedFilterRawResult
from SymbolicDSGE.kalman.interface import KalmanInterface


def _raw_filter_result(T: int = 3, n: int = 1, m: int = 2) -> FilterRawResult:
    x = np.zeros((T, n), dtype=np.float64)
    y = np.zeros((T, m), dtype=np.float64)
    P = np.zeros((T, n, n), dtype=np.float64)
    S = np.zeros((T, m, m), dtype=np.float64)
    return FilterRawResult(
        status=0,
        x_pred=x,
        x_filt=x,
        P_pred=P,
        P_filt=P,
        y_pred=y,
        y_filt=y,
        innov=y,
        std_innov=y,
        S=S,
        eps_hat=None,
        loglik=np.float64(0.0),
    )


def _raw_unscented_result(
    T: int = 3,
    n_state: int = 1,
    n_var: int = 1,
) -> UnscentedFilterRawResult:
    x = np.zeros((T, n_var), dtype=np.float64)
    xb = np.zeros((T, n_state), dtype=np.float64)
    y = np.zeros((T, 2), dtype=np.float64)
    P = np.zeros((T, 2 * n_state, 2 * n_state), dtype=np.float64)
    S = np.zeros((T, 2, 2), dtype=np.float64)
    return UnscentedFilterRawResult(
        status=0,
        x_pred=x,
        x_filt=x,
        x1_pred=xb,
        x2_pred=xb,
        x1_filt=xb,
        x2_filt=xb,
        P_pred=P,
        P_filt=P,
        y_pred=y,
        y_filt=y,
        innov=y,
        std_innov=y,
        S=S,
        eps_hat=None,
        loglik=np.float64(0.0),
    )


def _make_second_order_test_model() -> tuple[solved_model_module.SolvedModel, dict]:
    hx = np.array([[0.5, 0.1], [0.0, 0.8]], dtype=np.float64)
    gx = np.array([[2.0, -1.0]], dtype=np.float64)
    bx = np.array([[1.0], [0.25]], dtype=np.float64)
    hxx = np.array(
        [
            [[0.2, 0.1], [0.1, -0.2]],
            [[0.0, 0.3], [0.3, 0.1]],
        ],
        dtype=np.float64,
    )
    gxx = np.array([[[0.4, -0.1], [-0.1, 0.2]]], dtype=np.float64)
    hss = np.array([0.01, -0.02], dtype=np.float64)
    gss = np.array([0.03], dtype=np.float64)

    compiled = SimpleNamespace(
        idx={"e": 0, "k": 1, "c": 2},
        var_names=["e", "k", "c"],
        n_exog=1,
        n_state=2,
        observable_names=[],
        config=SimpleNamespace(
            shock_map={Symbol("eps"): Symbol("e")},
            calibration=SimpleNamespace(parameters={}, shock_std={}),
        ),
    )
    solved = solved_model_module.SolvedModel(
        compiled=compiled,
        policy=SimpleNamespace(
            p=hx,
            f=gx,
            order=2,
            hxx=hxx,
            gxx=gxx,
            hss=hss,
            gss=gss,
            steady_state=np.zeros(3, dtype=np.float64),
        ),
        A=np.eye(3, dtype=np.float64),
        B=np.vstack([bx, np.zeros((1, 1), dtype=np.float64)]),
    )
    data = {
        "hx": hx,
        "gx": gx,
        "bx": bx,
        "hxx": hxx,
        "gxx": gxx,
        "hss": hss,
        "gss": gss,
    }
    return solved, data


def _manual_second_order_path(data, shock, x0_state) -> np.ndarray:
    T = shock.shape[0]
    expected = np.empty((T + 1, 3), dtype=np.float64)
    x1 = x0_state.copy()
    x2 = np.zeros(2, dtype=np.float64)
    for t in range(T + 1):
        state = x1 + x2
        outer = np.outer(x1, x1)
        expected[t, :2] = state
        expected[t, 2] = (
            data["gx"][0] @ state
            + 0.5 * np.sum(data["gxx"][0] * outer)
            + 0.5 * data["gss"][0]
        )
        if t == T:
            break
        x1_next = data["hx"] @ x1 + data["bx"][:, 0] * shock[t]
        x2_next = (
            data["hx"] @ x2
            + 0.5 * np.einsum("ijk,jk->i", data["hxx"], outer)
            + 0.5 * data["hss"]
        )
        x1, x2 = x1_next, x2_next
    return expected


def test_solved_model_sim_shapes_and_keys(solved_test):
    T = 12
    out = solved_test.sim(T)

    assert "_X" in out
    assert out["_X"].shape == (T + 1, solved_test.A.shape[0])
    for name in solved_test.compiled.var_names:
        assert name in out
        assert out[name].shape == (T + 1,)


def test_linear_simulation_kernel_writes_manual_recursion() -> None:
    A = np.array([[0.5, 0.0], [0.25, 0.75]], dtype=np.float64)
    B = np.array([[1.0, -0.5], [0.0, 0.25]], dtype=np.float64)
    x0 = np.array([1.0, -1.0], dtype=np.float64)
    shock_mat = np.array(
        [[0.5, 1.0], [-1.0, 0.0], [0.25, -0.5]],
        dtype=np.float64,
    )
    out = np.empty((shock_mat.shape[0] + 1, A.shape[0]), dtype=np.float64)
    py_out = np.empty_like(out)

    simulate_linear_states_into(A, B, x0, shock_mat, out)
    _simulate_linear_states_into_numba.py_func(A, B, x0, shock_mat, py_out)

    expected = np.empty_like(out)
    expected[0] = x0
    for t in range(shock_mat.shape[0]):
        expected[t + 1] = A @ expected[t] + B @ shock_mat[t]
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(py_out, expected)


def test_affine_observation_kernel_writes_with_state_offset() -> None:
    states = np.array(
        [
            [1.0, 2.0],
            [3.0, -1.0],
            [0.5, 4.0],
        ],
        dtype=np.float64,
    )
    C = np.array([[2.0, -0.5], [0.0, 1.5]], dtype=np.float64)
    d = np.array([1.0, -2.0], dtype=np.float64)
    out = np.empty((2, 2), dtype=np.float64)
    py_out = np.empty_like(out)

    affine_observations_into(states, C, d, 1, out)
    _affine_observations_into_numba.py_func(states, C, d, 1, py_out)

    np.testing.assert_allclose(out, states[1:] @ C.T + d)
    np.testing.assert_allclose(py_out, states[1:] @ C.T + d)


def test_solved_model_sim_matches_manual_state_recursion(solved_test):
    T = 4
    shocks = {
        "u": np.array([0.5, -1.0, 0.25, 0.75], dtype=np.float64),
        "v": np.array([1.0, 0.0, -0.5, 0.25], dtype=np.float64),
    }

    out = solved_test.sim(T, shocks=shocks, shock_scale=0.5)

    shock_mat = solved_test._simulation_shock_matrix(
        T=T,
        shocks=shocks,
        shock_scale=0.5,
    )
    expected = np.empty_like(out["_X"])
    expected[0] = solved_test._simulation_initial_state(None)
    for t in range(T):
        expected[t + 1] = solved_test.A @ expected[t] + solved_test.B @ shock_mat[t]
    np.testing.assert_allclose(out["_X"], expected)


def test_solved_model_second_order_sim_matches_pruned_recursion() -> None:
    T = 3
    solved, data = _make_second_order_test_model()
    shock = np.array([0.0, 0.05, -0.02], dtype=np.float64)
    x0 = np.array([0.2, -0.1, 99.0], dtype=np.float64)

    out = solved.sim(T, shocks={"e": shock}, x0=x0)["_X"]
    expected = _manual_second_order_path(data, shock, x0[:2])

    np.testing.assert_allclose(out, expected)
    assert out[0, 2] != x0[2]


def test_solved_model_second_order_irf_subtracts_pruned_baseline() -> None:
    T = 3
    solved, data = _make_second_order_test_model()
    shock = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    zero = np.zeros(T, dtype=np.float64)

    out = solved.irf(shocks=["e"], T=T)["_X"]
    expected = _manual_second_order_path(
        data,
        shock,
        np.zeros(2, dtype=np.float64),
    ) - _manual_second_order_path(data, zero, np.zeros(2, dtype=np.float64))

    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(out[0], np.zeros(3, dtype=np.float64))


def test_solved_model_sim_rejects_wrong_shock_length(solved_test):
    with pytest.raises(ValueError, match="must have length"):
        solved_test.sim(8, shocks={"u": np.ones(7)})


def test_solved_model_sim_with_observables_includes_measurements(solved_test):
    out = solved_test.sim(10, observables=True)
    for obs in solved_test.compiled.observable_names:
        assert obs in out
        assert out[obs].shape == (11,)


def test_solved_model_affine_observables_can_drop_initial_row(solved_test):
    out = solved_test.sim(10, observables=True)

    Y = solved_test._simulate_observable_matrix(out["_X"], drop_initial=True)

    expected = np.column_stack(
        [out[name][1:] for name in solved_test.compiled.observable_names]
    )
    np.testing.assert_allclose(Y, expected)


def test_solved_model_sim_uses_non_affine_measurement_branch(monkeypatch):
    compiled = SimpleNamespace(
        idx={"g": 0, "x": 1},
        var_names=["g", "x"],
        n_exog=1,
        n_state=1,
        observable_names=["Obs"],
        config=SimpleNamespace(equations=SimpleNamespace(obs_is_affine={"Obs": False})),
    )
    solved = solved_model_module.SolvedModel(
        compiled=compiled,
        policy=SimpleNamespace(f=np.array([[0.0]], dtype=np.float64), order=1),
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

    with pytest.raises(ValueError, match="not an exogenous model variable"):
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


def test_solved_model_shock_unpack_names_unknown_multivar_member(solved_test):
    # An unknown member of a multivar key is named alongside the entry it came
    # from, so a typo is traceable to the exact grouped spec.
    arr = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(ValueError, match=r"'Pi'.*entry 'u,Pi'"):
        solved_test._shock_unpack({"u,Pi": arr})


def test_solved_model_shock_unpack_rejects_variable_in_two_entries(solved_test):
    # 'u' is driven by both a multivar and a univariate entry: each exogenous
    # variable may appear in at most one entry, caught by the single pass.
    mv = np.zeros((4, 2), dtype=np.float64)
    uni = np.zeros((4,), dtype=np.float64)
    with pytest.raises(ValueError, match=r"'u' is driven by more than one"):
        solved_test._shock_unpack({"u,v": mv, "u": uni})


def test_solved_model_kalman_smoke(solved_post82):
    sim = solved_post82.sim(20, observables=True)
    y = pd.DataFrame({"Infl": sim["Infl"][1:], "Rate": sim["Rate"][1:]})

    out = solved_post82.kalman(y, observables=["Infl", "Rate"])
    assert out is not None


def test_solved_model_non_affine_measurement_matches_reference(solved_test):
    # Native measurement-path cfunc must match an independent sympy.lambdify eval
    # of the observable exprs, with output columns remapped to y_names order.
    solved = solved_test
    compiled = solved.compiled
    n_var = len(compiled.cur_syms)
    rng = np.random.default_rng(0)
    state = rng.normal(size=(4, n_var)).astype(np.float64)

    # Reversed so the sorted-cfunc columns must be remapped back to y_names order.
    y_names = list(reversed(compiled.observable_names))

    args = [*compiled.cur_syms, *compiled.calib_params]
    par = np.array(
        [
            np.float64(compiled.config.calibration.parameters[p])
            for p in compiled.calib_params
        ],
        dtype=np.float64,
    )
    obs_lambd = {
        name: sp.lambdify(args, compiled.observable_eqs[i], "numpy")
        for i, name in enumerate(compiled.observable_names)
    }

    got = solved._non_affine_measurement(y_names, state)

    expected = np.empty((state.shape[0], len(y_names)), dtype=np.float64)
    for j, name in enumerate(y_names):
        f = obs_lambd[name]
        for t in range(state.shape[0]):
            expected[t, j] = f(*state[t], *par)

    assert np.allclose(got, expected)


def test_solved_model_kalman_extended_uses_default_obs_and_debug(monkeypatch):
    alpha = Symbol("alpha")
    captured = {}

    class _FakeKalmanInterface:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self._debug_info = None

        def _ML_estimate_R_diag(self, scale_factor=1.0):
            captured["scale_factor"] = scale_factor

        def filter_raw(self, x0=None, _debug=False):
            captured["filter_raw"] = {"x0": x0, "_debug": _debug}
            self._debug_info = {"debug": True}
            return _raw_filter_result()

    compiled = SimpleNamespace(
        calib_params=[alpha],
        observable_names=["ObsA", "ObsB"],
        construct_measurement_cfunc=lambda obs: SimpleNamespace(address=456),
        construct_observable_jacobian_cfunc=lambda obs: SimpleNamespace(address=789),
        config=SimpleNamespace(calibration=SimpleNamespace(parameters={alpha: 1.5})),
        kalman=SimpleNamespace(y_names=["ObsB", "ObsA"]),
    )
    solved = solved_model_module.SolvedModel(
        compiled=compiled,
        policy=SimpleNamespace(order=1),
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

    np.testing.assert_allclose(out.x_pred, np.zeros((3, 1), dtype=np.float64))
    assert captured["init"]["meas_addr"] == 456
    assert captured["init"]["jac_addr"] == 789
    assert np.array_equal(captured["init"]["calib_params"], np.array([1.5]))
    assert captured["init"]["estimate_R_diag"] is True
    assert captured["scale_factor"] == pytest.approx(2.5)
    assert captured["filter_raw"] == {"x0": None, "_debug": True}
    assert printed == [({"debug": True},)]


def test_solved_model_kalman_unscented_uses_measurement_cfunc(monkeypatch):
    alpha = Symbol("alpha")
    captured = {}

    class _FakeKalmanInterface:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self._debug_info = None

        def filter_raw(self, x0=None, _debug=False):
            captured["filter_raw"] = {"x0": x0, "_debug": _debug}
            return _raw_unscented_result()

    compiled = SimpleNamespace(
        calib_params=[alpha],
        observable_names=["ObsA", "ObsB"],
        construct_measurement_cfunc=lambda obs: SimpleNamespace(
            address=456,
            obs=tuple(obs),
        ),
        construct_observable_jacobian_cfunc=lambda obs: SimpleNamespace(address=789),
        config=SimpleNamespace(calibration=SimpleNamespace(parameters={alpha: 1.5})),
        kalman=SimpleNamespace(y_names=["ObsB", "ObsA"]),
    )
    solved = solved_model_module.SolvedModel(
        compiled=compiled,
        policy=SimpleNamespace(order=2),
        A=np.eye(1, dtype=np.float64),
        B=np.eye(1, dtype=np.float64),
    )

    monkeypatch.setattr(solved_model_module, "KalmanInterface", _FakeKalmanInterface)

    out = solved.kalman(
        y=np.zeros((3, 2), dtype=np.float64),
        filter_mode="unscented",
        observables=None,
        x0=np.array([0.1], dtype=np.float64),
    )

    np.testing.assert_allclose(out.x_pred, np.zeros((3, 1), dtype=np.float64))
    assert captured["init"]["filter_mode"] == "unscented"
    assert captured["init"]["meas_addr"] == 456
    assert np.array_equal(captured["init"]["calib_params"], np.array([1.5]))
    assert np.array_equal(captured["filter_raw"]["x0"], np.array([0.1]))


def test_solved_model_kalman_unscented_rejects_return_shocks(monkeypatch):
    alpha = Symbol("alpha")
    compiled = SimpleNamespace(
        calib_params=[alpha],
        observable_names=["ObsA"],
        construct_measurement_cfunc=lambda obs: SimpleNamespace(address=456),
        construct_observable_jacobian_cfunc=lambda obs: SimpleNamespace(address=789),
        config=SimpleNamespace(calibration=SimpleNamespace(parameters={alpha: 1.5})),
        kalman=SimpleNamespace(y_names=["ObsA"]),
    )
    solved = solved_model_module.SolvedModel(
        compiled=compiled,
        policy=SimpleNamespace(order=2),
        A=np.eye(1, dtype=np.float64),
        B=np.eye(1, dtype=np.float64),
    )

    with pytest.raises(ValueError, match="return_shocks is not supported"):
        solved.kalman(
            y=np.zeros((3, 1), dtype=np.float64),
            filter_mode="unscented",
            return_shocks=True,
        )


def test_kalman_interface_rebuilds_symbolic_R_from_current_calibration(
    post82_test_model_path,
):
    model, kalman = ModelParser(post82_test_model_path).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()

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
