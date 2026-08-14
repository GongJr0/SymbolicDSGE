# type: ignore
from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sympy as sp
import yaml
from sympy import Symbol
from sympy.core.function import AppliedUndef

import SymbolicDSGE.core.solved_model as solved_model_module
from SymbolicDSGE.core.compiled_model import VariableLayout
from SymbolicDSGE.core.sim_result import SimResult
from SymbolicDSGE.core.solved_model import (
    FirstOrderSolvedModel,
    SecondOrderSolvedModel,
)
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
from SymbolicDSGE.core.solved_model.measurement import (
    build_measurement,
    non_affine_measurement,
)
from SymbolicDSGE.core.solved_model.shocks import (
    shock_unpack,
    simulation_shock_matrix,
)


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
    # The control row is nonzero: an innovation reaches a control
    # contemporaneously, and a state-only loading cannot express that.
    bu = np.array([[1.0], [0.25], [-0.4]], dtype=np.float64)
    hxx = np.array(
        [
            [[0.2, 0.1], [0.1, -0.2]],
            [[0.0, 0.3], [0.3, 0.1]],
        ],
        dtype=np.float64,
    )
    gxx = np.array([[[0.4, -0.1], [-0.1, 0.2]]], dtype=np.float64)
    # Shock-quadratic blocks: nonzero, so the fixture would catch a recursion
    # that silently drops them.
    hxu = np.array([[[0.05], [-0.03]], [[0.02], [0.04]]], dtype=np.float64)
    gxu = np.array([[[-0.06], [0.01]]], dtype=np.float64)
    huu = np.array([[[0.07]], [[-0.02]]], dtype=np.float64)
    guu = np.array([[[0.03]]], dtype=np.float64)
    hss = np.array([0.01, -0.02], dtype=np.float64)
    gss = np.array([0.03], dtype=np.float64)

    compiled = SimpleNamespace(
        idx={"e": 0, "k": 1, "c": 2},
        var_names=["e", "k", "c"],
        # Real layout rather than a stub: x0 is written in declaration order and
        # resolved through it, so a fake would let the permutation drift.
        layout=VariableLayout(
            n_var=3,
            n_declared=3,
            n_generated=0,
            n_exog=1,
            n_state=2,
            n_ctrl=1,
            idx={"e": 0, "k": 1, "c": 2},
            declared_names=("e", "k", "c"),
            canonical_names=("e", "k", "c"),
            state_names=("e", "k"),
            control_names=("c",),
        ),
        n_exog=1,
        n_var=3,
        n_state=2,
        n_ctrl=1,
        observable_names=[],
        shock_names=("eps",),
        shock_idx={"eps": 0},
        config=SimpleNamespace(
            shocks=[Symbol("eps")],
            calibration=SimpleNamespace(parameters={}, shock_std={}),
        ),
    )
    solved = SecondOrderSolvedModel(
        compiled=compiled,
        policy=SimpleNamespace(
            p=hx,
            f=gx,
            order=2,
            hxx=hxx,
            gxx=gxx,
            hxu=hxu,
            gxu=gxu,
            huu=huu,
            guu=guu,
            hss=hss,
            gss=gss,
            steady_state=np.zeros(3, dtype=np.float64),
            A=np.eye(3, dtype=np.float64),
            B=bu,
        ),
    )
    data = {
        "hx": hx,
        "gx": gx,
        "bu": bu,
        "hxx": hxx,
        "gxx": gxx,
        "hxu": hxu,
        "gxu": gxu,
        "huu": huu,
        "guu": guu,
        "hss": hss,
        "gss": gss,
    }
    return solved, data


def _manual_second_order_path(data, shock, x0_state) -> np.ndarray:
    """Every row of a period from the previous state and this period's shock,
    the way Dynare's simult_.m writes the order-2 pruned branch. The controls
    read the same previous state as the states do, not the updated one."""
    T = shock.shape[0]
    bu = data["bu"]
    expected = np.empty((T, 3), dtype=np.float64)
    x1 = x0_state.copy()
    x2 = np.zeros(2, dtype=np.float64)
    for t in range(T):
        u = shock[t]
        xx = np.outer(x1, x1)
        x1_next = data["hx"] @ x1 + bu[:2, 0] * u
        x2_next = (
            data["hx"] @ x2
            + 0.5 * np.einsum("ijk,jk->i", data["hxx"], xx)
            + data["hxu"][:, :, 0] @ x1 * u
            + 0.5 * data["huu"][:, 0, 0] * u * u
            + 0.5 * data["hss"]
        )
        expected[t, 2] = (
            data["gx"][0] @ (x1 + x2)
            + bu[2, 0] * u
            + 0.5 * np.sum(data["gxx"][0] * xx)
            + data["gxu"][0, :, 0] @ x1 * u
            + 0.5 * data["guu"][0, 0, 0] * u * u
            + 0.5 * data["gss"][0]
        )
        x1, x2 = x1_next, x2_next
        expected[t, :2] = x1 + x2
    return expected


def test_solved_model_sim_shapes_and_keys(solved_test):
    T = 12
    out = solved_test.sim(T)

    assert out.X.shape == (T, solved_test.policy.A.shape[0])
    for name in solved_test.compiled.var_names:
        assert out.states[name].shape == (T,)


def test_linear_simulation_kernel_writes_manual_recursion() -> None:
    A = np.array([[0.5, 0.0], [0.25, 0.75]], dtype=np.float64)
    B = np.array([[1.0, -0.5], [0.0, 0.25]], dtype=np.float64)
    x0 = np.array([1.0, -1.0], dtype=np.float64)
    shock_mat = np.array(
        [[0.5, 1.0], [-1.0, 0.0], [0.25, -0.5]],
        dtype=np.float64,
    )
    out = np.empty((shock_mat.shape[0], A.shape[0]), dtype=np.float64)
    py_out = np.empty_like(out)

    zero_ss = np.zeros(A.shape[0], dtype=np.float64)
    simulate_linear_states_into(A, B, x0, shock_mat, out)
    _simulate_linear_states_into_numba.py_func(A, B, x0, shock_mat, py_out, zero_ss)

    expected = np.empty_like(out)
    previous = x0
    for t in range(shock_mat.shape[0]):
        expected[t] = A @ previous + B @ shock_mat[t]
        previous = expected[t]
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(py_out, expected)


def test_affine_observation_kernel_writes_all_states() -> None:
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
    out = np.empty((3, 2), dtype=np.float64)
    py_out = np.empty_like(out)

    affine_observations_into(states, C, d, out)
    _affine_observations_into_numba.py_func(states, C, d, py_out)

    np.testing.assert_allclose(out, states @ C.T + d)
    np.testing.assert_allclose(py_out, states @ C.T + d)


def test_solved_model_sim_matches_manual_state_recursion(solved_test):
    T = 4
    shocks = {
        "e_u": np.array([0.5, -1.0, 0.25, 0.75], dtype=np.float64),
        "e_v": np.array([1.0, 0.0, -0.5, 0.25], dtype=np.float64),
    }

    out = solved_test.sim(T, shocks=shocks, shock_scale=0.5)

    shock_mat = simulation_shock_matrix(
        solved_test.compiled,
        T=T,
        shocks=shocks,
        shock_scale=0.5,
    )
    expected = np.empty_like(out.X)
    previous = solved_test._initial_state()
    for t in range(T):
        expected[t] = (
            solved_test.policy.A @ previous + solved_test.policy.B @ shock_mat[t]
        )
        previous = expected[t]
    np.testing.assert_allclose(out.X, expected)


def test_solved_model_second_order_sim_matches_pruned_recursion() -> None:
    T = 3
    solved, data = _make_second_order_test_model()
    shock = np.array([0.0, 0.05, -0.02], dtype=np.float64)
    x0 = np.array([0.2, -0.1, 99.0], dtype=np.float64)

    out = solved.sim(T, shocks={"eps": shock}, x0=x0).X
    expected = _manual_second_order_path(data, shock, x0[:2])

    np.testing.assert_allclose(out, expected)
    assert out[0, 2] != x0[2]


def test_solved_model_second_order_irf_subtracts_pruned_baseline() -> None:
    T = 3
    solved, data = _make_second_order_test_model()
    shock = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    zero = np.zeros(T, dtype=np.float64)

    out = solved.irf(shocks=["eps"], T=T).X
    expected = _manual_second_order_path(
        data,
        shock,
        np.zeros(2, dtype=np.float64),
    ) - _manual_second_order_path(data, zero, np.zeros(2, dtype=np.float64))

    np.testing.assert_allclose(out, expected)
    assert out.shape == (T, 3)


def test_solved_model_sim_rejects_wrong_shock_length(solved_test):
    with pytest.raises(ValueError, match="must have length"):
        solved_test.sim(8, shocks={"e_u": np.ones(7)})


def test_solved_model_sim_with_observables_includes_measurements(solved_test):
    out = solved_test.sim(10, observables=True)
    for obs in solved_test.compiled.observable_names:
        assert out.observables[obs].shape == (10,)


def test_solved_model_affine_observables_can_drop_initial_row(solved_test):
    out = solved_test.sim(10, observables=True)

    Y = solved_test._simulate_observable_matrix(out.X, drop_initial=True)

    expected = np.column_stack(
        [out.observables[name][1:] for name in solved_test.compiled.observable_names]
    )
    np.testing.assert_allclose(Y, expected)


def test_solved_model_sim_uses_non_affine_measurement_branch(monkeypatch):
    compiled = SimpleNamespace(
        idx={"g": 0, "x": 1},
        var_names=["g", "x"],
        n_exog=1,
        n_var=2,
        n_state=1,
        n_ctrl=1,
        observable_names=["Obs"],
        config=SimpleNamespace(equations=SimpleNamespace(obs_is_affine={"Obs": False})),
    )
    solved = FirstOrderSolvedModel(
        compiled=compiled,
        policy=SimpleNamespace(
            f=np.array([[0.0]], dtype=np.float64),
            order=1,
            A=np.eye(2, dtype=np.float64),
            B=np.zeros((2, 1), dtype=np.float64),
            steady_state=np.zeros(2, dtype=np.float64),
        ),
    )

    def fake_non_affine(compiled_arg, y_names, state):
        assert y_names == ["Obs"]
        return np.arange(state.shape[0], dtype=np.float64).reshape(-1, 1)

    monkeypatch.setattr(
        solved_model_module.measurement,
        "non_affine_measurement",
        fake_non_affine,
    )

    out = solved.sim(3, observables=True)

    assert np.array_equal(out.observables["Obs"], np.array([0.0, 1.0, 2.0]))


def test_solved_model_irf_validation_errors(solved_test):
    with pytest.raises(ValueError, match="At least one shock"):
        solved_test.irf(shocks=[], T=10)
    # A variable name is not a shock name, even a shocked one.
    with pytest.raises(ValueError, match=r"Unknown shock\(s\) \['Pi'\]"):
        solved_test.irf(shocks=["Pi"], T=10)


def test_solved_model_irf_runs_for_a_model_shock(solved_test):
    out = solved_test.irf(shocks=["e_u"], T=8, observables=True)
    assert out.states["u"].shape == (8,)
    assert out.X.shape[0] == 8
    assert {"Infl", "Rate"} <= set(out.observables)


def test_solved_model_transition_plot_renders_observables_and_shocks(
    solved_test, monkeypatch
):
    def fake_irf(self, shocks, T, scale=1.0, observables=False):
        return SimResult(
            var_names=("u", "x"),
            X=np.column_stack([np.linspace(1.0, 0.0, T), np.linspace(-1.0, 0.0, T)]),
            observable_names=("Infl",),
            y=np.linspace(0.0, 1.0, T).reshape(-1, 1),
        )

    monkeypatch.setattr(solved_model_module.SolvedModel, "irf", fake_irf)
    monkeypatch.setattr(plt, "show", lambda: None)

    solved_test.transition_plot(T=3, shocks=["e_u"], observables=True)

    assert plt.get_fignums()
    plt.close("all")


def test_solved_model_get_param_and_get_rho_helpers(solved_test):
    assert (
        solved_test.config.calibration.get_param("beta")
        == solved_test.config.calibration.parameters["beta"]
    )
    assert solved_test.config.calibration.get_rho("e_u", "e_u") == 1.0
    assert solved_test.config.calibration.get_rho("e_u", "e_v", default=0.0) == 0.0

    with pytest.raises(KeyError):
        solved_test.config.calibration.get_param("not_a_param")


def test_solved_model_get_param_default_and_configured_rho(solved_post82):
    assert solved_post82.config.calibration.get_param(
        "missing_param", default=2.5
    ) == pytest.approx(2.5)
    assert solved_post82.config.calibration.get_rho("e_g", "e_z") == pytest.approx(0.36)


def test_solved_model_build_measurement_matrices(solved_test):
    spec = {
        "Obs1": {"lin": {"Pi": 2.0, "x": -1.0}, "const": [1.5, "pi_mean"]},
        "Obs2": {"lin": {"r": 1.0}, "const": [0.0]},
    }
    C, d, names = build_measurement(solved_test.compiled, spec)

    assert C.shape == (2, solved_test.policy.A.shape[0])
    assert d.shape == (2,)
    assert names == ["Obs1", "Obs2"]

    idx = solved_test.compiled.idx
    assert C[0, idx["Pi"]] == 2.0
    assert C[0, idx["x"]] == -1.0
    assert C[1, idx["r"]] == 1.0


def test_solved_model_build_measurement_rejects_unknown_variable(solved_test):
    with pytest.raises(KeyError, match="Variable 'ghost' not found"):
        build_measurement(
            solved_test.compiled, {"Obs": {"lin": {"ghost": 1.0}, "const": []}}
        )


def test_solved_model_build_C_d_from_observables(solved_test):
    C, d = solved_test._build_C_d_from_obs(solved_test.compiled.observable_names)
    m = len(solved_test.compiled.observable_names)
    n = solved_test.policy.A.shape[0]

    assert C.shape == (m, n)
    assert d.shape == (m,)


def test_solved_model_shock_unpack_multivar_key_order_is_canonical(solved_test):
    T = 6

    def mv_shock(cov):
        # deterministic mapping from covariance -> shock matrix
        base = np.array([cov[0, 0], cov[1, 1]], dtype=float)
        return np.tile(base, (T, 1))

    unpack_1 = shock_unpack(solved_test.compiled, {"e_u,e_v": mv_shock})
    unpack_2 = shock_unpack(solved_test.compiled, {"e_v,e_u": mv_shock})

    idx_to_vec_1 = {idx: vec for idx, vec in unpack_1}
    idx_to_vec_2 = {idx: vec for idx, vec in unpack_2}

    assert idx_to_vec_1.keys() == idx_to_vec_2.keys()
    for k in idx_to_vec_1:
        assert np.array_equal(idx_to_vec_1[k], idx_to_vec_2[k])


def test_solved_model_shock_unpack_univariate_callable_and_errors(solved_test):
    out = shock_unpack(
        solved_test.compiled, {"e_u": lambda sig: np.full((4,), sig, dtype=np.float64)}
    )

    assert out[0][0] == solved_test.compiled.shock_idx["e_u"]
    assert np.array_equal(out[0][1], np.full((4,), 0.50, dtype=np.float64))

    with pytest.raises(ValueError, match="is not a model shock"):
        shock_unpack(solved_test.compiled, {"Pi": np.ones((4,), dtype=np.float64)})

    with pytest.raises(TypeError, match="must be a callable or ndarray"):
        shock_unpack(solved_test.compiled, {"e_u": "bad-shock"})


def test_solved_model_shock_unpack_multivariate_error_paths(solved_test):
    def bad_shape(_cov):
        return np.ones((3, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="must return array with shape"):
        shock_unpack(solved_test.compiled, {"e_u,e_v": bad_shape})

    with pytest.raises(TypeError, match="must be a callable or ndarray"):
        shock_unpack(solved_test.compiled, {"e_u,e_v": "bad-shock"})


def test_solved_model_shock_unpack_names_unknown_multivar_member(solved_test):
    # An unknown member of a multivar key is named alongside the entry it came
    # from, so a typo is traceable to the exact grouped spec.
    arr = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(ValueError, match=r"'Pi'.*entry 'e_u,Pi'"):
        shock_unpack(solved_test.compiled, {"e_u,Pi": arr})


def test_solved_model_shock_unpack_rejects_shock_in_two_entries(solved_test):
    # 'e_u' is driven by both a multivar and a univariate entry: each shock may
    # appear in at most one entry, caught by the single pass.
    mv = np.zeros((4, 2), dtype=np.float64)
    uni = np.zeros((4,), dtype=np.float64)
    with pytest.raises(ValueError, match=r"'e_u' is driven by more than one"):
        shock_unpack(solved_test.compiled, {"e_u,e_v": mv, "e_u": uni})


def test_solved_model_kalman_smoke(solved_post82):
    sim = solved_post82.sim(20, observables=True)
    y = pd.DataFrame({"Infl": sim.observables["Infl"], "Rate": sim.observables["Rate"]})

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

    got = non_affine_measurement(solved.compiled, y_names, state)

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
    solved = FirstOrderSolvedModel(
        compiled=compiled,
        policy=SimpleNamespace(
            order=1,
            A=np.eye(1, dtype=np.float64),
            B=np.eye(1, dtype=np.float64),
            steady_state=np.zeros(1, dtype=np.float64),
        ),
    )
    printed = []

    monkeypatch.setattr(
        solved_model_module.base, "KalmanInterface", _FakeKalmanInterface
    )
    monkeypatch.setattr(builtins, "print", lambda *args: printed.append(args))

    out = solved.kalman(
        y=np.zeros((3, 2), dtype=np.float64),
        filter_mode="extended",
        observables=None,
        _debug=True,
    )

    np.testing.assert_allclose(out.x_pred, np.zeros((3, 1), dtype=np.float64))
    assert captured["init"]["meas_addr"] == 456
    assert captured["init"]["jac_addr"] == 789
    assert np.array_equal(captured["init"]["calib_params"], np.array([1.5]))
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
    solved = SecondOrderSolvedModel(
        compiled=compiled,
        policy=SimpleNamespace(
            order=2,
            A=np.eye(1, dtype=np.float64),
            B=np.eye(1, dtype=np.float64),
            steady_state=np.zeros(1, dtype=np.float64),
        ),
    )

    monkeypatch.setattr(
        solved_model_module.base, "KalmanInterface", _FakeKalmanInterface
    )

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
    solved = SecondOrderSolvedModel(
        compiled=compiled,
        policy=SimpleNamespace(
            order=2,
            A=np.eye(1, dtype=np.float64),
            B=np.eye(1, dtype=np.float64),
            steady_state=np.zeros(1, dtype=np.float64),
        ),
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


# --- shock behaviour through the solve --------------------------------------
# From test_shock_impact.py, which existed because lifting every shock
# into a state left the raw symbol in exactly one equation and made the impact
# block a selection matrix. Nothing lifts a shock now, so the two tests pinning
# that selection are gone; `B` is `ghu`, a solve over every variable, and it is
# checked against Dynare in test_dynare_post82_parity. What survives is the
# behaviour those shapes were meant to protect: a loading that is not one, a
# shock reaching several equations, and an equation not normalized on the
# variable its shock drives.
#
# The check on each is a residual: simulate, substitute the paths back into the
# equations the author wrote, and require zero.

t = sp.Symbol("t", integer=True)

TEST_MODEL_PATH = Path(__file__).resolve().parents[2] / "MODELS" / "test.yaml"

#: A fixed draw for both shocks. Pinned so a residual is the solve's, not a seed's.
EPS = np.random.default_rng(20260805).normal(0.0, 0.3, size=(24, 2))


@pytest.fixture
def variant(tmp_path):
    """Solve ``MODELS/test.yaml`` with some of its equations replaced."""

    def build(**equations: str):
        data = yaml.safe_load(TEST_MODEL_PATH.read_text(encoding="utf-8"))
        data["equations"]["model"].update(equations)
        path = tmp_path / f"variant_{len(list(tmp_path.iterdir()))}.yaml"
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


def _worst_residual(model, compiled, solved, eps: np.ndarray) -> float:
    """Largest |residual| of the authored equations on a simulated path.

    A lead is an expectation, so ``E_t y(t+1) = A y(t)`` stands in for the
    realized one. Every equation then holds exactly, which is what makes a
    nonzero here a solve error rather than a forecast error.
    """
    sim = solved.sim(
        T=eps.shape[0],
        shocks={name: eps[:, j] for j, name in enumerate(compiled.shock_names)},
    )
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


def test_simulated_paths_satisfy_the_authored_equations(
    parsed_test, compiled_test, solved_test
):
    model, _ = parsed_test

    assert _worst_residual(model, compiled_test, solved_test, EPS) < 1e-12


def test_non_unit_loading_scales_the_response(solved_test, variant):
    model, compiled, solved = variant(u_process="u(t) = rho_u*u(t-1) + 2.5*e_u")

    shocks = _impulse(12, 2, 0)
    keys = {name: shocks[:, j] for j, name in enumerate(compiled.shock_names)}
    base = solved_test.sim(T=12, shocks=keys)
    scaled = solved.sim(T=12, shocks=keys)

    for name in ("u", "x", "r", "Pi"):
        np.testing.assert_allclose(
            scaled.states[name], 2.5 * base.states[name], rtol=0, atol=1e-13
        )
    assert _worst_residual(model, compiled, solved, EPS) < 1e-12


def test_one_shock_reaches_several_equations_contemporaneously(variant):
    # e_u drives its own process, a forward-looking equation, and a static one.
    # None of the three is the variable shocks names it against. This was the
    # awkward case when a shock had to reach the pencil through one state; it is
    # the ordinary case now that the residual carries innovations directly.
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
    assert _worst_residual(model, compiled, solved, EPS) < 1e-12


def test_unnormalized_equation_keeps_its_effective_loading(solved_test, variant):
    # Scaled through, so u carries 2.5/2 of e_u. Reading the impact off the shock
    # jacobian alone would take the 2.5 and miss the 2.
    model, compiled, solved = variant(u_process="2*u(t) = 2*rho_u*u(t-1) + 2.5*e_u")

    shocks = _impulse(12, 2, 0)
    keys = {name: shocks[:, j] for j, name in enumerate(compiled.shock_names)}
    base = solved_test.sim(T=12, shocks=keys)
    scaled = solved.sim(T=12, shocks=keys)

    for name in ("u", "x", "r", "Pi"):
        np.testing.assert_allclose(
            scaled.states[name], 1.25 * base.states[name], rtol=0, atol=1e-13
        )
    assert _worst_residual(model, compiled, solved, EPS) < 1e-12
