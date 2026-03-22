# type: ignore
from __future__ import annotations

import builtins
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sympy import Symbol

import SymbolicDSGE.kalman.interface as interface_module
from SymbolicDSGE.kalman.interface import KalmanInterface
from SymbolicDSGE.kalman.validator import FilterMode

FLOAT = np.float64

E_U = Symbol("e_u")
E_V = Symbol("e_v")
SIG_U = Symbol("sig_u")
SIG_V = Symbol("sig_v")
RHO_UV = Symbol("rho_uv")
MEAS_A = Symbol("meas_a")
MEAS_B = Symbol("meas_b")


def _make_stub_model(
    *,
    kalman_config=...,
    params: dict[Symbol | str, float] | None = None,
):
    observable_names = ["ObsA", "ObsB"]
    var_names = ["u", "v", "x"]
    parameters = {
        SIG_U: 0.2,
        SIG_V: 0.3,
        RHO_UV: 0.25,
        MEAS_A: 4.0,
        MEAS_B: 9.0,
    }
    if params is not None:
        parameters.update(params)

    calibration = SimpleNamespace(
        parameters=parameters,
        shock_std={E_U: SIG_U, E_V: SIG_V},
        shock_corr={frozenset({E_U, E_V}): RHO_UV},
    )
    config = SimpleNamespace(
        calibration=calibration,
        shock_map={E_U: Symbol("u"), E_V: Symbol("v")},
    )
    compiled = SimpleNamespace(
        observable_names=observable_names,
        var_names=var_names,
        n_exog=2,
    )

    rows = {
        "ObsA": np.array([1.0, 0.0, 0.0], dtype=FLOAT),
        "ObsB": np.array([0.0, 1.0, 1.0], dtype=FLOAT),
    }
    constants = {"ObsA": 1.0, "ObsB": -1.0}

    def build_measurement(obs_names):
        C = np.vstack([rows[name] for name in obs_names]).astype(FLOAT)
        d = np.array([constants[name] for name in obs_names], dtype=FLOAT)
        return C, d

    if kalman_config is ...:
        kalman_config = SimpleNamespace(
            y_names=["ObsB", "ObsA"],
            R=np.array([[4.0, 0.6], [0.6, 9.0]], dtype=FLOAT),
            jitter=0.125,
            symmetrize=True,
            P0=SimpleNamespace(
                mode="diag",
                scale=2.0,
                diag={"u": 1.0, "v": 3.0, "x": 5.0},
            ),
            R_builder=None,
            R_param_names=None,
        )

    model = SimpleNamespace(
        A=np.eye(3, dtype=FLOAT),
        B=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=FLOAT),
        compiled=compiled,
        config=config,
        kalman_config=kalman_config,
    )
    model._build_C_d_from_obs = build_measurement
    return model


def _make_shell(model=None, observables=None):
    ki = KalmanInterface.__new__(KalmanInterface)
    ki.model = _make_stub_model() if model is None else model
    ki.observables = ["ObsA", "ObsB"] if observables is None else observables
    return ki


def test_interface_init_reorders_obs_and_builds_state_space():
    model = _make_stub_model()
    y = np.array([[10.0, 1.0], [20.0, 2.0]], dtype=FLOAT)

    ki = KalmanInterface(
        model=model,
        observables=["ObsB", "ObsA"],
        y=y,
        return_shocks=1,
    )

    assert ki.observables == ["ObsA", "ObsB"]
    assert np.array_equal(ki.y, np.array([[1.0, 10.0], [2.0, 20.0]], dtype=FLOAT))
    assert np.array_equal(
        ki.C,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=FLOAT),
    )
    assert np.array_equal(ki.d, np.array([1.0, -1.0], dtype=FLOAT))
    assert np.allclose(
        ki.Q,
        np.array([[0.04, 0.015], [0.015, 0.09]], dtype=FLOAT),
    )
    assert np.array_equal(ki.P0, np.diag([2.0, 6.0, 10.0]).astype(FLOAT))
    assert np.array_equal(
        ki.R,
        np.array([[4.0, 0.6], [0.6, 9.0]], dtype=FLOAT),
    )
    assert ki.jitter == pytest.approx(0.125)
    assert ki.symmetrize is True
    assert ki.return_shocks is True


def test_interface_init_raises_if_reordering_fails(monkeypatch):
    monkeypatch.setattr(
        KalmanInterface,
        "_reorder_obs",
        lambda self, observables, y: (None, np.zeros((1, 1), dtype=FLOAT)),
    )

    with pytest.raises(ValueError, match="Reordering of observables failed"):
        KalmanInterface(
            model=_make_stub_model(),
            observables=["ObsA"],
            y=np.zeros((2, 1), dtype=FLOAT),
        )


def test_get_symmetrize_and_jitter_cover_overrides_and_defaults():
    ki = _make_shell(
        _make_stub_model(
            kalman_config=SimpleNamespace(
                y_names=["ObsA"],
                R=np.array([[1.0]], dtype=FLOAT),
                jitter=None,
                symmetrize=None,
                P0=SimpleNamespace(mode="eye", scale=1.0, diag=None),
                R_builder=None,
                R_param_names=None,
            )
        )
    )

    assert ki._get_symmetrize(True) is True
    assert ki._get_symmetrize(None) is False
    assert ki._get_jitter(0.25) == pytest.approx(0.25)
    assert ki._get_jitter(None) == pytest.approx(0.0)


def test_validate_user_r_and_build_constant_r_subset_paths():
    ki = _make_shell(observables=["ObsB"])

    with pytest.raises(ValueError, match="Provided R matrix has shape"):
        ki._validate_user_R(np.eye(2, dtype=FLOAT))

    user_R = np.array([[2.5]], dtype=FLOAT)
    assert np.array_equal(ki._validate_user_R(user_R), user_R)
    assert np.array_equal(ki._build_constant_R(user_R), user_R)
    assert np.array_equal(ki._build_constant_R(None), np.array([[9.0]], dtype=FLOAT))


def test_build_constant_r_uses_builder_and_validates_builder_contract():
    builder_calls = []

    def builder(meas_a, meas_b):
        builder_calls.append((meas_a, meas_b))
        return np.array([[meas_a, 0.5], [0.5, meas_b]], dtype=FLOAT)

    builder_conf = SimpleNamespace(
        y_names=["ObsA", "ObsB"],
        R=None,
        jitter=0.0,
        symmetrize=False,
        P0=SimpleNamespace(mode="eye", scale=1.0, diag=None),
        R_builder=builder,
        R_param_names=["meas_a", "meas_b"],
    )
    ki = _make_shell(_make_stub_model(kalman_config=builder_conf), observables=["ObsB"])

    assert np.array_equal(ki._build_constant_R(None), np.array([[9.0]], dtype=FLOAT))
    assert builder_calls == [(4.0, 9.0)]

    missing_param_conf = SimpleNamespace(
        y_names=["ObsA"],
        R=None,
        jitter=0.0,
        symmetrize=False,
        P0=SimpleNamespace(mode="eye", scale=1.0, diag=None),
        R_builder=lambda meas_a: np.array([[meas_a]], dtype=FLOAT),
        R_param_names=["missing_param"],
    )
    missing_param_ki = _make_shell(
        _make_stub_model(kalman_config=missing_param_conf),
        observables=["ObsA"],
    )
    with pytest.raises(KeyError, match="Missing R-builder parameter"):
        missing_param_ki._build_constant_R(None)

    bad_shape_conf = SimpleNamespace(
        y_names=["ObsA"],
        R=None,
        jitter=0.0,
        symmetrize=False,
        P0=SimpleNamespace(mode="eye", scale=1.0, diag=None),
        R_builder=lambda meas_a, meas_b: np.array([[meas_a]], dtype=FLOAT),
        R_param_names=["meas_a", "meas_b"],
    )
    bad_shape_ki = _make_shell(
        _make_stub_model(kalman_config=bad_shape_conf),
        observables=["ObsA"],
    )
    with pytest.raises(ValueError, match="R builder returned shape"):
        bad_shape_ki._build_constant_R(None)

    no_r_conf = SimpleNamespace(
        y_names=["ObsA"],
        R=None,
        jitter=0.0,
        symmetrize=False,
        P0=SimpleNamespace(mode="eye", scale=1.0, diag=None),
        R_builder=None,
        R_param_names=None,
    )
    no_r_ki = _make_shell(
        _make_stub_model(kalman_config=no_r_conf), observables=["ObsA"]
    )
    with pytest.raises(ValueError, match="Constant R matrix not specified"):
        no_r_ki._build_constant_R(None)


def test_build_p0_supports_diag_and_eye_and_reports_invalid_configs():
    ki = _make_shell()
    assert np.array_equal(ki._build_P0(), np.diag([2.0, 6.0, 10.0]).astype(FLOAT))
    assert np.array_equal(ki._build_P0(p0_mode="eye", p0_scale=1.5), np.eye(3) * 1.5)

    missing_diag_value = _make_shell(
        _make_stub_model(
            kalman_config=SimpleNamespace(
                y_names=["ObsA"],
                R=np.array([[1.0]], dtype=FLOAT),
                jitter=0.0,
                symmetrize=False,
                P0=SimpleNamespace(mode="diag", scale=1.0, diag={"u": 1.0, "v": 2.0}),
                R_builder=None,
                R_param_names=None,
            )
        )
    )
    with pytest.raises(ValueError, match="must include all model variables"):
        missing_diag_value._build_P0()

    missing_diag_spec = _make_shell(
        _make_stub_model(
            kalman_config=SimpleNamespace(
                y_names=["ObsA"],
                R=np.array([[1.0]], dtype=FLOAT),
                jitter=0.0,
                symmetrize=False,
                P0=SimpleNamespace(mode="diag", scale=1.0, diag=None),
                R_builder=None,
                R_param_names=None,
            )
        )
    )
    with pytest.raises(ValueError, match="P0 diagonal specification missing"):
        missing_diag_spec._build_P0()

    bad_mode = _make_shell(
        _make_stub_model(
            kalman_config=SimpleNamespace(
                y_names=["ObsA"],
                R=np.array([[1.0]], dtype=FLOAT),
                jitter=0.0,
                symmetrize=False,
                P0=SimpleNamespace(mode="triangle", scale=1.0, diag={}),
                R_builder=None,
                R_param_names=None,
            )
        )
    )
    with pytest.raises(ValueError, match="Unrecognized P0 mode"):
        bad_mode._build_P0()

    no_p0 = _make_shell(
        _make_stub_model(
            kalman_config=SimpleNamespace(
                y_names=["ObsA"],
                R=np.array([[1.0]], dtype=FLOAT),
                jitter=0.0,
                symmetrize=False,
                P0=None,
                R_builder=None,
                R_param_names=None,
            )
        )
    )
    with pytest.raises(ValueError, match="Both p0_mode and p0_scale must be provided"):
        no_p0._build_P0()
    with pytest.raises(ValueError, match="must be provided in configuration"):
        no_p0._build_P0(p0_mode="diag", p0_scale=2.0)
    assert np.array_equal(no_p0._build_P0(p0_mode="eye", p0_scale=3.0), np.eye(3) * 3.0)
    with pytest.raises(ValueError, match="Unrecognized p0_mode"):
        no_p0._build_P0(p0_mode="triangle", p0_scale=2.0)


def test_reorder_obs_uses_defaults_and_aligns_dataframe_and_ndarray_inputs():
    ki = _make_shell()
    df = pd.DataFrame({"ObsA": [1.0, 2.0], "ObsB": [10.0, 20.0]})

    obs_df, y_df = ki._reorder_obs(None, df)
    assert obs_df == ["ObsA", "ObsB"]
    assert np.array_equal(y_df, np.array([[1.0, 10.0], [2.0, 20.0]], dtype=FLOAT))

    obs_arr, y_arr = ki._reorder_obs(
        ["ObsB", "ObsA"],
        np.array([[10.0, 1.0], [20.0, 2.0]], dtype=FLOAT),
    )
    assert obs_arr == ["ObsA", "ObsB"]
    assert np.array_equal(y_arr, np.array([[1.0, 10.0], [2.0, 20.0]], dtype=FLOAT))

    canon_default = _make_shell(
        _make_stub_model(
            kalman_config=SimpleNamespace(
                y_names=None,
                R=np.array([[4.0, 0.6], [0.6, 9.0]], dtype=FLOAT),
                jitter=0.0,
                symmetrize=False,
                P0=SimpleNamespace(mode="eye", scale=1.0, diag=None),
                R_builder=None,
                R_param_names=None,
            )
        )
    )
    obs_default, y_default = canon_default._reorder_obs(
        None,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=FLOAT),
    )
    assert obs_default == ["ObsA", "ObsB"]
    assert np.array_equal(y_default, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=FLOAT))


@pytest.mark.parametrize(
    ("observables", "y", "match"),
    [
        ([], np.zeros((2, 0), dtype=FLOAT), "Observable list is empty"),
        (
            ["ObsA", "ObsA"],
            np.zeros((2, 2), dtype=FLOAT),
            "Duplicate observables provided",
        ),
        (["Ghost"], np.zeros((2, 1), dtype=FLOAT), "Unknown observables"),
        (
            ["ObsA"],
            pd.DataFrame({"ObsB": [1.0]}),
            "DataFrame is missing observable columns",
        ),
        (["ObsA"], np.zeros((2,), dtype=FLOAT), "Observation data must be 2D"),
        (
            ["ObsA", "ObsB"],
            np.zeros((2, 1), dtype=FLOAT),
            "y has 1 columns but obs list has 2 names",
        ),
        (
            ["ObsA"],
            np.array([[np.nan]], dtype=FLOAT),
            "Observation data contains NaN values",
        ),
    ],
)
def test_reorder_obs_rejects_invalid_inputs(observables, y, match):
    ki = _make_shell()

    with pytest.raises(ValueError, match=match):
        ki._reorder_obs(observables, y)


def test_validate_mode_and_kalman_config_property_error_paths():
    linear = _make_shell()
    linear.mode = FilterMode.LINEAR
    linear.C = None
    linear.d = np.zeros((1,), dtype=FLOAT)
    linear.h_func = None
    linear.H_jac = None

    with pytest.raises(ValueError, match="C and d matrices are required"):
        linear._validate_mode_and_inputs()

    extended = _make_shell()
    extended.mode = FilterMode.EXTENDED
    extended.C = np.zeros((1, 3), dtype=FLOAT)
    extended.d = np.zeros((1,), dtype=FLOAT)
    extended.h_func = None
    extended.H_jac = lambda *args: np.ones((1, 3), dtype=FLOAT)

    with pytest.raises(ValueError, match="h_func and H_jac are required"):
        extended._validate_mode_and_inputs()

    no_config = _make_shell(_make_stub_model(kalman_config=None))
    with pytest.raises(
        ValueError, match="Kalman Filter configuration with the R matrix is required"
    ):
        _ = no_config.kalman_config


def test_filter_dispatches_linear_run_and_populates_debug_info(monkeypatch):
    ki = KalmanInterface(
        model=_make_stub_model(),
        observables=["ObsA"],
        y=np.array([[1.0], [2.0]], dtype=FLOAT),
        filter_mode="linear",
        jitter=0.5,
        symmetrize=False,
        return_shocks=True,
    )
    captured = {}

    def fake_validate(**kwargs):
        captured["validate"] = kwargs

    def fake_run(**kwargs):
        captured["run"] = kwargs
        return "linear-run"

    monkeypatch.setattr(interface_module, "validate_kf_inputs", fake_validate)
    monkeypatch.setattr(KalmanInterface, "run", staticmethod(fake_run))

    out = ki.filter(
        _debug=True,
        _arg_overrides={"R": np.array([[0.25]], dtype=FLOAT)},
    )

    assert out == "linear-run"
    assert np.array_equal(captured["validate"]["x0"], np.zeros((3,), dtype=FLOAT))
    assert captured["validate"]["filter_mode"] == FilterMode.LINEAR
    assert captured["validate"]["probe_measurement"] is False
    assert np.array_equal(captured["validate"]["R"], np.array([[0.25]], dtype=FLOAT))
    assert captured["run"]["jitter"] == pytest.approx(0.5)
    assert captured["run"]["symmetrize"] is False
    assert captured["run"]["return_shocks"] is True
    assert np.array_equal(captured["run"]["R"], np.array([[0.25]], dtype=FLOAT))
    assert ki._debug_info is not None
    assert np.array_equal(ki._debug_info.x0, np.zeros((3,), dtype=FLOAT))


def test_filter_dispatches_extended_and_rejects_unknown_runtime_mode(monkeypatch):
    ki = KalmanInterface(
        model=_make_stub_model(),
        observables=["ObsA"],
        y=np.array([[1.0], [2.0]], dtype=FLOAT),
        filter_mode="extended",
        h_func=lambda u, v, x, alpha: np.array([x + alpha], dtype=FLOAT),
        H_jac=lambda u, v, x, alpha: np.array([[0.0, 0.0, 1.0]], dtype=FLOAT),
        calib_params=np.array([0.5], dtype=FLOAT),
    )
    captured = {}

    def fake_validate(**kwargs):
        captured["validate"] = kwargs

    def fake_run_extended(**kwargs):
        captured["run_extended"] = kwargs
        return "extended-run"

    monkeypatch.setattr(interface_module, "validate_kf_inputs", fake_validate)
    monkeypatch.setattr(
        KalmanInterface, "run_extended", staticmethod(fake_run_extended)
    )

    out = ki.filter(x0=np.ones((3,), dtype=FLOAT))

    assert out == "extended-run"
    assert captured["validate"]["filter_mode"] == FilterMode.EXTENDED
    assert captured["validate"]["probe_measurement"] is True
    assert np.array_equal(captured["validate"]["x0"], np.ones((3,), dtype=FLOAT))
    assert captured["run_extended"]["h"] is ki.h_func
    assert captured["run_extended"]["H_jac"] is ki.H_jac
    assert np.array_equal(
        captured["run_extended"]["calib_params"],
        np.array([0.5], dtype=FLOAT),
    )

    ki.mode = "mystery"
    with pytest.raises(ValueError, match="Unrecognized filter mode"):
        ki.filter()


def test_ml_estimate_r_diag_success_path(monkeypatch):
    ki = KalmanInterface(
        model=_make_stub_model(),
        observables=["ObsA", "ObsB"],
        y=np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=FLOAT),
    )
    captured = {"filter_calls": []}
    printed = []

    def fake_filter(self, x0=None, _debug=False, _arg_overrides=None):
        captured["filter_calls"].append(
            {
                "x0": x0.copy(),
                "_debug": _debug,
                "R": _arg_overrides["R"].copy(),
            }
        )
        return SimpleNamespace(loglik=-float(np.trace(_arg_overrides["R"])))

    def fake_minimize(obj, x0, bounds, method):
        captured["x0"] = x0.copy()
        captured["bounds"] = bounds
        captured["method"] = method
        captured["objective_at_x0"] = obj(x0)
        return SimpleNamespace(
            success=True,
            x=np.log(np.array([0.3, 0.7], dtype=FLOAT)),
            fun=-4.5,
        )

    monkeypatch.setattr(KalmanInterface, "filter", fake_filter)
    monkeypatch.setattr(interface_module.optimize, "minimize", fake_minimize)
    monkeypatch.setattr(builtins, "print", lambda *args: printed.append(args))

    ki._ML_estimate_R_diag(scale_factor=1.5)

    assert captured["method"] == "L-BFGS-B"
    assert captured["bounds"] == [(-30, 10), (-30, 10)]
    assert captured["objective_at_x0"] > 0.0
    assert len(captured["filter_calls"]) == 1
    assert np.array_equal(
        captured["filter_calls"][0]["x0"],
        np.zeros((3,), dtype=FLOAT),
    )
    assert np.allclose(
        ki.R,
        np.diag([0.45, 1.05]).astype(FLOAT),
    )
    assert printed and "optimization successful" in printed[0][0]


def test_ml_estimate_r_diag_warning_path(monkeypatch):
    diag_R = np.diag([4.0, 9.0]).astype(FLOAT)
    ki = KalmanInterface(
        model=_make_stub_model(
            kalman_config=SimpleNamespace(
                y_names=["ObsA", "ObsB"],
                R=diag_R,
                jitter=0.0,
                symmetrize=False,
                P0=SimpleNamespace(mode="eye", scale=1.0, diag=None),
                R_builder=None,
                R_param_names=None,
            )
        ),
        observables=["ObsA", "ObsB"],
        y=np.array([[1.0, 10.0], [2.0, 20.0]], dtype=FLOAT),
    )
    printed = []

    def fake_filter(self, x0=None, _debug=False, _arg_overrides=None):
        return SimpleNamespace(loglik=-float(np.trace(_arg_overrides["R"])))

    def fake_minimize(obj, x0, bounds, method):
        obj(x0)
        return SimpleNamespace(
            success=False,
            x=np.log(np.array([4.0, 9.0], dtype=FLOAT)),
            fun=-1.0,
            message="stalled",
        )

    monkeypatch.setattr(KalmanInterface, "filter", fake_filter)
    monkeypatch.setattr(interface_module.optimize, "minimize", fake_minimize)
    monkeypatch.setattr(builtins, "print", lambda *args: printed.append(args))

    with pytest.warns(UserWarning, match="did not converge"):
        ki._ML_estimate_R_diag(scale_factor=2.0)

    assert np.allclose(ki.R, np.diag([8.0, 18.0]).astype(FLOAT))
    assert len(printed) == 2
    assert "Using Config R matrix" in printed[0][0]
    assert "optimization successful" in printed[1][0]
