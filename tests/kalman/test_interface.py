# type: ignore
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sympy import Symbol

import SymbolicDSGE.kalman.interface as interface_module
from SymbolicDSGE.kalman.filter import FilterRawResult, UnscentedFilterRawResult
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


def _raw_filter_result(T: int = 2, n: int = 3, m: int = 1) -> FilterRawResult:
    x = np.zeros((T, n), dtype=FLOAT)
    y = np.zeros((T, m), dtype=FLOAT)
    P = np.zeros((T, n, n), dtype=FLOAT)
    S = np.zeros((T, m, m), dtype=FLOAT)
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
        loglik=FLOAT(0.0),
    )


def _raw_unscented_result(
    T: int = 2,
    n_state: int = 2,
    n_var: int = 3,
) -> UnscentedFilterRawResult:
    x = np.zeros((T, n_var), dtype=FLOAT)
    xb = np.zeros((T, n_state), dtype=FLOAT)
    y = np.zeros((T, 1), dtype=FLOAT)
    P = np.zeros((T, 2 * n_state, 2 * n_state), dtype=FLOAT)
    S = np.zeros((T, 1, 1), dtype=FLOAT)
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
        loglik=FLOAT(0.0),
    )


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
        n_state=2,
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
            R_std_param_map=None,
            R_corr_param_map=None,
        )

    model = SimpleNamespace(
        A=np.eye(3, dtype=FLOAT),
        B=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=FLOAT),
        compiled=compiled,
        config=config,
        kalman_config=kalman_config,
        policy=SimpleNamespace(
            order=2,
            p=np.array([[0.8, 0.1], [0.0, 0.7]], dtype=FLOAT),
            f=np.array([[0.2, 0.3]], dtype=FLOAT),
            hxx=np.zeros((2, 2, 2), dtype=FLOAT),
            gxx=np.zeros((1, 2, 2), dtype=FLOAT),
            hss=np.array([0.01, 0.02], dtype=FLOAT),
            gss=np.array([0.03], dtype=FLOAT),
            steady_state=np.array([1.0, 2.0, 3.0], dtype=FLOAT),
        ),
    )
    model._build_C_d_from_obs = build_measurement

    # Mirror SolvedModel's Kalman-matrix cache contract so the interface's
    # cache-aware construction path works against the stub.
    _cache: dict = {}

    def _calibration_fingerprint():
        p = calibration.parameters
        return hash((tuple(p.keys()), tuple(float(v) for v in p.values())))

    model._calibration_fingerprint = _calibration_fingerprint
    model._kf_cache_get = _cache.get
    model._kf_cache_put = _cache.__setitem__
    return model


def _make_shell(model=None, observables=None):
    ki = KalmanInterface.__new__(KalmanInterface)
    ki.model = _make_stub_model() if model is None else model
    ki.observables = ["ObsA", "ObsB"] if observables is None else observables
    ki.mode = FilterMode.LINEAR
    return ki


def test_interface_init_reorders_obs_and_builds_state_space():
    model = _make_stub_model()
    y = np.array([[10.0, 1.0], [20.0, 2.0]], dtype=FLOAT)

    ki = KalmanInterface(
        model=model,
        observables=["ObsB", "ObsA"],
        y=y,
        return_shocks=1,
        jitter=0.125,
        symmetrize=True,
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


def test_interface_init_extended_skips_linear_measurement_builder():
    model = _make_stub_model()

    def bomb(obs_names):
        raise AssertionError(
            "_build_C_d_from_obs should not be called in extended mode"
        )

    model._build_C_d_from_obs = bomb

    ki = KalmanInterface(
        model=model,
        observables=["ObsA"],
        y=np.array([[1.0], [2.0]], dtype=FLOAT),
        filter_mode="extended",
        meas_addr=1,
        jac_addr=1,
        calib_params=np.array([0.5], dtype=FLOAT),
    )

    assert ki.C is None
    assert ki.d is None


def test_interface_init_accepts_user_r_override():
    user_R = np.array([[0.75]], dtype=FLOAT)
    ki = KalmanInterface(
        model=_make_stub_model(),
        observables=["ObsB"],
        y=np.array([[10.0], [20.0]], dtype=FLOAT),
        R=user_R,
    )

    assert np.array_equal(ki.R, user_R)


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
                R_std_param_map=None,
                R_corr_param_map=None,
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


def test_build_q_defaults_missing_shock_correlation_to_zero():
    model = _make_stub_model()
    model.config.calibration.shock_corr = {}
    ki = _make_shell(model)

    np.testing.assert_allclose(
        ki._build_Q(),
        np.diag([0.04, 0.09]).astype(FLOAT),
    )


def test_build_constant_r_assembles_from_param_maps_and_current_calibration():
    # Named R: the interface assembles the constant R from the CURRENT calibration
    # via the std/corr param maps (make_R), then subsets to included observables.
    # make_R treats the std params as standard deviations, so the diagonal is
    # sig**2 and the off-diagonal is sig_i * sig_j * corr.
    named_conf = SimpleNamespace(
        y_names=["ObsA", "ObsB"],
        R=None,
        jitter=0.0,
        symmetrize=False,
        P0=SimpleNamespace(mode="eye", scale=1.0, diag=None),
        R_std_param_map={"ObsA": "meas_a", "ObsB": "meas_b"},
        R_corr_param_map={frozenset({"ObsA", "ObsB"}): "meas_rho"},
    )
    model = _make_stub_model(
        kalman_config=named_conf,
        params={Symbol("meas_rho"): 0.1},
    )
    # std meas_a=4, meas_b=9, corr meas_rho=0.1 ->
    #   [[16, 3.6], [3.6, 81]]; subset to [ObsB] -> [[81]].
    ki_full = _make_shell(model, observables=["ObsA", "ObsB"])
    assert np.allclose(
        ki_full._build_constant_R(None),
        np.array([[16.0, 3.6], [3.6, 81.0]], dtype=FLOAT),
    )
    ki_sub = _make_shell(model, observables=["ObsB"])
    assert np.allclose(ki_sub._build_constant_R(None), np.array([[81.0]], dtype=FLOAT))

    # A std/corr param that is absent from calibration is a hard error.
    missing_param_conf = SimpleNamespace(
        y_names=["ObsA", "ObsB"],
        R=None,
        jitter=0.0,
        symmetrize=False,
        P0=SimpleNamespace(mode="eye", scale=1.0, diag=None),
        R_std_param_map={"ObsA": "not_calibrated", "ObsB": "meas_b"},
        R_corr_param_map={},
    )
    missing_param_ki = _make_shell(
        _make_stub_model(kalman_config=missing_param_conf),
        observables=["ObsA"],
    )
    with pytest.raises(KeyError, match="Missing R parameter"):
        missing_param_ki._build_constant_R(None)

    # No param maps and no static R -> nothing to build from.
    no_r_conf = SimpleNamespace(
        y_names=["ObsA"],
        R=None,
        jitter=0.0,
        symmetrize=False,
        P0=SimpleNamespace(mode="eye", scale=1.0, diag=None),
        R_std_param_map=None,
        R_corr_param_map=None,
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
                R_std_param_map=None,
                R_corr_param_map=None,
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
                R_std_param_map=None,
                R_corr_param_map=None,
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
                R_std_param_map=None,
                R_corr_param_map=None,
            )
        )
    )
    with pytest.raises(ValueError, match="Unrecognized P0 mode"):
        bad_mode._build_P0()


def test_build_unscented_p0_uses_state_block_and_identity_second_block():
    ki = _make_shell()
    ki.mode = FilterMode.UNSCENTED

    expected = np.diag([2.0, 6.0, 1.0, 1.0]).astype(FLOAT)
    assert np.array_equal(ki._build_P0(), expected)

    expected_eye = np.diag([1.5, 1.5, 1.0, 1.0]).astype(FLOAT)
    assert np.array_equal(
        ki._build_P0(p0_mode="eye", p0_scale=1.5),
        expected_eye,
    )

    state_only_diag = _make_shell(
        _make_stub_model(
            kalman_config=SimpleNamespace(
                y_names=["ObsA"],
                R=np.array([[1.0]], dtype=FLOAT),
                jitter=0.0,
                symmetrize=False,
                P0=SimpleNamespace(mode="diag", scale=2.0, diag={"u": 1.0, "v": 2.0}),
                R_std_param_map=None,
                R_corr_param_map=None,
            )
        )
    )
    state_only_diag.mode = FilterMode.UNSCENTED
    assert np.array_equal(
        state_only_diag._build_P0(),
        np.diag([2.0, 4.0, 1.0, 1.0]).astype(FLOAT),
    )

    missing_state_diag = _make_shell(
        _make_stub_model(
            kalman_config=SimpleNamespace(
                y_names=["ObsA"],
                R=np.array([[1.0]], dtype=FLOAT),
                jitter=0.0,
                symmetrize=False,
                P0=SimpleNamespace(mode="diag", scale=1.0, diag={"u": 1.0, "x": 5.0}),
                R_std_param_map=None,
                R_corr_param_map=None,
            )
        )
    )
    missing_state_diag.mode = FilterMode.UNSCENTED
    with pytest.raises(ValueError, match="must include all state variables"):
        missing_state_diag._build_P0()

    eye_override = _make_shell(
        _make_stub_model(
            kalman_config=SimpleNamespace(
                y_names=["ObsA"],
                R=np.array([[1.0]], dtype=FLOAT),
                jitter=0.0,
                symmetrize=False,
                P0=SimpleNamespace(mode="eye", scale=1.0, diag=None),
                R_std_param_map=None,
                R_corr_param_map=None,
            )
        )
    )
    eye_override.mode = FilterMode.UNSCENTED
    assert np.array_equal(
        eye_override._build_P0(p0_mode="eye", p0_scale=3.0),
        np.diag([3.0, 3.0, 1.0, 1.0]).astype(FLOAT),
    )


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
                R_std_param_map=None,
                R_corr_param_map=None,
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

    with pytest.raises(ValueError, match="C and d matrices are required"):
        linear._validate_mode_and_inputs()

    extended = _make_shell()
    extended.mode = FilterMode.EXTENDED
    extended.meas_addr = None
    extended.jac_addr = 1

    with pytest.raises(ValueError, match="meas_addr and jac_addr are required"):
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

    def fake_run_raw(**kwargs):
        captured["run_raw"] = kwargs
        return _raw_filter_result()

    monkeypatch.setattr(interface_module, "validate_kf_inputs", fake_validate)
    monkeypatch.setattr(KalmanInterface, "run_raw", staticmethod(fake_run_raw))

    out = ki.filter(
        _debug=True,
        _arg_overrides={"R": np.array([[0.25]], dtype=FLOAT)},
    )

    np.testing.assert_allclose(out.x_pred, np.zeros((2, 3), dtype=FLOAT))
    assert np.array_equal(captured["validate"]["x0"], np.zeros((3,), dtype=FLOAT))
    assert captured["validate"]["filter_mode"] == FilterMode.LINEAR
    assert captured["validate"]["probe_measurement"] is False
    assert np.array_equal(captured["validate"]["R"], np.array([[0.25]], dtype=FLOAT))
    assert captured["run_raw"]["jitter"] == pytest.approx(0.5)
    assert captured["run_raw"]["symmetrize"] is False
    assert captured["run_raw"]["return_shocks"] is True
    assert np.array_equal(captured["run_raw"]["R"], np.array([[0.25]], dtype=FLOAT))
    assert ki._debug_info is not None
    assert np.array_equal(ki._debug_info.x0, np.zeros((3,), dtype=FLOAT))


def test_filter_raw_dispatches_linear_run_raw(monkeypatch):
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

    def fake_run_raw(**kwargs):
        captured["run_raw"] = kwargs
        return _raw_filter_result()

    monkeypatch.setattr(interface_module, "validate_kf_inputs", fake_validate)
    monkeypatch.setattr(KalmanInterface, "run_raw", staticmethod(fake_run_raw))

    out = ki.filter_raw(
        _debug=True,
        _arg_overrides={"R": np.array([[0.25]], dtype=FLOAT)},
    )

    assert isinstance(out, FilterRawResult)
    assert np.array_equal(captured["validate"]["x0"], np.zeros((3,), dtype=FLOAT))
    assert captured["validate"]["filter_mode"] == FilterMode.LINEAR
    assert captured["validate"]["probe_measurement"] is False
    assert captured["run_raw"]["jitter"] == pytest.approx(0.5)
    assert captured["run_raw"]["symmetrize"] is False
    assert captured["run_raw"]["return_shocks"] is True
    assert np.array_equal(captured["run_raw"]["R"], np.array([[0.25]], dtype=FLOAT))
    assert ki._debug_info is not None
    assert np.array_equal(ki._debug_info.x0, np.zeros((3,), dtype=FLOAT))


def test_filter_dispatches_extended_and_rejects_unknown_runtime_mode(monkeypatch):
    ki = KalmanInterface(
        model=_make_stub_model(),
        observables=["ObsA"],
        y=np.array([[1.0], [2.0]], dtype=FLOAT),
        filter_mode="extended",
        meas_addr=111,
        jac_addr=222,
        calib_params=np.array([0.5], dtype=FLOAT),
    )
    captured = {}

    def fake_validate(**kwargs):
        captured["validate"] = kwargs

    def fake_run_extended_raw(**kwargs):
        captured["run_extended_raw"] = kwargs
        return _raw_filter_result()

    monkeypatch.setattr(interface_module, "validate_kf_inputs", fake_validate)
    monkeypatch.setattr(
        KalmanInterface, "run_extended_raw", staticmethod(fake_run_extended_raw)
    )

    out = ki.filter(x0=np.ones((3,), dtype=FLOAT))

    np.testing.assert_allclose(out.x_pred, np.zeros((2, 3), dtype=FLOAT))
    assert captured["validate"]["filter_mode"] == FilterMode.EXTENDED
    assert captured["validate"]["probe_measurement"] is True
    assert np.array_equal(captured["validate"]["x0"], np.ones((3,), dtype=FLOAT))
    assert "C" not in captured["validate"]
    assert "d" not in captured["validate"]
    assert captured["run_extended_raw"]["meas_addr"] == ki.meas_addr
    assert captured["run_extended_raw"]["jac_addr"] == ki.jac_addr
    assert np.array_equal(
        captured["run_extended_raw"]["calib_params"],
        np.array([0.5], dtype=FLOAT),
    )

    ki.mode = "mystery"
    with pytest.raises(ValueError, match="Unrecognized filter mode"):
        ki.filter()


def test_filter_dispatches_unscented_and_populates_debug_info(monkeypatch):
    ki = KalmanInterface(
        model=_make_stub_model(),
        observables=["ObsA"],
        y=np.array([[1.0], [2.0]], dtype=FLOAT),
        filter_mode="unscented",
        meas_addr=123,
        calib_params=np.array([0.5], dtype=FLOAT),
        jitter=0.25,
        symmetrize=False,
    )
    captured = {}

    def fake_run_unscented_raw(**kwargs):
        captured["run_unscented_raw"] = kwargs
        return _raw_unscented_result()

    monkeypatch.setattr(
        KalmanInterface, "run_unscented_raw", staticmethod(fake_run_unscented_raw)
    )

    x0 = np.array([0.2, 0.3, 99.0], dtype=FLOAT)
    out = ki.filter(x0=x0, _debug=True)

    np.testing.assert_allclose(out.x_pred, np.zeros((2, 3), dtype=FLOAT))
    run_args = captured["run_unscented_raw"]
    assert run_args["meas_addr"] == 123
    assert np.array_equal(run_args["z0"], np.array([0.2, 0.3, 0.0, 0.0]))
    assert np.array_equal(run_args["bx"], np.eye(2, dtype=FLOAT))
    assert np.array_equal(run_args["steady_state"], np.array([1.0, 2.0, 3.0]))
    assert run_args["alpha"] == pytest.approx(1.0)
    assert run_args["beta"] == pytest.approx(2.0)
    assert run_args["kappa"] == pytest.approx(1.0)
    assert run_args["jitter"] == pytest.approx(0.25)
    assert run_args["symmetrize"] is False
    assert ki._debug_info is not None
    assert ki._debug_info.meas_addr == 123
    assert np.array_equal(ki._debug_info.z0, np.array([0.2, 0.3, 0.0, 0.0]))
    assert np.array_equal(ki._debug_info.hx, ki.model.policy.p)
    assert np.array_equal(ki._debug_info.gx, ki.model.policy.f)
    assert ki._debug_info.alpha == pytest.approx(1.0)


def test_filter_unscented_rejects_return_shocks_and_bad_x0():
    ki = KalmanInterface(
        model=_make_stub_model(),
        observables=["ObsA"],
        y=np.array([[1.0], [2.0]], dtype=FLOAT),
        filter_mode="unscented",
        meas_addr=123,
        calib_params=np.array([0.5], dtype=FLOAT),
        return_shocks=True,
    )
    with pytest.raises(ValueError, match="return_shocks is not supported"):
        ki.filter()

    ki.return_shocks = False
    with pytest.raises(ValueError, match="x0 must have length"):
        ki.filter(x0=np.array([1.0], dtype=FLOAT))


def test_filter_uses_current_self_r_after_validated_args_access(monkeypatch):
    ki = KalmanInterface(
        model=_make_stub_model(),
        observables=["ObsA", "ObsB"],
        y=np.array([[1.0, 10.0], [2.0, 20.0]], dtype=FLOAT),
    )
    _ = ki._linear_validated_args
    ki.R = np.diag([0.3, 0.7]).astype(FLOAT)

    captured = {}

    def fake_run_raw(*, R, **kwargs):
        captured["R"] = R.copy()
        return _raw_filter_result()

    monkeypatch.setattr(KalmanInterface, "run_raw", staticmethod(fake_run_raw))

    ki.filter(x0=np.zeros((3,), dtype=FLOAT))

    assert np.array_equal(captured["R"], np.diag([0.3, 0.7]).astype(FLOAT))
