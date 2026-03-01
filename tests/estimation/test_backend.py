# type: ignore
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sympy import Symbol

from SymbolicDSGE import ModelParser, DSGESolver
from SymbolicDSGE.estimation import backend
from SymbolicDSGE.kalman.config import KalmanConfig, P0Config


class _ConstPrior:
    def __init__(self, value: float):
        self.value = float(value)

    def logpdf(self, x):
        return self.value


@pytest.fixture(scope="module")
def post82_bundle():
    model, kalman = ModelParser("MODELS/POST82.yaml").get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile(n_state=3, n_exog=3)

    steady = np.zeros((len(compiled.var_names),), dtype=np.float64)
    solved = solver.solve(compiled=compiled, steady_state=steady, log_linear=False)

    params = model.calibration.parameters
    std_map = model.calibration.shock_std
    sig_g = float(params[std_map[Symbol("e_g")]])
    sig_z = float(params[std_map[Symbol("e_z")]])
    sig_r = float(params[std_map[Symbol("e_r")]])

    T = 24
    rng = np.random.default_rng(20260303)
    sim = solved.sim(
        T=T,
        shocks={
            "g": rng.normal(0.0, sig_g, size=T),
            "z": rng.normal(0.0, sig_z, size=T),
            "r": rng.normal(0.0, sig_r, size=T),
        },
        x0=np.zeros((len(compiled.var_names),), dtype=np.float64),
        observables=True,
    )
    y = pd.DataFrame(
        {
            "Infl": sim["Infl"][1:],
            "Rate": sim["Rate"][1:],
        }
    )
    return {
        "model": model,
        "solver": solver,
        "compiled": compiled,
        "solved": solved,
        "steady": steady,
        "y": y,
    }


def test_name_extract_and_builders_basic():
    a = Symbol("a")
    b = Symbol("b")
    compiled = SimpleNamespace(
        config=SimpleNamespace(
            calibration=SimpleNamespace(parameters={a: 1.5, b: 2.5})
        ),
        calib_params=[b, a],
    )

    assert backend._name_of(a) == "a"
    assert backend._name_of("x") == "x"

    base = backend.extract_base_params(compiled)
    assert base == {"a": pytest.approx(1.5), "b": pytest.approx(2.5)}

    full = backend.build_full_params(base, ["a"], np.array([3.0], dtype=np.float64))
    assert full["a"] == pytest.approx(3.0)
    assert full["b"] == pytest.approx(2.5)

    with pytest.raises(ValueError, match="1D"):
        backend.build_full_params(base, ["a"], np.array([[1.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="does not match"):
        backend.build_full_params(base, ["a", "b"], np.array([1.0], dtype=np.float64))

    vec = backend.build_calib_param_vector(compiled, {"a": 10.0, "b": 20.0})
    assert np.allclose(vec, np.array([20.0, 10.0], dtype=np.float64))


def test_reorder_observables_dataframe_and_ndarray_paths():
    compiled = SimpleNamespace(observable_names=["Infl", "Rate"])
    kalman = SimpleNamespace(y_names=["Rate", "Infl"])

    df = pd.DataFrame({"Rate": [1.0, 2.0], "Infl": [3.0, 4.0]})
    obs, y_df = backend.reorder_observables(compiled, kalman, None, df)
    assert obs == ["Infl", "Rate"]
    assert np.allclose(y_df, np.array([[3.0, 1.0], [4.0, 2.0]], dtype=np.float64))

    arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)  # [Rate, Infl]
    obs_arr, y_arr = backend.reorder_observables(
        compiled, kalman, ["Rate", "Infl"], arr
    )
    assert obs_arr == ["Infl", "Rate"]
    assert np.allclose(y_arr, np.array([[20.0, 10.0], [40.0, 30.0]], dtype=np.float64))


def test_infer_filter_mode_affine_vs_non_affine():
    compiled_aff = SimpleNamespace(
        kalman=None,
        observable_names=["Infl", "Rate"],
        config=SimpleNamespace(
            equations=SimpleNamespace(obs_is_affine={"Infl": True, "Rate": True})
        ),
    )
    compiled_mix = SimpleNamespace(
        kalman=None,
        observable_names=["Infl", "Rate"],
        config=SimpleNamespace(
            equations=SimpleNamespace(obs_is_affine={"Infl": True, "Rate": False})
        ),
    )

    assert backend.infer_filter_mode(compiled_aff, ["Infl", "Rate"]) == "linear"
    assert backend.infer_filter_mode(compiled_mix, ["Infl", "Rate"]) == "extended"


@pytest.mark.parametrize(
    "observables,y,match",
    [
        ([], np.zeros((2, 2), dtype=np.float64), "empty"),
        (["Infl", "Infl"], np.zeros((2, 2), dtype=np.float64), "duplicates"),
        (["Bad"], np.zeros((2, 1), dtype=np.float64), "Unknown observables"),
        (["Infl", "Rate"], np.array([1.0, 2.0], dtype=np.float64), "must be 2D"),
        (["Infl", "Rate"], np.zeros((2, 1), dtype=np.float64), "columns"),
        (["Infl", "Rate"], np.array([[1.0, np.nan]], dtype=np.float64), "contains NaN"),
    ],
)
def test_reorder_observables_errors(observables, y, match):
    compiled = SimpleNamespace(observable_names=["Infl", "Rate"])
    with pytest.raises(ValueError, match=match):
        backend.reorder_observables(compiled, None, observables, y)


def test_build_Q_matches_post82_manual_structure(post82_bundle):
    compiled = post82_bundle["compiled"]
    params = backend.extract_base_params(compiled)
    Q = backend.build_Q(compiled, params)

    sig_g = params["sig_g"]
    sig_z = params["sig_z"]
    sig_r = params["sig_r"]
    rho_gz = params["rho_gz"]
    expected = np.array(
        [
            [sig_g**2, sig_g * sig_z * rho_gz, 0.0],
            [sig_g * sig_z * rho_gz, sig_z**2, 0.0],
            [0.0, 0.0, sig_r**2],
        ],
        dtype=np.float64,
    )
    assert np.allclose(Q, expected)


def test_build_C_d_matches_solved_model_helper(post82_bundle):
    compiled = post82_bundle["compiled"]
    solved = post82_bundle["solved"]
    params = backend.extract_base_params(compiled)

    C1, d1 = backend.build_C_d(compiled, params, ["Infl", "Rate"])
    C2, d2 = solved._build_C_d_from_obs(["Infl", "Rate"])
    assert np.allclose(C1, C2)
    assert np.allclose(d1, d2)


def test_build_P0_branches():
    compiled = SimpleNamespace(var_names=["g", "z"])

    assert backend.build_P0(compiled, None, None, None) is None

    eye = backend.build_P0(compiled, None, "eye", 2.0)
    assert np.allclose(eye, 2.0 * np.eye(2))

    kalman = KalmanConfig(
        y_names=["Infl"],
        R=np.eye(1, dtype=np.float64),
        jitter=0.0,
        symmetrize=False,
        P0=P0Config(mode="diag", scale=3.0, diag={"g": 1.0, "z": 2.0}),
    )
    p0 = backend.build_P0(compiled, kalman, None, None)
    assert np.allclose(p0, np.diag([3.0, 6.0]))

    bad_kalman = KalmanConfig(
        y_names=["Infl"],
        R=np.eye(1, dtype=np.float64),
        jitter=0.0,
        symmetrize=False,
        P0=P0Config(mode="diag", scale=1.0, diag={"g": 1.0}),
    )
    with pytest.raises(ValueError, match="Missing P0 diagonal entry"):
        backend.build_P0(compiled, bad_kalman, None, None)

    with pytest.raises(ValueError, match="requires diagonal entries"):
        backend.build_P0(compiled, None, "diag", 1.0)
    with pytest.raises(ValueError, match="Unrecognized p0_mode"):
        backend.build_P0(compiled, None, "unknown", 1.0)


def test_resolve_R_branches():
    compiled = SimpleNamespace(observable_names=["Infl", "Rate", "Out"])

    with pytest.raises(ValueError, match="R is not available"):
        backend.resolve_R(
            compiled=compiled,
            kalman=None,
            observables=["Infl", "Rate"],
            R=None,
        )

    with pytest.raises(ValueError, match="Provided R has shape"):
        backend.resolve_R(
            compiled=compiled,
            kalman=None,
            observables=["Infl", "Rate"],
            R=np.eye(3, dtype=np.float64),
        )

    R_ok = np.array([[1.0, 0.1], [0.1, 2.0]], dtype=np.float64)
    out_direct = backend.resolve_R(
        compiled=compiled,
        kalman=None,
        observables=["Infl", "Rate"],
        R=R_ok,
    )
    assert np.allclose(out_direct, R_ok)

    kalman = KalmanConfig(
        y_names=["Infl", "Rate", "Out"],
        R=np.array(
            [[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]], dtype=np.float64
        ),
        jitter=0.0,
        symmetrize=False,
        P0=P0Config(mode="eye", scale=1.0, diag=None),
    )
    subset = backend.resolve_R(
        compiled=compiled,
        kalman=kalman,
        observables=["Rate", "Infl"],
        R=None,
    )
    assert np.allclose(subset, np.array([[5.0, 2.0], [2.0, 1.0]], dtype=np.float64))


def test_kalman_symbolic_R_builder_matches_numeric_config_R(post82_bundle):
    compiled = post82_bundle["compiled"]
    kalman = compiled.kalman
    assert kalman is not None
    assert kalman.R_symbolic is not None
    assert kalman.R_builder is not None
    assert kalman.R_param_names == ["meas_infl", "meas_rate", "meas_rho_ir"]

    params = backend.extract_base_params(compiled)
    R_from_builder = backend.build_R_from_config_params(
        compiled=compiled,
        kalman=kalman,
        observables=["Infl", "Rate"],
        params=params,
    )
    R_resolved = backend.resolve_R(
        compiled=compiled,
        kalman=kalman,
        observables=["Infl", "Rate"],
        R=None,
    )
    assert np.allclose(R_from_builder, R_resolved)


def test_build_R_from_config_params_raises_on_missing_param(post82_bundle):
    compiled = post82_bundle["compiled"]
    kalman = compiled.kalman
    assert kalman is not None
    params = backend.extract_base_params(compiled)
    params.pop("meas_rho_ir")

    with pytest.raises(KeyError, match="Missing R-builder parameter"):
        backend.build_R_from_config_params(
            compiled=compiled,
            kalman=kalman,
            observables=["Infl", "Rate"],
            params=params,
        )


def test_evaluate_loglik_linear_and_extended_match_model_kalman(post82_bundle):
    solver = post82_bundle["solver"]
    compiled = post82_bundle["compiled"]
    solved = post82_bundle["solved"]
    steady = post82_bundle["steady"]
    y = post82_bundle["y"]
    params = backend.extract_base_params(compiled)

    ll_backend_lin = backend.evaluate_loglik(
        solver=solver,
        compiled=compiled,
        y=y,
        params=params,
        filter_mode="linear",
        observables=["Infl", "Rate"],
        steady_state=steady,
        log_linear=False,
        x0=None,
        p0_mode=None,
        p0_scale=None,
        jitter=None,
        symmetrize=None,
        R=None,
    )
    ll_model_lin = solved.kalman(
        y=y,
        filter_mode="linear",
        observables=["Infl", "Rate"],
    ).loglik
    assert np.allclose(ll_backend_lin, ll_model_lin, rtol=1e-10, atol=1e-10)

    ll_backend_ext = backend.evaluate_loglik(
        solver=solver,
        compiled=compiled,
        y=y,
        params=params,
        filter_mode="extended",
        observables=["Infl", "Rate"],
        steady_state=steady,
        log_linear=False,
        x0=None,
        p0_mode=None,
        p0_scale=None,
        jitter=None,
        symmetrize=None,
        R=None,
    )
    ll_model_ext = solved.kalman(
        y=y,
        filter_mode="extended",
        observables=["Infl", "Rate"],
    ).loglik
    assert np.allclose(ll_backend_ext, ll_model_ext, rtol=1e-10, atol=1e-10)


def test_evaluate_loglik_respects_R_override_and_mode_validation(post82_bundle):
    solver = post82_bundle["solver"]
    compiled = post82_bundle["compiled"]
    steady = post82_bundle["steady"]
    y = post82_bundle["y"]
    params = backend.extract_base_params(compiled)

    with pytest.raises(ValueError, match="Unrecognized filter_mode"):
        backend.evaluate_loglik(
            solver=solver,
            compiled=compiled,
            y=y,
            params=params,
            filter_mode="bad_mode",
            observables=["Infl", "Rate"],
            steady_state=steady,
            log_linear=False,
            x0=None,
            p0_mode=None,
            p0_scale=None,
            jitter=None,
            symmetrize=None,
            R=None,
        )

    base_R = compiled.kalman.R[:2, :2]
    scaled_R = 2.0 * base_R
    ll_direct = backend.evaluate_loglik(
        solver=solver,
        compiled=compiled,
        y=y,
        params=params,
        filter_mode="linear",
        observables=["Infl", "Rate"],
        steady_state=steady,
        log_linear=False,
        x0=None,
        p0_mode=None,
        p0_scale=None,
        jitter=None,
        symmetrize=None,
        R=scaled_R,
    )
    ll_config = backend.evaluate_loglik(
        solver=solver,
        compiled=compiled,
        y=y,
        params=params,
        filter_mode="linear",
        observables=["Infl", "Rate"],
        steady_state=steady,
        log_linear=False,
        x0=None,
        p0_mode=None,
        p0_scale=None,
        jitter=None,
        symmetrize=None,
        R=None,
    )
    assert not np.isclose(ll_direct, ll_config)


def test_estimate_R_diag_returns_positive_diagonal(post82_bundle):
    solver = post82_bundle["solver"]
    compiled = post82_bundle["compiled"]
    steady = post82_bundle["steady"]
    y = post82_bundle["y"]
    params = backend.extract_base_params(compiled)

    R = backend.estimate_R_diag(
        solver=solver,
        compiled=compiled,
        y=y,
        params=params,
        observables=["Infl", "Rate"],
        steady_state=steady,
        log_linear=False,
        x0=None,
        p0_mode=None,
        p0_scale=None,
        jitter=None,
        symmetrize=None,
    )
    assert R.shape == (2, 2)
    assert np.all(np.diag(R) > 0.0)


def test_estimate_R_tries_map_then_falls_back_to_mle(post82_bundle, monkeypatch):
    solver = post82_bundle["solver"]
    compiled = post82_bundle["compiled"]
    steady = post82_bundle["steady"]
    y = post82_bundle["y"]
    params = backend.extract_base_params(compiled)

    calls = []

    def fake_minimize(fun, x0, bounds=None, method=None):
        calls.append(fun.__name__)
        if len(calls) == 1:
            return SimpleNamespace(success=False, x=np.asarray(x0, dtype=np.float64))
        return SimpleNamespace(success=True, x=np.asarray(x0, dtype=np.float64))

    monkeypatch.setattr(backend.optimize, "minimize", fake_minimize)

    R = backend.estimate_R(
        solver=solver,
        compiled=compiled,
        y=y,
        params=params,
        observables=["Infl", "Rate"],
        steady_state=steady,
        log_linear=False,
        x0=None,
        p0_mode=None,
        p0_scale=None,
        jitter=None,
        symmetrize=None,
    )
    assert calls == ["nlogpost", "nloglik"]
    assert R.shape == (2, 2)
    assert np.all(np.diag(R) > 0.0)


def test_estimate_R_stops_after_successful_map(post82_bundle, monkeypatch):
    solver = post82_bundle["solver"]
    compiled = post82_bundle["compiled"]
    steady = post82_bundle["steady"]
    y = post82_bundle["y"]
    params = backend.extract_base_params(compiled)

    calls = []

    def fake_minimize(fun, x0, bounds=None, method=None):
        calls.append(fun.__name__)
        return SimpleNamespace(success=True, x=np.asarray(x0, dtype=np.float64))

    monkeypatch.setattr(backend.optimize, "minimize", fake_minimize)

    _ = backend.estimate_R(
        solver=solver,
        compiled=compiled,
        y=y,
        params=params,
        observables=["Infl", "Rate"],
        steady_state=steady,
        log_linear=False,
        x0=None,
        p0_mode=None,
        p0_scale=None,
        jitter=None,
        symmetrize=None,
    )
    assert calls == ["nlogpost"]


def test_estimate_R_falls_back_to_diag_when_solve_raises(post82_bundle, monkeypatch):
    solver = post82_bundle["solver"]
    compiled = post82_bundle["compiled"]
    steady = post82_bundle["steady"]
    y = post82_bundle["y"]
    params = backend.extract_base_params(compiled)

    def _boom(*args, **kwargs):
        raise SystemExit("invertibility violation")

    monkeypatch.setattr(solver, "solve", _boom)
    R = backend.estimate_R(
        solver=solver,
        compiled=compiled,
        y=y,
        params=params,
        observables=["Infl", "Rate"],
        steady_state=steady,
        log_linear=False,
        x0=None,
        p0_mode=None,
        p0_scale=None,
        jitter=None,
        symmetrize=None,
    )
    assert R.shape == (2, 2)
    assert np.all(np.diag(R) > 0.0)
    assert np.allclose(R, np.diag(np.diag(R)))


def test_corr_chol_and_R_unconstrained_parameterization():
    z = np.array([0.2, -0.1, 0.3], dtype=np.float64)
    L = backend._corr_chol_from_unconstrained(z, K=3)
    corr = L @ L.T
    assert np.allclose(np.diag(corr), np.ones(3), atol=1e-10)
    assert np.all(np.linalg.eigvalsh(corr) > 0.0)

    u = np.array([np.log(0.4), np.log(0.8), 0.1], dtype=np.float64)
    R, std, _ = backend._R_from_unconstrained(u, K=2)
    assert R.shape == (2, 2)
    assert np.allclose(np.sqrt(np.diag(R)), std)
    assert np.all(np.linalg.eigvalsh(R) > 0.0)


def test_evaluate_logprior_branches():
    params = {"a": 1.0, "b": 2.0}
    assert backend.evaluate_logprior(params, None) == pytest.approx(0.0)

    priors = {"a": _ConstPrior(1.5), "b": _ConstPrior(-0.5)}
    assert backend.evaluate_logprior(params, priors) == pytest.approx(1.0)

    with pytest.raises(KeyError, match="unknown parameter"):
        backend.evaluate_logprior({"a": 1.0}, {"b": _ConstPrior(1.0)})
