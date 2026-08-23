# type: ignore
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sympy import Symbol

from SymbolicDSGE import ModelParser, DSGESolver
from SymbolicDSGE.estimation import Estimator
from SymbolicDSGE.estimation import backend
from SymbolicDSGE.kalman.config import KalmanConfig


@pytest.fixture(scope="module")
def post82_bundle(post82_test_model_path):
    model, kalman = ModelParser(post82_test_model_path).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()

    steady = np.zeros((compiled.n_declared,), dtype=np.float64)
    solved = solver.solve(compiled=compiled, ss_seed=steady)

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
            "e_g": rng.normal(0.0, sig_g, size=T),
            "e_z": rng.normal(0.0, sig_z, size=T),
            "e_r": rng.normal(0.0, sig_r, size=T),
        },
        x0=np.zeros((compiled.n_var,), dtype=np.float64),
        observables=True,
    )
    y = pd.DataFrame(
        {
            "OutGap": sim.observables["OutGap"][1:],
            "Infl": sim.observables["Infl"][1:],
            "Rate": sim.observables["Rate"][1:],
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


def test_extract_base_params_reads_the_calibration_by_name():
    a = Symbol("a")
    b = Symbol("b")
    compiled = SimpleNamespace(
        config=SimpleNamespace(
            calibration=SimpleNamespace(parameters={a: 1.5, b: 2.5})
        ),
        calib_params=[b, a],
    )

    base = backend.extract_base_params(compiled)
    assert base == {"a": pytest.approx(1.5), "b": pytest.approx(2.5)}


def test_reorder_observables_dataframe_and_ndarray_paths():
    compiled = SimpleNamespace(observable_names=["Infl", "Rate"])

    df = pd.DataFrame({"Rate": [1.0, 2.0], "Infl": [3.0, 4.0]})
    obs, y_df = backend.reorder_observables(compiled, None, df)
    assert obs == ["Infl", "Rate"]
    assert np.allclose(y_df, np.array([[3.0, 1.0], [4.0, 2.0]], dtype=np.float64))

    arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)  # [Rate, Infl]
    obs_arr, y_arr = backend.reorder_observables(compiled, ["Rate", "Infl"], arr)
    assert obs_arr == ["Infl", "Rate"]
    assert np.allclose(y_arr, np.array([[20.0, 10.0], [40.0, 30.0]], dtype=np.float64))


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
        backend.reorder_observables(compiled, observables, y)


def test_reorder_observables_uses_compiled_default_and_validates_dataframe_columns():
    compiled = SimpleNamespace(observable_names=["Infl", "Rate"])
    df = pd.DataFrame({"Infl": [1.0], "Rate": [2.0]})

    obs, y = backend.reorder_observables(compiled, None, df)
    assert obs == ["Infl", "Rate"]
    assert np.allclose(y, np.array([[1.0, 2.0]], dtype=np.float64))

    with pytest.raises(ValueError, match="missing observable columns"):
        backend.reorder_observables(
            compiled,
            None,
            pd.DataFrame({"Infl": [1.0]}),
        )


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


def test_estimator_loglik_reuses_prepared_measurement_dispatchers(
    post82_bundle, monkeypatch
):
    compiled = post82_bundle["compiled"]
    compiled_type = type(compiled)
    est = Estimator(
        compiled=compiled,
        y=post82_bundle["y"],
        observables=["Infl", "Rate"],
        ss_seed=post82_bundle["steady"],
    )

    monkeypatch.setattr(
        compiled_type,
        "construct_measurement_array_func",
        lambda self, obs: (_ for _ in ()).throw(
            AssertionError("measurement constructor called in hot path")
        ),
    )
    monkeypatch.setattr(
        compiled_type,
        "construct_observable_jacobian_array_func",
        lambda self, obs: (_ for _ in ()).throw(
            AssertionError("jacobian constructor called in hot path")
        ),
    )

    ll = est.loglik(est.theta0())
    assert np.isfinite(ll)


def test_build_R_override_and_config_branches():
    compiled = SimpleNamespace(observable_names=["Infl", "Rate", "Out"])
    kalman = KalmanConfig(
        R=None,
        R_std_param_map={"Infl": "s_i", "Rate": "s_r", "Out": "s_o"},
        R_corr_param_map={},
    )

    # An override wins, but its shape must match the observable count.
    with pytest.raises(ValueError, match="Provided R has shape"):
        backend.build_R(
            compiled,
            kalman,
            ["Infl", "Rate"],
            {},
            R_override=np.eye(3, dtype=np.float64),
        )

    R_ok = np.array([[1.0, 0.1], [0.1, 2.0]], dtype=np.float64)
    out_override = backend.build_R(
        compiled, kalman, ["Infl", "Rate"], {}, R_override=R_ok
    )
    assert np.allclose(out_override, R_ok)

    # No override: R is rebuilt from params via the std/corr maps and sliced to
    # the requested observable order (diag from s_i/s_r/s_o, no correlations).
    params = {"s_i": 1.0, "s_r": 2.0, "s_o": 3.0}
    out_config = backend.build_R(compiled, kalman, ["Rate", "Infl"], params)
    assert np.allclose(out_config, np.array([[4.0, 0.0], [0.0, 1.0]], dtype=np.float64))

    # A directly-configured constant R with no named-param maps is sliced as-is
    # (nothing to rebuild from params), preserving observable order.
    const_kalman = KalmanConfig(
        R=np.array(
            [[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]], dtype=np.float64
        ),
    )
    out_const = backend.build_R(compiled, const_kalman, ["Rate", "Infl"], {})
    assert np.allclose(out_const, np.array([[5.0, 2.0], [2.0, 1.0]], dtype=np.float64))

    # No override, no maps, no constant R: genuinely unavailable.
    empty_kalman = KalmanConfig(R=None)
    with pytest.raises(ValueError, match="R is not available"):
        backend.build_R(compiled, empty_kalman, ["Infl", "Rate"], {})


@pytest.mark.skip(
    reason="R-estimation rework: R_builder/R_symbolic removed; build_R_from_config_params "
    "is being rewired onto make_R and the R std/corr param maps."
)
def test_kalman_symbolic_R_builder_matches_numeric_config_R(post82_bundle):
    compiled = post82_bundle["compiled"]
    kalman = compiled.kalman
    assert kalman is not None
    assert kalman.R_symbolic is not None
    assert kalman.R_builder is not None
    assert kalman.R_param_names == [
        "meas_outgap",
        "meas_infl",
        "meas_rate",
        "meas_rho_ir",
        "meas_rho_gi",
        "meas_rho_gr",
    ]

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


@pytest.mark.skip(
    reason="R-estimation rework: build_R_from_config_params is being rewired onto make_R "
    "and the R std/corr param maps; the R_builder metadata guard no longer applies."
)
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


def _post82_estimator(bundle, **kwargs):
    """One estimated parameter held at its calibration, so ``loglik(theta0())``
    is the likelihood at the base calibration and can be compared against
    ``SolvedModel.kalman`` on the same data."""
    return Estimator(
        compiled=bundle["compiled"],
        y=bundle["y"],
        observables=["Infl", "Rate"],
        ss_seed=bundle["steady"],
        estimated_params=["psi_pi"],
        **kwargs,
    )


def test_estimator_loglik_linear_matches_model_kalman(post82_bundle):
    est = _post82_estimator(post82_bundle, filter_mode="linear")
    ll_model_lin = (
        post82_bundle["solved"]
        .kalman(
            y=post82_bundle["y"],
            filter_mode="linear",
            observables=["Infl", "Rate"],
        )
        .loglik
    )
    assert np.allclose(
        float(est.loglik(est.theta0())), ll_model_lin, rtol=1e-10, atol=1e-10
    )


def test_estimator_loglik_extended_matches_model_kalman(post82_bundle):
    est = _post82_estimator(post82_bundle, filter_mode="extended")
    ll_model_ext = (
        post82_bundle["solved"]
        .kalman(
            y=post82_bundle["y"],
            filter_mode="extended",
            observables=["Infl", "Rate"],
        )
        .loglik
    )
    assert np.allclose(
        float(est.loglik(est.theta0())), ll_model_ext, rtol=1e-10, atol=1e-10
    )


def test_estimator_loglik_respects_R_override_and_mode_validation(post82_bundle):
    with pytest.raises(ValueError, match="is not a valid FilterMode"):
        _post82_estimator(post82_bundle, filter_mode="bad_mode")

    scaled_R = 2.0 * post82_bundle["compiled"].kalman.R[:2, :2]
    overridden = _post82_estimator(post82_bundle, filter_mode="linear", R=scaled_R)
    from_config = _post82_estimator(post82_bundle, filter_mode="linear")
    assert not np.isclose(
        float(overridden.loglik(overridden.theta0())),
        float(from_config.loglik(from_config.theta0())),
    )


def test_corr_chol_unconstrained_parameterization():
    z = np.array([0.2, -0.1, 0.3], dtype=np.float64)
    L = backend._corr_chol_from_unconstrained(z, K=3)
    corr = L @ L.T
    assert np.allclose(np.diag(corr), np.ones(3), atol=1e-10)
    assert np.all(np.linalg.eigvalsh(corr) > 0.0)
    z_from_L = backend._unconstrained_from_corr_chol(L)
    z_from_corr = backend._unconstrained_from_corr(corr)
    assert np.allclose(z_from_L, z, atol=1e-10, rtol=0.0)
    assert np.allclose(z_from_corr, z, atol=1e-10, rtol=0.0)


def test_resolve_filter_options_prefers_defaults_and_honors_overrides():
    # jitter/symmetrize are call-site concerns; defaulted when unset, else honored.
    assert backend.resolve_filter_options(None, None) == pytest.approx((0.0, False))
    assert backend.resolve_filter_options(0.5, True) == pytest.approx((0.5, True))


def test_build_R_from_config_params_error_branches():
    compiled = SimpleNamespace(observable_names=["a", "b"])
    params = {"sig_a": 1.0, "sig_b": 1.0}

    with pytest.raises(ValueError, match="KalmanConfig is required"):
        backend.build_R_from_config_params(
            compiled=compiled,
            kalman=None,
            observables=["a", "b"],
            params=params,
        )

    with pytest.raises(ValueError, match="named R parameter metadata"):
        backend.build_R_from_config_params(
            compiled=compiled,
            kalman=SimpleNamespace(R_std_param_map=None, R_corr_param_map=None),
            observables=["a", "b"],
            params=params,
        )

    with pytest.raises(KeyError, match="Missing R parameter"):
        backend.build_R_from_config_params(
            compiled=compiled,
            kalman=SimpleNamespace(
                R_std_param_map={"a": "sig_a", "b": "not_in_params"},
                R_corr_param_map={},
            ),
            observables=["a", "b"],
            params=params,
        )


def test_backend_corr_cov_helpers_and_validation_error_paths():
    z = np.array([0.2, -0.1, 0.3], dtype=np.float64)
    L = backend._corr_chol_from_unconstrained(z, 3)
    assert L.shape == (3, 3)
    assert np.allclose(np.diag(L @ L.T), np.ones(3), atol=1e-10)

    z_back = backend._unconstrained_from_corr_chol(L)
    assert np.allclose(z_back, z, atol=1e-10, rtol=0.0)

    with pytest.raises(ValueError, match="Expected 3 unconstrained CPC elements"):
        backend._corr_chol_from_unconstrained(np.array([0.1], dtype=np.float64), K=3)
    with pytest.raises(ValueError, match="square lower-triangular"):
        backend._unconstrained_from_corr_chol(np.array([1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="lower triangular"):
        backend._unconstrained_from_corr_chol(
            np.array([[1.0, 0.2], [0.0, 1.0]], dtype=np.float64)
        )
    with pytest.raises(ValueError, match="must be positive"):
        backend._unconstrained_from_corr_chol(
            np.array([[0.0, 0.0], [0.1, 1.0]], dtype=np.float64)
        )
    with pytest.raises(ValueError, match="unit norm"):
        backend._unconstrained_from_corr_chol(
            np.array([[1.0, 0.0], [0.8, 0.3]], dtype=np.float64)
        )
    with pytest.raises(ValueError, match="must be square"):
        backend._unconstrained_from_corr(np.array([1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="must be symmetric"):
        backend._unconstrained_from_corr(
            np.array([[1.0, 0.2], [0.1, 1.0]], dtype=np.float64)
        )
    with pytest.raises(ValueError, match="unit diagonal"):
        backend._unconstrained_from_corr(
            np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        )
    with pytest.raises(ValueError, match="positive definite"):
        backend._unconstrained_from_corr(
            np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
        )


def test_unconstrained_from_corr_chol_clips_extreme_cpc_values():
    # The native kernel clamps partial correlations to the open unit interval.
    # Feed factors whose rows exceed unit norm (bypassing the validating wrapper)
    # so the raw clamp path is exercised directly.
    z_pos = backend.unconstrained_from_corr_chol(
        np.array([[1.0, 0.0], [1.1, 0.1]], dtype=np.float64)
    )
    z_neg = backend.unconstrained_from_corr_chol(
        np.array([[1.0, 0.0], [-1.1, 0.1]], dtype=np.float64)
    )
    assert np.isfinite(z_pos[0])
    assert np.isfinite(z_neg[0])


@pytest.fixture(scope="module")
def rbc_ukf_bundle():
    """Second-order RBC with a minimal one-observable Kalman config, solved to
    order 2. The observed series is pure RNG around the consumption steady state;
    parity only compares two filter implementations on the same data, so it need
    not be model-consistent."""
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "models"
        / "rbc_second_order.yaml"
    )
    model, _ = ModelParser(path).get_all()
    kalman = KalmanConfig(R=np.array([[0.01]], dtype=np.float64))
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    solved = solver.solve(compiled=compiled, order=2)

    # ndarray (not a single-column DataFrame): pandas hands back a read-only
    # view for one column under copy-on-write, which ``ukf_hot_loop`` rejects on
    # the standalone interface path. Same values reach both filters either way.
    c_ss = float(model.calibration.parameters[Symbol("c_ss")])
    rng = np.random.default_rng(20260713)
    y = (c_ss + rng.normal(0.0, 0.05, size=(6, 1))).astype(np.float64)
    return {"solver": solver, "compiled": compiled, "solved": solved, "y": y}


def test_estimator_loglik_unscented_matches_model_kalman(rbc_ukf_bundle):
    """End-to-end UKF estimation parity: one ``Estimator.loglik`` at the
    calibration point (unscented) must equal ``SolvedModel.kalman`` unscented on
    the same data. Exercises the whole new path in one estimation: the
    ``order=2`` solve, the augmented ``build_unscented_P0``, and the seam's
    unscented branch (``bx``/``z0``/policy tensors -> ``run_unscented_raw``)."""
    solver = rbc_ukf_bundle["solver"]
    compiled = rbc_ukf_bundle["compiled"]
    solved = rbc_ukf_bundle["solved"]
    y = rbc_ukf_bundle["y"]

    est = Estimator(
        compiled=compiled,
        y=y,
        observables=["c_obs"],
        filter_mode="unscented",
        estimated_params=["rho"],
        symmetrize=True,
    )
    ll_est = float(est.loglik(est.theta0()))

    ll_ref = float(
        solved.kalman(
            y=y,
            filter_mode="unscented",
            observables=["c_obs"],
            symmetrize=True,
        ).loglik
    )

    assert np.isfinite(ll_est)
    assert ll_est == pytest.approx(ll_ref, rel=1e-9, abs=1e-9)


def test_estimator_loglik_unscented_accepts_full_length_x0(rbc_ukf_bundle):
    """A full n_var x0 exercises the seam's z0 slicing branch (raw[:n_state])."""
    compiled = rbc_ukf_bundle["compiled"]
    x0 = np.zeros(
        (len(compiled.var_names),), dtype=np.float64
    )  # length n_var > n_state

    est = Estimator(
        compiled=compiled,
        y=rbc_ukf_bundle["y"],
        observables=["c_obs"],
        filter_mode="unscented",
        estimated_params=["rho"],
        x0=x0,
        symmetrize=True,
    )
    assert np.isfinite(float(est.loglik(est.theta0())))
