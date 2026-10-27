# type: ignore
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sympy import Symbol

from SymbolicDSGE.core.resolvers import (
    measurement_covariance,
    shock_covariance,
)
from SymbolicDSGE.core.config import (
    CovarianceSpec,
    PairGetterDict,
    SymbolGetterDict,
    cov_from_spec,
)
from SymbolicDSGE.estimation import Estimator
from SymbolicDSGE.estimation import backend


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


def test_build_Q_matches_post82_manual_structure(parsed_post82):
    params = parsed_post82.calibration.parameters
    Q = shock_covariance(parsed_post82)

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
    solved_post82, y_post82, monkeypatch
):
    config = solved_post82.config
    compiled = solved_post82.compiled
    compiled_type = type(compiled)
    est = Estimator(
        config=config,
        compiled=compiled,
        y=y_post82,
        observables=["Infl", "Rate"],
        estimated_params=["psi_pi", "rho_r"],
        ss_seed=solved_post82.policy.steady_state,
    )

    # prepare_filter_run resolves both cfunc addresses once; a loglik that
    # reached for a constructor again would rebuild them per eval.
    monkeypatch.setattr(
        compiled_type,
        "construct_measurement_cfunc",
        lambda self, obs=None: (_ for _ in ()).throw(
            AssertionError("measurement constructor called in hot path")
        ),
    )
    monkeypatch.setattr(
        compiled_type,
        "construct_measurement_jacobian_cfunc",
        lambda self, obs=None: (_ for _ in ()).throw(
            AssertionError("jacobian constructor called in hot path")
        ),
    )

    ll = est.loglik()
    assert np.isfinite(ll)


@pytest.fixture
def injected_measurement_noise(parsed_post82, monkeypatch):
    """Put a hand-built covariance spec on the model's own calibration.

    ``_parse_measurement_noise`` requires a std entry per observable, so a
    block that is empty or short of one cannot come out of a config file. A
    programmatically built or post-parse edited config reaches these branches,
    which is the state they guard, so the spec is assembled directly here and
    restored when the test ends.
    """

    def apply(std, corr=None):
        monkeypatch.setattr(
            parsed_post82.calibration,
            "measurement_noise",
            CovarianceSpec(std=SymbolGetterDict(std), corr=PairGetterDict(corr or {})),
        )
        return parsed_post82

    return apply


def test_build_R_override_and_config_branches(injected_measurement_noise):
    compiled = injected_measurement_noise(
        std={"OutGap": "s_o", "Infl": "s_i", "Rate": "s_r"}
    )

    # An override wins, but its shape must match the observable count.
    with pytest.raises(ValueError, match="Provided R has shape"):
        backend._build_R(
            compiled,
            ["Infl", "Rate"],
            {},
            R_override=np.eye(3, dtype=np.float64),
        )

    R_ok = np.array([[1.0, 0.1], [0.1, 2.0]], dtype=np.float64)
    out_override = backend._build_R(compiled, ["Infl", "Rate"], {}, R_override=R_ok)
    assert np.allclose(out_override, R_ok)

    # No override: R is rebuilt from params via the std/corr maps and sliced to
    # the requested observable order (diag from s_i/s_r/s_o, no correlations).
    params = {"s_i": 1.0, "s_r": 2.0, "s_o": 3.0}
    out_config = backend._build_R(compiled, ["Rate", "Infl"], params)
    assert np.allclose(out_config, np.array([[4.0, 0.0], [0.0, 1.0]], dtype=np.float64))

    # A block naming no standard deviation leaves nothing to assemble from.
    empty = injected_measurement_noise(std={})
    with pytest.raises(ValueError, match="measurement_noise` is required"):
        backend._build_R(empty, ["Infl", "Rate"], {})


def _post82_estimator(solved, y, **kwargs):
    """One estimated parameter held at its calibration, so ``loglik()``
    is the likelihood at the base calibration and can be compared against
    ``SolvedModel.kalman`` on the same data."""
    config = solved.config
    compiled = solved.compiled
    return Estimator(
        config=config,
        compiled=compiled,
        y=y,
        observables=["Infl", "Rate"],
        ss_seed=solved.policy.steady_state,
        estimated_params=["psi_pi"],
        **kwargs,
    )


def test_estimator_loglik_linear_matches_model_kalman(solved_post82, y_post82):
    est = _post82_estimator(solved_post82, y_post82, filter_mode="linear")
    ll_model_lin = solved_post82.kalman(
        y=y_post82,
        filter_mode="linear",
        observables=["Infl", "Rate"],
    ).loglik
    assert np.allclose(float(est.loglik()), ll_model_lin, rtol=1e-10, atol=1e-10)


def test_estimator_loglik_extended_matches_model_kalman(solved_post82, y_post82):
    est = _post82_estimator(solved_post82, y_post82, filter_mode="extended")
    ll_model_ext = solved_post82.kalman(
        y=y_post82,
        filter_mode="extended",
        observables=["Infl", "Rate"],
    ).loglik
    assert np.allclose(float(est.loglik()), ll_model_ext, rtol=1e-10, atol=1e-10)


def test_estimator_loglik_respects_R_override_and_mode_validation(
    solved_post82, y_post82
):
    with pytest.raises(ValueError, match="is not a valid FilterMode"):
        _post82_estimator(solved_post82, y_post82, filter_mode="bad_mode")

    calib = solved_post82.config.calibration
    scaled_R = 2.0 * cov_from_spec(
        solved_post82.compiled.observable_names,
        calib.measurement_noise,
        calib.parameters,
        subset=["Infl", "Rate"],
    )
    overridden = _post82_estimator(
        solved_post82, y_post82, filter_mode="linear", R=scaled_R
    )
    from_config = _post82_estimator(solved_post82, y_post82, filter_mode="linear")
    assert not np.isclose(
        float(overridden.loglik()),
        float(from_config.loglik()),
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


def test_measurement_covariance_error_branches(injected_measurement_noise):
    params = {"s_o": 1.0, "s_i": 1.0}

    # A block that names no standard deviation has nothing to build from.
    no_metadata = injected_measurement_noise(std={})
    with pytest.raises(ValueError, match="does not expose named parameter metadata"):
        measurement_covariance(no_metadata, params, observables=["Infl", "Rate"])

    # A named parameter absent from the supplied mapping is a hard error.
    missing_param = injected_measurement_noise(
        std={"OutGap": "s_o", "Infl": "s_i", "Rate": "not_in_params"}
    )
    with pytest.raises(KeyError, match="Missing covariance parameter"):
        measurement_covariance(missing_param, params, observables=["Infl", "Rate"])


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
def rbc_ukf_y(parsed_rbc_so):
    """Pure RNG around the consumption steady state. Parity compares two filter
    implementations on the same data, so it need not be model-consistent.

    ndarray, not a one-column frame: pandas hands back a read-only view for a
    single column under copy-on-write, which ``ukf_hot_loop`` rejects on the
    standalone interface path.
    """
    c_ss = float(parsed_rbc_so.calibration.parameters[Symbol("c_ss")])
    rng = np.random.default_rng(20260713)
    return c_ss + rng.normal(0.0, 0.05, size=(6, 1))


def test_estimator_loglik_unscented_matches_model_kalman(solved_rbc_so, rbc_ukf_y):
    """End-to-end UKF estimation parity: one ``Estimator.loglik`` at the
    calibration point (unscented) must equal ``SolvedModel.kalman`` unscented on
    the same data. Exercises the whole new path in one estimation: the
    ``order=2`` solve, the augmented ``build_unscented_P0``, and the seam's
    unscented branch (``bx``/``z0``/policy tensors -> ``run_unscented_raw``)."""
    solved, y = solved_rbc_so, rbc_ukf_y
    config = solved.config
    compiled = solved.compiled

    est = Estimator(
        config=config,
        compiled=compiled,
        y=y,
        observables=["c_obs"],
        filter_mode="unscented",
        estimated_params=["rho"],
        symmetrize=True,
    )
    ll_est = float(est.loglik())

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


def test_estimator_loglik_unscented_accepts_full_length_x0(solved_rbc_so, rbc_ukf_y):
    """A full n_var x0 exercises the seam's z0 slicing branch (raw[:n_state])."""
    config = solved_rbc_so.config
    compiled = solved_rbc_so.compiled

    est = Estimator(
        config=config,
        compiled=compiled,
        y=rbc_ukf_y,
        observables=["c_obs"],
        filter_mode="unscented",
        estimated_params=["rho"],
        symmetrize=True,
    )
    assert np.isfinite(float(est.loglik()))
