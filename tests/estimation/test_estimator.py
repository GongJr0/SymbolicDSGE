# type: ignore
from types import SimpleNamespace

import numpy as np
import pytest
from numpy import float64
from scipy.optimize import OptimizeResult
from sympy import Matrix, Symbol

import SymbolicDSGE.estimation.backend as est_backend
from SymbolicDSGE.bayesian.distributions.lkj_chol import LKJChol
from SymbolicDSGE.estimation import Estimator
from SymbolicDSGE.bayesian.priors import Prior
from SymbolicDSGE.bayesian.transforms import CholeskyCorrTransform, Identity
from SymbolicDSGE.core.config import PairGetterDict, SymbolGetterDict
from SymbolicDSGE.estimation.estimator import MissingConfigError, _MatrixPriorResolution


class _QuadraticPrior:
    def __init__(self, mean: float, weight: float):
        self.mean = float64(mean)
        self.weight = float64(weight)

    def logpdf(self, x):
        return float64(-self.weight * (float64(x) - self.mean) ** 2)


def _stub_compiled():
    a = Symbol("a")
    calibration = SimpleNamespace(parameters={a: float64(0.0)})
    config = SimpleNamespace(calibration=calibration)
    kalman = SimpleNamespace(y_names=["y"])
    return SimpleNamespace(
        config=config,
        calib_params=[a],
        kalman=kalman,
        observable_names=["y"],
    )


def _stub_compiled_with_r():
    a = Symbol("a")
    meas = Symbol("meas")
    calibration = SimpleNamespace(parameters={a: float64(0.0), meas: float64(1.0)})
    config = SimpleNamespace(calibration=calibration)
    kalman = SimpleNamespace(
        R_param_names=["a"],
        R_builder=lambda *vals: np.array([[vals[0]]], dtype=np.float64),
        y_names=["y"],
    )
    return SimpleNamespace(
        config=config,
        calib_params=[a, meas],
        kalman=kalman,
        observable_names=["y"],
    )


def _stub_compiled_with_dense_r_block():
    meas_a = Symbol("meas_a")
    meas_b = Symbol("meas_b")
    meas_rho_ab = Symbol("meas_rho_ab")
    calibration = SimpleNamespace(
        parameters={
            meas_a: float64(1.0),
            meas_b: float64(1.0),
            meas_rho_ab: float64(0.0),
        }
    )
    config = SimpleNamespace(calibration=calibration)
    R = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    def _R_builder(meas_a_val, meas_b_val, meas_rho_ab_val):
        return np.array(
            [
                [meas_a_val**2, meas_a_val * meas_b_val * meas_rho_ab_val],
                [meas_a_val * meas_b_val * meas_rho_ab_val, meas_b_val**2],
            ],
            dtype=np.float64,
        )

    kalman = SimpleNamespace(
        R=R,
        R_symbolic=Matrix(
            [
                [meas_a**2, meas_a * meas_b * meas_rho_ab],
                [meas_a * meas_b * meas_rho_ab, meas_b**2],
            ]
        ),
        R_param_names=["meas_a", "meas_b", "meas_rho_ab"],
        R_builder=_R_builder,
        R_std_param_map={"A": "meas_a", "B": "meas_b"},
        R_corr_param_map={frozenset(("A", "B")): "meas_rho_ab"},
        y_names=["A", "B"],
    )
    return SimpleNamespace(
        config=config,
        calib_params=[meas_a, meas_b, meas_rho_ab],
        kalman=kalman,
        observable_names=["A", "B"],
    )


def _stub_compiled_with_sparse_q_block():
    e1 = Symbol("e1")
    e2 = Symbol("e2")
    e3 = Symbol("e3")
    x1 = Symbol("x1")
    x2 = Symbol("x2")
    x3 = Symbol("x3")
    sig1 = Symbol("sig1")
    sig2 = Symbol("sig2")
    sig3 = Symbol("sig3")
    rho12 = Symbol("rho12")
    calibration = SimpleNamespace(
        parameters={
            sig1: float64(1.0),
            sig2: float64(1.0),
            sig3: float64(1.0),
            rho12: float64(0.0),
        },
        shock_std=SymbolGetterDict({e1: sig1, e2: sig2, e3: sig3}),
        shock_corr=PairGetterDict(
            {
                frozenset((e1, e2)): rho12,
                frozenset((e1, e3)): None,
                frozenset((e2, e3)): None,
            }
        ),
    )
    config = SimpleNamespace(
        calibration=calibration,
        shock_map={e1: x1, e2: x2, e3: x3},
    )
    return SimpleNamespace(
        config=config,
        calib_params=[sig1, sig2, sig3, rho12],
        kalman=SimpleNamespace(y_names=["y"]),
        observable_names=["y"],
        var_names=[x1, x2, x3],
        n_exog=3,
    )


def _fake_loglik(**kwargs):
    a = float64(kwargs["params"]["a"])
    return float64(-((a - 2.0) ** 2))


def test_mle_finds_quadratic_mode(monkeypatch):
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    out = est.mle(theta0=np.array([0.0], dtype=np.float64), bounds=[(-5.0, 5.0)])
    assert out.success
    assert abs(out.theta["a"] - 2.0) < 1e-4


def test_map_is_public_and_uses_prior(monkeypatch):
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)

    prior = _QuadraticPrior(mean=1.0, weight=5.0)
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": prior},
    )

    mle = est.mle(theta0=np.array([0.0], dtype=np.float64), bounds=[(-5.0, 5.0)])
    map_res = est.map(theta0=np.array([0.0], dtype=np.float64), bounds=[(-5.0, 5.0)])

    assert map_res.success
    assert abs(map_res.theta["a"] - 1.0) < abs(mle.theta["a"] - 1.0)
    assert 1.0 < map_res.theta["a"] < 2.0


def test_map_without_priors_raises(monkeypatch):
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    with pytest.raises(ValueError, match="requires priors"):
        est.map(theta0=np.array([0.0], dtype=np.float64))


def test_mcmc_returns_expected_shapes_and_stats(monkeypatch):
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": _QuadraticPrior(mean=1.0, weight=5.0)},
    )

    out = est.mcmc(
        n_draws=40,
        burn_in=40,
        thin=1,
        theta0=np.array([0.0], dtype=np.float64),
        random_state=123,
        adapt=True,
    )
    assert out.param_names == ["a"]
    assert out.samples.shape == (40, 1)
    assert out.logpost_trace.shape == (40,)
    assert 0.0 <= out.accept_rate <= 1.0


def test_estimator_make_prior_utility():
    prior = Estimator.make_prior(
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
        transform="identity",
    )
    assert isinstance(prior, Prior)


def test_safe_loglik_invalidates_system_exit(monkeypatch):
    def _boom(**kwargs):
        raise SystemExit("invertibility violation")

    monkeypatch.setattr(est_backend, "evaluate_loglik", _boom)
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    val = est._safe_loglik(np.array([0.0], dtype=np.float64))
    assert np.isneginf(val)


def test_safe_loglik_invalidates_stdout_warning_signal(monkeypatch):
    def _warn_only(**kwargs):
        print("Warning: unstable candidate")
        return float64(0.0)

    monkeypatch.setattr(est_backend, "evaluate_loglik", _warn_only)
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    val = est._safe_loglik(np.array([0.0], dtype=np.float64))
    assert np.isneginf(val)
    assert est._warning_signal_count >= 1


def test_estimation_reports_warning_count_once(monkeypatch, capsys):
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    _ = est.mle(theta0=np.array([0.0], dtype=np.float64), bounds=[(-5.0, 5.0)])
    lines = [
        ln
        for ln in capsys.readouterr().out.splitlines()
        if "BK stability warnings encountered" in ln
    ]
    assert len(lines) == 1


def test_theta_to_params_uses_prior_inverse_transform():
    prior = Estimator.make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": prior},
    )
    params = est.theta_to_params(np.array([-1.0], dtype=np.float64))
    assert params["a"] > 0.0


def test_params_to_theta_applies_forward_transform_for_mapping():
    prior = Estimator.make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": prior},
    )
    theta = est.params_to_theta({"a": np.e})
    assert np.allclose(theta[0], 1.0)


def test_matrix_prior_on_R_reparameterizes_pairwise_correlation_block():
    prior = LKJChol(eta=2.0, K=2, random_state=None)
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R"],
        priors={"R": prior},
    )

    theta = est.params_to_theta({"meas_rho_ab": 0.3})
    assert np.allclose(theta[0], np.arctanh(0.3))

    params = est.theta_to_params(theta)
    assert params["meas_rho_ab"] == pytest.approx(0.3)
    assert params["meas_a"] == pytest.approx(1.0)
    assert params["meas_b"] == pytest.approx(1.0)

    Lcorr = est_backend._corr_chol_from_unconstrained(theta, K=2)
    logdet = CholeskyCorrTransform(K=2).log_det_abs_jacobian_inverse(theta)
    assert est.logprior(theta) == pytest.approx(prior.logpdf(Lcorr) + logdet)


def test_matrix_prior_created_via_make_prior_uses_cholesky_corr_transform():
    prior = Estimator.make_prior(
        distribution="lkj_chol",
        parameters={"eta": 2.0, "K": 2},
        transform="cholesky_corr",
    )
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R"],
        priors={"R": prior},
    )

    theta = np.array([0.25], dtype=np.float64)
    Lcorr = CholeskyCorrTransform(K=2).inverse(theta)
    expected = prior.dist.logpdf(Lcorr) + prior.transform.log_det_abs_jacobian_inverse(
        theta
    )

    assert est.logprior(theta) == pytest.approx(expected)


def test_matrix_key_in_estimated_params_expands_to_member_names():
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R"],
        priors={"R": LKJChol(eta=2.0, K=2, random_state=None)},
    )

    assert est.param_names == ["meas_rho_ab"]
    assert list(est.priors.keys()) == ["R"]


def test_estimated_params_none_uses_prior_keys_when_priors_supplied():
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=None,
        priors={
            "R": LKJChol(eta=2.0, K=2, random_state=None),
            "meas_a": _QuadraticPrior(mean=1.0, weight=1.0),
        },
    )

    assert est.param_names == ["meas_rho_ab", "meas_a"]
    assert list(est.priors.keys()) == ["R", "meas_a"]


def test_extra_priors_outside_estimated_params_are_ignored():
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R"],
        priors={
            "R": LKJChol(eta=2.0, K=2, random_state=None),
            "meas_a": _QuadraticPrior(mean=1.0, weight=1.0),
        },
    )

    assert est.param_names == ["meas_rho_ab"]
    assert list(est.priors.keys()) == ["R"]


def test_matrix_prior_overlap_with_scalar_component_prior_raises():
    with pytest.raises(ValueError, match="meas_rho_ab"):
        Estimator(
            solver=SimpleNamespace(),
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["R"],
            priors={
                "R": LKJChol(eta=2.0, K=2, random_state=None),
                "meas_rho_ab": _QuadraticPrior(mean=0.0, weight=1.0),
            },
        )


def test_matrix_prior_on_R_keeps_mcmc_samples_in_valid_correlation_support(monkeypatch):
    monkeypatch.setattr(est_backend, "evaluate_loglik", lambda **kwargs: float64(0.0))

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R"],
        priors={"R": LKJChol(eta=2.0, K=2, random_state=None)},
    )

    out = est.mcmc(
        n_draws=12,
        burn_in=8,
        thin=1,
        theta0=np.array([0.0], dtype=np.float64),
        random_state=123,
        adapt=False,
    )

    assert np.all(np.abs(out.samples[:, 0]) < 1.0)


def test_sparse_q_block_for_lkj_prior_raises_descriptive_error():
    with pytest.raises(ValueError, match="dense correlation block") as excinfo:
        Estimator(
            solver=SimpleNamespace(),
            compiled=_stub_compiled_with_sparse_q_block(),
            y=np.zeros((4, 1), dtype=np.float64),
            estimated_params=["Q"],
            priors={"Q": LKJChol(eta=2.0, K=3, random_state=None)},
        )

    msg = str(excinfo.value)
    assert "sparse" in msg
    assert "fall back to their defaults" in msg
    assert "config DSL" in msg
    assert "placeholder default value" in msg


def test_mcmc_reports_samples_in_constrained_space_for_log_transform(monkeypatch):
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)

    prior = Estimator.make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": prior},
    )
    out = est.mcmc(
        n_draws=20,
        burn_in=10,
        thin=1,
        theta0=np.array([0.0], dtype=np.float64),
        random_state=123,
        adapt=False,
    )
    assert np.all(out.samples[:, 0] > 0.0)


def test_loglik_overrides_parameters_per_candidate(monkeypatch):
    seen = []

    def _capture(**kwargs):
        seen.append(float(kwargs["params"]["a"]))
        return float64(0.0)

    monkeypatch.setattr(est_backend, "evaluate_loglik", _capture)
    prior = Estimator.make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": prior},
    )
    _ = est._safe_loglik(np.array([-1.0], dtype=np.float64))
    _ = est._safe_loglik(np.array([1.0], dtype=np.float64))

    assert len(seen) == 2
    assert seen[0] == pytest.approx(np.exp(-1.0))
    assert seen[1] == pytest.approx(np.exp(1.0))


def test_mcmc_update_R_in_iterations_rebuilds_R_when_relevant_params_estimated(
    monkeypatch,
):
    compiled = _stub_compiled_with_r()
    build_calls = {"n": 0}
    seen_R = []

    def _build_R(**kwargs):
        build_calls["n"] += 1
        a_val = float(kwargs["params"]["a"])
        return np.array([[a_val]], dtype=np.float64)

    def _capture_loglik(**kwargs):
        seen_R.append(np.asarray(kwargs["R"], dtype=np.float64))
        return float64(0.0)

    monkeypatch.setattr(est_backend, "build_R_from_config_params", _build_R)
    monkeypatch.setattr(est_backend, "evaluate_loglik", _capture_loglik)

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=compiled,
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": _QuadraticPrior(mean=0.0, weight=1.0)},
    )
    _ = est.mcmc(
        n_draws=10,
        burn_in=5,
        thin=1,
        theta0=np.array([0.0], dtype=np.float64),
        random_state=123,
        adapt=False,
        update_R_in_iterations=True,
    )
    assert build_calls["n"] > 1
    assert len(seen_R) > 1


def test_mcmc_update_R_in_iterations_false_does_not_rebuild(monkeypatch):
    compiled = _stub_compiled_with_r()

    def _build_R(**kwargs):
        raise AssertionError(
            "R rebuild should not be called when update flag is false."
        )

    monkeypatch.setattr(est_backend, "build_R_from_config_params", _build_R)
    monkeypatch.setattr(est_backend, "evaluate_loglik", lambda **kwargs: float64(0.0))

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=compiled,
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": _QuadraticPrior(mean=0.0, weight=1.0)},
    )
    _ = est.mcmc(
        n_draws=5,
        burn_in=3,
        thin=1,
        theta0=np.array([0.0], dtype=np.float64),
        random_state=123,
        adapt=False,
        update_R_in_iterations=False,
    )


def test_mcmc_update_R_in_iterations_skips_when_R_params_not_estimated(monkeypatch):
    compiled = _stub_compiled_with_r()

    def _build_R(**kwargs):
        raise AssertionError(
            "R rebuild should not be called without relevant estimated params."
        )

    monkeypatch.setattr(est_backend, "build_R_from_config_params", _build_R)
    monkeypatch.setattr(est_backend, "evaluate_loglik", lambda **kwargs: float64(0.0))

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=compiled,
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["meas"],
        priors={"meas": _QuadraticPrior(mean=1.0, weight=1.0)},
    )
    _ = est.mcmc(
        n_draws=5,
        burn_in=3,
        thin=1,
        theta0=np.array([1.0], dtype=np.float64),
        random_state=123,
        adapt=False,
        update_R_in_iterations=True,
    )


def test_estimator_constructor_and_lkj_prior_validation_error_branches():
    with pytest.raises(ValueError, match="Unknown estimated parameters"):
        Estimator(
            solver=SimpleNamespace(),
            compiled=_stub_compiled(),
            y=np.zeros((3, 1), dtype=np.float64),
            estimated_params=["ghost"],
        )

    with pytest.raises(ValueError, match="specified more than once"):
        Estimator(
            solver=SimpleNamespace(),
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["R", "meas_rho_ab"],
            priors={"R": LKJChol(eta=2.0, K=2, random_state=None)},
        )

    with pytest.raises(TypeError, match="CholeskyCorrTransform"):
        Estimator(
            solver=SimpleNamespace(),
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["R"],
            priors={
                "R": Prior(
                    dist=LKJChol(eta=2.0, K=2, random_state=None),
                    transform=Identity(),
                )
            },
        )

    with pytest.raises(TypeError, match="matching K values"):
        Estimator(
            solver=SimpleNamespace(),
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["R"],
            priors={
                "R": Prior(
                    dist=LKJChol(eta=2.0, K=2, random_state=None),
                    transform=CholeskyCorrTransform(K=3),
                )
            },
        )

    with pytest.raises(TypeError, match="must be an LKJChol"):
        Estimator(
            solver=SimpleNamespace(),
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["R"],
            priors={"R": _QuadraticPrior(mean=0.0, weight=1.0)},
        )


def test_cov_to_corr_and_matrix_resolution_error_branches():
    with pytest.raises(ValueError, match="square covariance matrix"):
        Estimator._cov_to_corr(np.array([1.0], dtype=np.float64), "R")
    with pytest.raises(ValueError, match="symmetric covariance matrix"):
        Estimator._cov_to_corr(
            np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float64), "R"
        )
    with pytest.raises(ValueError, match="strictly positive diagonal variances"):
        Estimator._cov_to_corr(
            np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64), "R"
        )

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )

    with pytest.raises(ValueError, match="named variance parameter"):
        est._build_matrix_resolution(
            key="R",
            labels=["a"],
            std_param_map={},
            corr_param_map={},
        )

    with pytest.raises(ValueError, match="unique named variance parameter"):
        est._build_matrix_resolution(
            key="R",
            labels=["a", "b"],
            std_param_map={"a": "sig", "b": "sig"},
            corr_param_map={frozenset(("b", "a")): "rho_ab"},
        )

    with pytest.raises(ValueError, match="unique named parameter per correlation pair"):
        est._build_matrix_resolution(
            key="R",
            labels=["a", "b", "c"],
            std_param_map={"a": "sig_a", "b": "sig_b", "c": "sig_c"},
            corr_param_map={
                frozenset(("b", "a")): "rho_shared",
                frozenset(("c", "a")): "rho_shared",
                frozenset(("c", "b")): "rho_cb",
            },
        )


def test_resolve_r_and_effective_observables_error_paths():
    a = Symbol("a")
    compiled_no_kalman = SimpleNamespace(
        config=SimpleNamespace(
            calibration=SimpleNamespace(parameters={a: float64(0.0)})
        ),
        calib_params=[a],
        observable_names=["y"],
    )
    est_no_kalman = Estimator(
        solver=SimpleNamespace(),
        compiled=compiled_no_kalman,
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    with pytest.raises(MissingConfigError, match="Kalman configuration"):
        est_no_kalman._resolve_R()

    est_missing_meta = Estimator(
        solver=SimpleNamespace(),
        compiled=SimpleNamespace(
            config=SimpleNamespace(
                calibration=SimpleNamespace(parameters={a: float64(0.0)})
            ),
            calib_params=[a],
            kalman=SimpleNamespace(y_names=["y"], R=np.eye(1, dtype=np.float64)),
            observable_names=["y"],
        ),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    with pytest.raises(ValueError, match="parser-generated R std/correlation metadata"):
        est_missing_meta._resolve_R()

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    est._prepared_filter = None
    est.observables = ["ghost"]
    with pytest.raises(ValueError, match="Unknown observables"):
        est._effective_observables()


def test_theta_conversion_logprior_and_safe_wrapper_error_branches():
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": _QuadraticPrior(mean=0.0, weight=1.0)},
    )

    with pytest.raises(ValueError, match="missing estimated parameters"):
        est.params_to_theta({})
    with pytest.raises(ValueError, match="params array must be 1D"):
        est.params_to_theta(np.array([[1.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="does not match estimated parameter count"):
        est.params_to_theta(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ValueError, match="theta must be a 1D array"):
        est.theta_to_params(np.array([[1.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="does not match estimated parameter count"):
        est.theta_to_params(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ValueError, match="theta must be a 1D array"):
        est.logprior(np.array([[0.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="does not match estimated parameter count"):
        est.logprior(np.array([0.0, 1.0], dtype=np.float64))

    est.priors = {"ghost": _QuadraticPrior(mean=0.0, weight=1.0)}
    with pytest.raises(KeyError, match="unknown parameter"):
        est.logprior(np.array([0.0], dtype=np.float64))

    assert est._count_stdout_warning_signals("Warning: one\nok\nWarning: two") == 2

    def _warn(th):
        print("Warning: unstable")
        return float64(0.0)

    est._logpost_with_overrides = _warn
    assert np.isneginf(est._safe_logpost(np.array([0.0], dtype=np.float64)))

    est._logpost_with_overrides = lambda th: (_ for _ in ()).throw(RuntimeError("boom"))
    assert np.isneginf(est._safe_logpost(np.array([0.0], dtype=np.float64)))

    est.logprior = lambda th: float64(np.inf)
    assert np.isneginf(est._safe_logprior(np.array([0.0], dtype=np.float64)))
    est.logprior = lambda th: (_ for _ in ()).throw(RuntimeError("boom"))
    assert np.isneginf(est._safe_logprior(np.array([0.0], dtype=np.float64)))


def test_pack_opt_result_and_mcmc_validation_branches(monkeypatch):
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": _QuadraticPrior(mean=0.0, weight=1.0)},
    )

    packed = est._pack_opt_result(
        "mle",
        OptimizeResult(
            x=np.array([1.0], dtype=np.float64),
            success=True,
            message="ok",
            fun=np.float64(-1.0),
            nfev=3,
            nit=None,
        ),
    )
    assert packed.kind == "mle"
    assert packed.nit is None

    with pytest.raises(ValueError, match="n_draws must be positive"):
        est.mcmc(n_draws=0)
    with pytest.raises(ValueError, match="burn_in must be non-negative"):
        est.mcmc(n_draws=1, burn_in=-1)
    with pytest.raises(ValueError, match="thin must be positive"):
        est.mcmc(n_draws=1, thin=0)

    est_no_priors = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    with pytest.raises(ValueError, match="requires priors"):
        est_no_priors.mcmc(n_draws=1)

    est_empty = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=[],
    )
    est_empty.priors = {"dummy": _QuadraticPrior(mean=0.0, weight=1.0)}
    with pytest.raises(ValueError, match="No estimated parameters"):
        est_empty.mcmc(n_draws=1)


def test_resolve_q_missing_pair_key_and_block_validation_branches(monkeypatch):
    e1 = Symbol("e1")
    e2 = Symbol("e2")
    x1 = Symbol("x1")
    x2 = Symbol("x2")
    sig1 = Symbol("sig1")
    sig2 = Symbol("sig2")
    calibration = SimpleNamespace(
        parameters={sig1: float64(1.0), sig2: float64(1.0)},
        shock_std=SymbolGetterDict({e1: sig1, e2: sig2}),
        shock_corr={},
    )
    compiled = SimpleNamespace(
        config=SimpleNamespace(calibration=calibration, shock_map={e1: x1, e2: x2}),
        calib_params=[sig1, sig2],
        kalman=SimpleNamespace(y_names=["y"]),
        observable_names=["y"],
        var_names=[x1, x2],
        n_exog=2,
    )
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=compiled,
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["sig1"],
    )
    resolution = est._resolve_Q()
    assert resolution.missing_pairs == [("e2", "e1")]

    est_base = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )

    res_dim1 = _MatrixPriorResolution(
        key="R",
        dim=1,
        labels=["A"],
        std_names=["sig_a"],
        member_names=[],
        pair_positions=[],
        missing_pairs=[],
        param_tags={
            "sig_a": est_base._build_matrix_resolution(
                key="R",
                labels=["A"],
                std_param_map={"A": "sig_a"},
                corr_param_map={},
            ).param_tags["sig_a"]
        },
    )
    est_base.priors = {"R": object()}
    monkeypatch.setattr(
        est_base,
        "_coerce_lkj_prior",
        lambda name, prior_obj: SimpleNamespace(
            dist=SimpleNamespace(_K=1), logpdf=lambda z: float64(0.0)
        ),
    )
    monkeypatch.setattr(est_base, "_resolve_R", lambda params=None: res_dim1)
    with pytest.raises(ValueError, match="dimension at least 2"):
        est_base._build_matrix_prior_blocks()

    res_short = _MatrixPriorResolution(
        key="R",
        dim=3,
        labels=["A", "B", "C"],
        std_names=["sig_a", "sig_b", "sig_c"],
        member_names=["rho_ba", "rho_ca"],
        pair_positions=[(1, 0), (2, 0)],
        missing_pairs=[],
        param_tags={
            "sig_a": est_base._build_matrix_resolution(
                key="R",
                labels=["A", "B", "C"],
                std_param_map={"A": "sig_a", "B": "sig_b", "C": "sig_c"},
                corr_param_map={
                    frozenset(("B", "A")): "rho_ba",
                    frozenset(("C", "A")): "rho_ca",
                    frozenset(("C", "B")): "rho_cb",
                },
            ).param_tags["sig_a"],
        },
    )
    est_base.priors = {"R": LKJChol(eta=2.0, K=3, random_state=None)}
    monkeypatch.setattr(est_base, "_resolve_R", lambda params=None: res_short)
    with pytest.raises(ValueError, match="dense correlation block"):
        est_base._build_matrix_prior_blocks()

    res_missing = est_base._build_matrix_resolution(
        key="R",
        labels=["A", "B"],
        std_param_map={"A": "sig_a", "B": "sig_b"},
        corr_param_map={frozenset(("B", "A")): "rho_ba"},
    )
    est_base.priors = {"R": LKJChol(eta=2.0, K=2, random_state=None)}
    monkeypatch.setattr(est_base, "_resolve_R", lambda params=None: res_missing)
    with pytest.raises(ValueError, match="Missing from estimated_params"):
        est_base._build_matrix_prior_blocks()


def test_matrix_block_overlap_k_mismatch_and_invalid_corr_error(monkeypatch):
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["meas_rho_ab"],
    )
    r_resolution = est._build_matrix_resolution(
        key="R",
        labels=["A", "B"],
        std_param_map={"A": "meas_a", "B": "meas_b"},
        corr_param_map={frozenset(("B", "A")): "meas_rho_ab"},
    )
    q_resolution = _MatrixPriorResolution(
        key="Q",
        dim=2,
        labels=["u", "v"],
        std_names=["sig_u", "sig_v"],
        member_names=["meas_rho_ab"],
        pair_positions=[(1, 0)],
        missing_pairs=[],
        param_tags={"meas_rho_ab": r_resolution.param_tags["meas_rho_ab"]},
    )
    est.priors = {
        "R": LKJChol(eta=2.0, K=2, random_state=None),
        "Q": LKJChol(eta=2.0, K=2, random_state=None),
    }
    monkeypatch.setattr(est, "_resolve_R", lambda params=None: r_resolution)
    monkeypatch.setattr(est, "_resolve_Q", lambda params=None: q_resolution)
    with pytest.raises(ValueError, match="cannot share member parameters"):
        est._build_matrix_prior_blocks()

    est_k = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R"],
    )
    est_k.priors = {"R": LKJChol(eta=2.0, K=3, random_state=None)}
    monkeypatch.setattr(est_k, "_resolve_R", lambda params=None: r_resolution)
    with pytest.raises(ValueError, match="has K=3"):
        est_k._build_matrix_prior_blocks()

    block = (
        est_k._build_matrix_prior_blocks.__self__._matrix_blocks
        if hasattr(est_k, "_matrix_blocks")
        else {}
    )
    good_est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R"],
        priors={"R": LKJChol(eta=2.0, K=2, random_state=None)},
    )
    good_block = good_est._matrix_blocks["R"]
    bad_corr = np.array([[1.0, 1.2], [1.2, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="do not form a valid"):
        good_est._block_cpc_from_corr(good_block, bad_corr)


def test_effective_observables_logprior_base_branch_and_logpost(monkeypatch):
    compiled_no_kalman = SimpleNamespace(
        config=SimpleNamespace(calibration=SimpleNamespace(parameters={})),
        calib_params=[],
        observable_names=["y1", "y2"],
    )
    est_obs = Estimator(
        solver=SimpleNamespace(),
        compiled=compiled_no_kalman,
        y=np.zeros((2, 2), dtype=np.float64),
        estimated_params=[],
    )
    est_obs._prepared_filter = None
    est_obs.observables = None
    assert est_obs._effective_observables() == ["y1", "y2"]

    est_prepared = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((2, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    est_prepared._prepared_filter = SimpleNamespace(observables=["y"])
    est_prepared.observables = ["other"]
    assert est_prepared._effective_observables() == ["y"]

    prior = Estimator.make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    est_base_prior = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R"],
        priors={"R": LKJChol(eta=2.0, K=2, random_state=None)},
    )
    est_base_prior.priors = {"meas_a": prior}
    est_base_prior._matrix_blocks = {}
    est_base_prior._matrix_member_names = set()
    theta = np.array([0.0], dtype=np.float64)
    x0 = float64(est_base_prior._base_params["meas_a"])
    expected = prior.logpdf(prior.transform.safe_forward(x0))
    assert est_base_prior.logprior(theta) == pytest.approx(expected)

    logpost_est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": _QuadraticPrior(mean=0.0, weight=1.0)},
    )
    theta0 = np.array([0.2], dtype=np.float64)
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)
    assert logpost_est.logpost(theta0) == pytest.approx(
        logpost_est.loglik(theta0) + logpost_est.logprior(theta0)
    )


def test_mle_map_dynamic_r_and_adaptation_branches(monkeypatch):
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": _QuadraticPrior(mean=0.0, weight=1.0)},
    )

    def fake_minimize(fun, x0, method=None, bounds=None, options=None):
        assert np.isinf(fun(np.asarray(x0, dtype=np.float64)))
        return OptimizeResult(
            x=np.asarray(x0, dtype=np.float64),
            success=True,
            message="ok",
            fun=np.float64(0.0),
            nfev=1,
            nit=1,
        )

    monkeypatch.setattr(
        est_backend, "evaluate_loglik", lambda **kwargs: float64(np.nan)
    )
    monkeypatch.setattr(est_backend.optimize, "minimize", fake_minimize)
    _ = est.mle(theta0=np.array([0.0], dtype=np.float64))
    _ = est.map(theta0=np.array([0.0], dtype=np.float64))

    compiled_r = _stub_compiled_with_r()
    monkeypatch.setattr(
        est_backend,
        "build_R_from_config_params",
        lambda **kwargs: np.array([[float(kwargs["params"]["a"])]], dtype=np.float64),
    )

    def _warn_logpost(th, *, params_override=None, R_override=None):
        print("Warning: dynamic R unstable")
        return float64(0.0)

    est_warn = Estimator(
        solver=SimpleNamespace(),
        compiled=compiled_r,
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": _QuadraticPrior(mean=0.0, weight=1.0)},
    )
    est_warn._logpost_with_overrides = _warn_logpost
    _ = est_warn.mcmc(
        n_draws=1,
        burn_in=1,
        thin=1,
        theta0=np.array([0.0], dtype=np.float64),
        random_state=123,
        adapt=False,
        update_R_in_iterations=True,
    )

    est_err = Estimator(
        solver=SimpleNamespace(),
        compiled=compiled_r,
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": _QuadraticPrior(mean=0.0, weight=1.0)},
    )
    monkeypatch.setattr(
        est_backend,
        "build_R_from_config_params",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    _ = est_err.mcmc(
        n_draws=1,
        burn_in=1,
        thin=1,
        theta0=np.array([0.0], dtype=np.float64),
        random_state=123,
        adapt=False,
        update_R_in_iterations=True,
    )

    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)
    est_adapt = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_r(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a", "meas"],
        priors={
            "a": _QuadraticPrior(mean=0.0, weight=1.0),
            "meas": _QuadraticPrior(mean=1.0, weight=1.0),
        },
    )
    _ = est_adapt.mcmc(
        n_draws=2,
        burn_in=2,
        thin=1,
        theta0=np.array([0.0, 1.0], dtype=np.float64),
        random_state=123,
        adapt=True,
        adapt_start=0,
        adapt_interval=1,
    )
