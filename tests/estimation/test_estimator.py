# type: ignore
from types import SimpleNamespace

import numpy as np
import pytest
from numpy import float64
from sympy import Symbol

import SymbolicDSGE.estimation.backend as est_backend
from SymbolicDSGE.estimation import Estimator
from SymbolicDSGE.bayesian.priors import Prior


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
    return SimpleNamespace(config=config, calib_params=[a])


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
        if "warning signals encountered" in ln
    ]
    assert len(lines) == 1


def test_theta_to_params_uses_prior_inverse_transform():
    prior = Estimator.make_prior(
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
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
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
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


def test_mcmc_reports_samples_in_constrained_space_for_log_transform(monkeypatch):
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)

    prior = Estimator.make_prior(
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
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
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
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
