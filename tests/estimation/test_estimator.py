# type: ignore
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from numpy import float64
from scipy.optimize import OptimizeResult
from sympy import Matrix, Symbol

import SymbolicDSGE.estimation.backend as est_backend
import SymbolicDSGE.estimation.estimator as est_estimator
from SymbolicDSGE.bayesian.distributions.lkj_chol import LKJChol
from SymbolicDSGE.estimation import Estimator
from SymbolicDSGE.bayesian.priors import Prior
from SymbolicDSGE.bayesian.transforms import (
    AffineLogitTransform,
    CholeskyCorrTransform,
    Identity,
    LogTransform,
    TanhTransform,
)
from SymbolicDSGE.core.config import PairGetterDict, SymbolGetterDict
from SymbolicDSGE.estimation.estimator import MissingConfigError
from SymbolicDSGE.estimation.backend import MatrixPriorBlock
from SymbolicDSGE.estimation.results import MLEResult


class _QuadraticPrior:
    def __init__(self, mean: float, weight: float):
        self.mean = float64(mean)
        self.weight = float64(weight)

    def logpdf(self, x):
        return float64(-self.weight * (float64(x) - self.mean) ** 2)


def _with_filter_prep(compiled):
    """Complete a stub with the surface Estimator's construction-time filter prep
    needs. ``Estimator.__init__`` builds the filter run unconditionally now (the
    old duck-typed guard is gone), so every stub must satisfy
    ``prepare_filter_run``. These tests fake ``evaluate_loglik``, so the cfunc
    addresses and P0 are never evaluated; they only have to exist."""
    if not hasattr(compiled, "observable_names"):
        compiled.observable_names = ["y"]
    if not hasattr(compiled, "var_names"):
        compiled.var_names = [
            Symbol(f"s{i}") for i in range(len(compiled.observable_names))
        ]
    if not hasattr(compiled, "cur_syms"):
        compiled.cur_syms = list(compiled.var_names)
    compiled.construct_measurement_cfunc = lambda obs: SimpleNamespace(address=0)
    compiled.construct_observable_jacobian_cfunc = lambda obs: SimpleNamespace(
        address=0
    )
    if not hasattr(compiled, "n_state"):
        compiled.n_state = len(compiled.var_names)
    if getattr(compiled.kalman, "P0", None) is None:
        compiled.kalman.P0 = np.eye(len(compiled.var_names), dtype=np.float64)
    if not hasattr(compiled.kalman, "R_param_names"):
        compiled.kalman.R_param_names = None
    return compiled


def _stub_compiled():
    a = Symbol("a")
    calibration = SimpleNamespace(parameters={a: float64(0.0)})
    config = SimpleNamespace(calibration=calibration)
    kalman = SimpleNamespace(y_names=["y"])
    return _with_filter_prep(
        SimpleNamespace(
            config=config,
            calib_params=[a],
            kalman=kalman,
            observable_names=["y"],
        )
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
    return _with_filter_prep(
        SimpleNamespace(
            config=config,
            calib_params=[a, meas],
            kalman=kalman,
            observable_names=["y"],
        )
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
    return _with_filter_prep(
        SimpleNamespace(
            config=config,
            calib_params=[meas_a, meas_b, meas_rho_ab],
            kalman=kalman,
            observable_names=["A", "B"],
        )
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
    return _with_filter_prep(
        SimpleNamespace(
            config=config,
            calib_params=[sig1, sig2, sig3, rho12],
            kalman=SimpleNamespace(y_names=["y"]),
            observable_names=["y"],
            var_names=[x1, x2, x3],
            n_exog=3,
        )
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


def test_to_spec_captures_targets_initials_and_method():
    from SymbolicDSGE.estimation.spec import EstimationSpec, PriorSpec

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )

    spec = est.to_spec(
        method="mle",
        priors={"a": PriorSpec(distribution="normal", parameters={"loc": 0.0})},
        observables=["y"],
    )

    assert isinstance(spec, EstimationSpec)
    assert spec.method == "mle"
    assert [p.name for p in spec.parameters] == ["a"]
    param = spec.parameters[0]
    assert param.estimate is True
    assert param.initial == 0.0  # pulled from calibration via _base_params
    assert param.prior is not None and param.prior.distribution == "normal"
    assert spec.observables == ["y"]


def test_to_spec_reverses_live_scalar_priors():
    from SymbolicDSGE.bayesian.priors import make_prior

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={
            "a": make_prior(
                distribution="normal",
                parameters={"mean": 0.0, "std": 1.0},
                transform="identity",
                transform_kwargs={},
            )
        },
    )

    spec = est.to_spec(method="map")  # no explicit priors -> auto-reversed

    param = spec.parameters[0]
    assert param.name == "a"
    assert param.prior is not None
    assert param.prior.distribution == "normal"
    assert param.prior.parameters == {"mean": 0.0, "std": 1.0}


def test_mle_records_optimizer_config(monkeypatch):
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    out = est.mle(
        theta0=np.array([0.0], dtype=np.float64),
        bounds=[(-5.0, 5.0)],
        options={"maxiter": 10},
    )

    assert out.optimizer_config["method"] == "L-BFGS-B"
    assert out.optimizer_config["bounds"] == [[-5.0, 5.0]]
    assert out.optimizer_config["options"] == {"maxiter": 10}
    # config survives projection to the serializable dict
    assert out.to_dict()["optimizer_config"] == out.optimizer_config


def test_mcmc_records_sampler_config(monkeypatch):
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": _QuadraticPrior(mean=1.0, weight=5.0)},
    )
    out = est.mcmc(n_draws=5, burn_in=2, thin=1, random_state=7, proposal_scale=0.2)

    cfg = out.sampler_config
    assert cfg["random_state"] == 7
    assert cfg["proposal_scale"] == 0.2
    assert set(cfg) == {
        "adapt",
        "adapt_start",
        "adapt_interval",
        "proposal_scale",
        "adapt_epsilon",
        "random_state",
    }
    # n_draws/burn_in/thin stay on the result itself (not duplicated in config)
    assert "n_draws" not in cfg
    assert out.to_meta().sampler_config == cfg


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


def test_mcmc_seed_zero_is_exactly_reproducible(monkeypatch):
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": _QuadraticPrior(mean=1.0, weight=5.0)},
    )

    kwargs = dict(
        n_draws=30,
        burn_in=30,
        thin=1,
        theta0=np.array([0.0], dtype=np.float64),
        random_state=0,
        adapt=True,
    )
    out1 = est.mcmc(**kwargs)
    out2 = est.mcmc(**kwargs)

    assert np.array_equal(out1.samples, out2.samples)
    assert np.array_equal(out1.logpost_trace, out2.logpost_trace)
    assert out1.accept_rate == pytest.approx(out2.accept_rate)


def test_mcmc_clones_generator_input_state(monkeypatch):
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
        priors={"a": _QuadraticPrior(mean=1.0, weight=5.0)},
    )

    shared_rng = np.random.default_rng(123)
    kwargs = dict(
        n_draws=30,
        burn_in=30,
        thin=1,
        theta0=np.array([0.0], dtype=np.float64),
        random_state=shared_rng,
        adapt=True,
    )
    out1 = est.mcmc(**kwargs)
    out2 = est.mcmc(**kwargs)

    assert np.array_equal(out1.samples, out2.samples)
    assert np.array_equal(out1.logpost_trace, out2.logpost_trace)
    assert out1.accept_rate == pytest.approx(out2.accept_rate)
    assert shared_rng.random() == pytest.approx(np.random.default_rng(123).random())


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


def test_safe_loglik_invalidates_warning_signal(monkeypatch):
    def _warn_only(**kwargs):
        warnings.warn("unstable candidate", RuntimeWarning)
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
        estimated_params=["R_corr"],
        priors={"R_corr": prior},
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
        estimated_params=["R_corr"],
        priors={"R_corr": prior},
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
        estimated_params=["R_corr"],
        priors={"R_corr": LKJChol(eta=2.0, K=2, random_state=None)},
    )

    assert est.param_names == ["meas_rho_ab"]
    assert list(est.priors.keys()) == ["R_corr"]


def test_estimated_params_none_uses_prior_keys_when_priors_supplied():
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=None,
        priors={
            "R_corr": LKJChol(eta=2.0, K=2, random_state=None),
            "meas_a": _QuadraticPrior(mean=1.0, weight=1.0),
        },
    )

    assert est.param_names == ["meas_rho_ab", "meas_a"]
    assert list(est.priors.keys()) == ["R_corr", "meas_a"]


def test_extra_priors_outside_estimated_params_are_ignored():
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R_corr"],
        priors={
            "R_corr": LKJChol(eta=2.0, K=2, random_state=None),
            "meas_a": _QuadraticPrior(mean=1.0, weight=1.0),
        },
    )

    assert est.param_names == ["meas_rho_ab"]
    assert list(est.priors.keys()) == ["R_corr"]


def test_matrix_prior_overlap_with_scalar_component_prior_raises():
    with pytest.raises(ValueError, match="meas_rho_ab"):
        Estimator(
            solver=SimpleNamespace(),
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["R_corr"],
            priors={
                "R_corr": LKJChol(eta=2.0, K=2, random_state=None),
                "meas_rho_ab": _QuadraticPrior(mean=0.0, weight=1.0),
            },
        )


def test_matrix_prior_on_R_keeps_mcmc_samples_in_valid_correlation_support(monkeypatch):
    monkeypatch.setattr(est_backend, "evaluate_loglik", lambda **kwargs: float64(0.0))

    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R_corr"],
        priors={"R_corr": LKJChol(eta=2.0, K=2, random_state=None)},
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
            estimated_params=["Q_corr"],
            priors={"Q_corr": LKJChol(eta=2.0, K=3, random_state=None)},
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
            estimated_params=["R_corr", "meas_rho_ab"],
            priors={"R_corr": LKJChol(eta=2.0, K=2, random_state=None)},
        )

    with pytest.raises(TypeError, match="CholeskyCorrTransform"):
        Estimator(
            solver=SimpleNamespace(),
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["R_corr"],
            priors={
                # Support-valid but not a CholeskyCorrTransform: exercises the
                # estimator's LKJ-transform check (AffineLogit(-1,1) matches the
                # LKJChol (-1, 1) support so Prior construction itself succeeds).
                "R_corr": Prior(
                    dist=LKJChol(eta=2.0, K=2, random_state=None),
                    transform=AffineLogitTransform(low=-1.0, high=1.0),
                )
            },
        )

    with pytest.raises(TypeError, match="matching K values"):
        Estimator(
            solver=SimpleNamespace(),
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["R_corr"],
            priors={
                "R_corr": Prior(
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
            estimated_params=["R_corr"],
            priors={"R_corr": _QuadraticPrior(mean=0.0, weight=1.0)},
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
            key="R_corr",
            labels=["a"],
            std_param_map={},
            corr_param_map={},
        )

    with pytest.raises(ValueError, match="unique named variance parameter"):
        est._build_matrix_resolution(
            key="R_corr",
            labels=["a", "b"],
            std_param_map={"a": "sig", "b": "sig"},
            corr_param_map={frozenset(("b", "a")): "rho_ab"},
        )

    with pytest.raises(ValueError, match="unique named parameter per correlation pair"):
        est._build_matrix_resolution(
            key="R_corr",
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
        kalman=None,
    )
    # A Kalman configuration is now non-negotiable: the estimator fails fast at
    # construction rather than lazily when the R block is resolved.
    with pytest.raises(MissingConfigError, match="Kalman configuration"):
        Estimator(
            solver=SimpleNamespace(),
            compiled=compiled_no_kalman,
            y=np.zeros((3, 1), dtype=np.float64),
            estimated_params=["a"],
        )

    est_missing_meta = Estimator(
        solver=SimpleNamespace(),
        compiled=_with_filter_prep(
            SimpleNamespace(
                config=SimpleNamespace(
                    calibration=SimpleNamespace(parameters={a: float64(0.0)})
                ),
                calib_params=[a],
                kalman=SimpleNamespace(
                    y_names=["y"],
                    R=np.eye(1, dtype=np.float64),
                    R_std_param_map=None,
                    R_corr_param_map=None,
                ),
                observable_names=["y"],
            )
        ),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    with pytest.raises(ValueError, match="parser-generated R std/correlation metadata"):
        est_missing_meta._resolve_R()

    # Unknown observables are now rejected at construction by the filter prep.
    with pytest.raises(ValueError, match="Unknown observables"):
        Estimator(
            solver=SimpleNamespace(),
            compiled=_stub_compiled(),
            y=np.zeros((3, 1), dtype=np.float64),
            observables=["ghost"],
            estimated_params=["a"],
        )


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

    def _warn(th):
        warnings.warn("unstable", RuntimeWarning)
        return float64(0.0)

    est._logpost = _warn
    assert np.isneginf(est._safe_logpost(np.array([0.0], dtype=np.float64)))

    est._logpost = lambda th: (_ for _ in ()).throw(RuntimeError("boom"))
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
    assert isinstance(packed, MLEResult)
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
    compiled = _with_filter_prep(
        SimpleNamespace(
            config=SimpleNamespace(calibration=calibration, shock_map={e1: x1, e2: x2}),
            calib_params=[sig1, sig2],
            kalman=SimpleNamespace(y_names=["y"]),
            observable_names=["y"],
            var_names=[x1, x2],
            n_exog=2,
        )
    )
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=compiled,
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["sig1"],
    )
    block = est._resolve_Q()
    present = {(int(r), int(c)) for r, c in block.positions}
    missing = [
        (block.labels[row], block.labels[col])
        for row in range(1, block.dim)
        for col in range(row)
        if (row, col) not in present
    ]
    assert missing == [("e2", "e1")]

    est_base = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    # These branches drive _build_matrix_prior_blocks directly with monkeypatched
    # resolutions; mark R_corr as a requested block so the loop runs over it.
    est_base._requested_reserved_keys = ("R_corr",)

    res_dim1 = MatrixPriorBlock(
        dim=1,
        labels=["A"],
        member_names=[],
        positions=np.empty((0, 2), dtype=np.int64),
        theta_slice=slice(0, 0),
        prior=None,
    )
    est_base.priors = {"R_corr": object()}
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

    res_short = MatrixPriorBlock(
        dim=3,
        labels=["A", "B", "C"],
        member_names=["rho_ba", "rho_ca"],
        positions=np.array([[1, 0], [2, 0]], dtype=np.int64),
        theta_slice=slice(0, 0),
        prior=None,
    )
    est_base.priors = {"R_corr": LKJChol(eta=2.0, K=3, random_state=None)}
    monkeypatch.setattr(est_base, "_resolve_R", lambda params=None: res_short)
    with pytest.raises(ValueError, match="dense correlation block"):
        est_base._build_matrix_prior_blocks()

    res_missing = est_base._build_matrix_resolution(
        key="R_corr",
        labels=["A", "B"],
        std_param_map={"A": "sig_a", "B": "sig_b"},
        corr_param_map={frozenset(("B", "A")): "rho_ba"},
    )
    est_base.priors = {"R_corr": LKJChol(eta=2.0, K=2, random_state=None)}
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
        key="R_corr",
        labels=["A", "B"],
        std_param_map={"A": "meas_a", "B": "meas_b"},
        corr_param_map={frozenset(("B", "A")): "meas_rho_ab"},
    )
    q_resolution = MatrixPriorBlock(
        dim=2,
        labels=["u", "v"],
        member_names=["meas_rho_ab"],
        positions=np.array([[1, 0]], dtype=np.int64),
        theta_slice=slice(0, 0),
        prior=None,
    )
    est.priors = {
        "R_corr": LKJChol(eta=2.0, K=2, random_state=None),
        "Q_corr": LKJChol(eta=2.0, K=2, random_state=None),
    }
    est._requested_reserved_keys = ("R_corr", "Q_corr")
    monkeypatch.setattr(est, "_resolve_R", lambda params=None: r_resolution)
    monkeypatch.setattr(est, "_resolve_Q", lambda params=None: q_resolution)
    with pytest.raises(ValueError, match="cannot share member parameters"):
        est._build_matrix_prior_blocks()

    est_k = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R_corr"],
    )
    est_k.priors = {"R_corr": LKJChol(eta=2.0, K=3, random_state=None)}
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
        estimated_params=["R_corr"],
        priors={"R_corr": LKJChol(eta=2.0, K=2, random_state=None)},
    )
    good_block = good_est._matrix_blocks["R_corr"]
    bad_corr = np.array([[1.0, 1.2], [1.2, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="do not form a valid"):
        good_est._block_cpc_from_corr(good_block, bad_corr)


def test_mle_std_member_without_prior_gets_log_transform():
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["meas_a"],
    )
    # A variance estimated prior-free is positivity-constrained by role.
    assert isinstance(est._param_transforms["meas_a"], LogTransform)


def test_mle_isolated_scalar_corr_without_prior_gets_tanh_transform():
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_sparse_q_block(),
        y=np.zeros((3, 1), dtype=np.float64),
        estimated_params=["rho12"],
    )
    # rho12 is the sole named Q correlation (e1, e2): sparse and isolated, so it
    # stays a standalone scalar tanh rather than folding into a block.
    assert "Q_corr" not in est._matrix_blocks
    assert isinstance(est._param_transforms["rho12"], TanhTransform)


def test_mle_full_dense_corr_set_promotes_to_cpc_block():
    est = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["meas_rho_ab"],
    )
    # The dense R correlation set folds into an R_corr CPC block instead of a
    # standalone scalar; its member is block-handled (scalar transform unused).
    assert "R_corr" in est._matrix_blocks
    assert "meas_rho_ab" in est._matrix_blocks["R_corr"].member_names
    assert isinstance(est._param_transforms["meas_rho_ab"], Identity)


def test_spd_std_member_rejects_conflicting_prior_transform():
    # An Identity-transform prior on a variance would map onto R, not (0, inf),
    # breaking the shared theta<->param map; rejected once, at construction.
    prior = Estimator.make_prior(
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
        transform="identity",
    )
    with pytest.raises(ValueError, match="requires a constraint to"):
        Estimator(
            solver=SimpleNamespace(),
            compiled=_stub_compiled_with_dense_r_block(),
            y=np.zeros((4, 2), dtype=np.float64),
            estimated_params=["meas_a"],
            priors={"meas_a": prior},
        )


def test_logprior_base_branch_and_logpost(monkeypatch):
    prior = Estimator.make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    est_base_prior = Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled_with_dense_r_block(),
        y=np.zeros((4, 2), dtype=np.float64),
        estimated_params=["R_corr"],
        priors={"R_corr": LKJChol(eta=2.0, K=2, random_state=None)},
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


def test_mle_map_and_adaptation_branches(monkeypatch):
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
    monkeypatch.setattr(est_estimator.optimize, "minimize", fake_minimize)
    _ = est.mle(theta0=np.array([0.0], dtype=np.float64))
    _ = est.map(theta0=np.array([0.0], dtype=np.float64))

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
