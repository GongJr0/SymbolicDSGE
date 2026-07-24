# type: ignore
from types import SimpleNamespace

import numpy as np
import pytest
from numpy import float64
from sympy import Symbol

import SymbolicDSGE.estimation.backend as est_backend
from SymbolicDSGE.core.solver import DSGESolver
from SymbolicDSGE.estimation.estimator import Estimator
from SymbolicDSGE.estimation.results import MLEResult


class _UnitIntervalPrior:
    def logpdf(self, x):
        x = float64(x)
        if not (0.0 < x < 1.0):
            raise ValueError("out of support")
        return float64(0.0)


class _QuadraticPrior:
    def __init__(self, mean: float, weight: float):
        self.mean = float64(mean)
        self.weight = float64(weight)

    def logpdf(self, x):
        return float64(-self.weight * (float64(x) - self.mean) ** 2)


def _make_solver() -> DSGESolver:
    return DSGESolver(
        model_config=SimpleNamespace(),
        kalman_config=SimpleNamespace(),
    )


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


def _make_compiled(a0: float = 0.0):
    a = Symbol("a")
    return _with_filter_prep(
        SimpleNamespace(
            config=SimpleNamespace(
                calibration=SimpleNamespace(parameters={a: float64(a0)}),
            ),
            calib_params=[a],
            kalman=SimpleNamespace(
                R=None, P0=None, R_std_param_map=None, R_corr_param_map=None
            ),
        )
    )


def _make_compiled_two(a0: float = 0.0, b0: float = 3.0):
    a = Symbol("a")
    b = Symbol("b")
    return _with_filter_prep(
        SimpleNamespace(
            config=SimpleNamespace(
                calibration=SimpleNamespace(
                    parameters={a: float64(a0), b: float64(b0)}
                ),
            ),
            calib_params=[a, b],
            kalman=SimpleNamespace(
                R=None, P0=None, R_std_param_map=None, R_corr_param_map=None
            ),
        )
    )


def _fake_loglik(**kwargs):
    a = float64(kwargs["params"]["a"])
    return float64(-((a - 2.0) ** 2))


def test_solver_exposes_private_estimator_factory():
    solver = _make_solver()
    compiled = _make_compiled(0.0)

    est = solver._estimator(
        compiled=compiled,
        y=np.zeros((4, 1), dtype=np.float64),
        estimated_params=["a"],
    )
    assert isinstance(est, Estimator)


def test_solver_estimate_validates_config_initial_guess_against_prior(post82):
    solver = post82["solver"]

    # map validates the initial guess against the priors' support before the
    # search; a psi_pi of 2.0 sits outside the unit interval.
    with pytest.raises(ValueError, match="incompatible with the provided priors"):
        solver.estimate(
            compiled=post82["compiled"],
            y=post82["y"],
            method="map",
            observables=post82["obs"],
            estimated_params=["psi_pi"],
            priors={"psi_pi": _UnitIntervalPrior()},
            ss_seed=post82["steady"],
            theta0={"psi_pi": 2.0},
        )


def test_solver_estimate_and_solve_mle(post82, monkeypatch):
    solver = post82["solver"]

    captured = {}

    def fake_solve(compiled, *, parameters=None, ss_seed=None):
        captured["parameters"] = parameters
        return SimpleNamespace(params=parameters)

    monkeypatch.setattr(solver, "solve", fake_solve)

    result, solved = solver.estimate_and_solve(
        compiled=post82["compiled"],
        y=post82["y"],
        method="mle",
        observables=post82["obs"],
        estimated_params=["psi_pi", "rho_r"],
        ss_seed=post82["steady"],
        theta0=np.array([2.0, 0.8], dtype=np.float64),
        bounds=[(1.0, 5.0), (0.0, 0.99)],
    )
    assert isinstance(result, MLEResult)
    # estimate_and_solve threads the estimated point into solve as plain floats
    assert "psi_pi" in captured["parameters"] and "rho_r" in captured["parameters"]
    assert captured["parameters"] == {k: float(v) for k, v in result.theta.items()}
    assert solved.params == captured["parameters"]


def test_solver_estimate_accepts_theta0_dictionary(post82):
    solver = post82["solver"]

    out = solver.estimate(
        compiled=post82["compiled"],
        y=post82["y"],
        method="mle",
        observables=post82["obs"],
        estimated_params=["psi_pi", "rho_r"],
        ss_seed=post82["steady"],
        theta0={"psi_pi": 2.0, "rho_r": 0.8},
        bounds=[(1.0, 5.0), (0.0, 0.99)],
    )
    assert isinstance(out, MLEResult)
    assert out.success
    assert "psi_pi" in out.theta and "rho_r" in out.theta


def test_solver_estimate_rejects_incomplete_theta0_dictionary(monkeypatch):
    solver = _make_solver()
    compiled = _make_compiled(0.0)
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)

    with pytest.raises(ValueError, match="missing estimated parameters"):
        solver.estimate(
            compiled=compiled,
            y=np.zeros((4, 1), dtype=np.float64),
            method="mle",
            estimated_params=["a"],
            theta0={},
        )


def test_solver_theta0_dictionary_is_mapped_to_unconstrained_for_transformed_prior(
    post82,
):
    solver = post82["solver"]

    # A log-transformed prior means the dict theta0 (given in parameter space) is
    # mapped through the transform to the unconstrained theta the driver expects;
    # a positive psi_pi must land in a finite basin rather than blowing up.
    prior = Estimator.make_prior(
        distribution="log_normal",
        parameters={"mean": 0.0, "std": 0.5},
        transform="log",
    )
    with pytest.warns(UserWarning, match="MLE will ignore"):
        out = solver.estimate(
            compiled=post82["compiled"],
            y=post82["y"],
            method="mle",
            observables=post82["obs"],
            estimated_params=["psi_pi"],
            priors={"psi_pi": prior},
            ss_seed=post82["steady"],
            theta0={"psi_pi": 2.0},
            bounds=[(1.0, 5.0)],
        )
    assert out.success


def test_solver_estimate_and_solve_mcmc(monkeypatch):
    solver = _make_solver()
    compiled = _make_compiled(0.0)
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)

    captured = {}

    def fake_solve(compiled, *, parameters=None, ss_seed=None):
        captured["parameters"] = parameters
        return SimpleNamespace(params=parameters)

    solver.solve = fake_solve  # type: ignore[method-assign]

    result, solved = solver.estimate_and_solve(
        compiled=compiled,
        y=np.zeros((4, 1), dtype=np.float64),
        method="mcmc",
        posterior_point="map",
        estimated_params=["a"],
        priors={"a": _QuadraticPrior(mean=1.0, weight=2.0)},
        n_draws=30,
        burn_in=30,
        random_state=123,
    )

    assert result.samples.shape == (30, 1)
    assert "a" in captured["parameters"]
    assert solved.params == captured["parameters"]


def test_solver_estimate_and_solve_mcmc_preserves_non_estimated_params(monkeypatch):
    solver = _make_solver()
    compiled = _make_compiled_two(0.0, 3.0)
    monkeypatch.setattr(est_backend, "evaluate_loglik", _fake_loglik)

    captured = {}

    def fake_solve(compiled, *, parameters=None, ss_seed=None):
        captured["parameters"] = parameters
        return SimpleNamespace(params=parameters)

    solver.solve = fake_solve  # type: ignore[method-assign]

    result, solved = solver.estimate_and_solve(
        compiled=compiled,
        y=np.zeros((4, 1), dtype=np.float64),
        method="mcmc",
        posterior_point="mean",
        estimated_params=["a"],
        priors={"a": _QuadraticPrior(mean=1.0, weight=2.0)},
        n_draws=20,
        burn_in=20,
        random_state=123,
    )

    assert result.samples.shape == (20, 1)
    assert "a" in captured["parameters"]
    assert "b" in captured["parameters"]
    assert float(captured["parameters"]["b"]) == pytest.approx(3.0)
    sym_map = {
        getattr(k, "name", k): v
        for k, v in compiled.config.calibration.parameters.items()
    }
    assert float(sym_map["a"]) == pytest.approx(captured["parameters"]["a"])
    assert float(sym_map["b"]) == pytest.approx(captured["parameters"]["b"])
    assert solved.params == captured["parameters"]
