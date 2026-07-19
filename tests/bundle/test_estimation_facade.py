from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from numpy import float64
from sympy import Symbol

from SymbolicDSGE.bayesian.priors import make_prior
from SymbolicDSGE.bundle.builder import BundleBuilder
from SymbolicDSGE.bundle.parquet import collapse_columns, from_parquet_columns
from SymbolicDSGE.estimation import Estimator
from SymbolicDSGE.estimation.results import MCMCResult, OptimizationResult


def _with_filter_prep(compiled: SimpleNamespace) -> SimpleNamespace:
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


def _stub_compiled() -> SimpleNamespace:
    a = Symbol("a")
    calibration = SimpleNamespace(parameters={a: float64(0.3)})
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


def _estimator() -> Estimator:
    return Estimator(
        solver=SimpleNamespace(),
        compiled=_stub_compiled(),
        y=np.zeros((6, 1), dtype=np.float64),
        observables=["y"],
        estimated_params=["a"],
        priors={
            "a": make_prior(
                distribution="normal",
                parameters={"mean": 0.3, "std": 0.1},
                transform="identity",
                transform_kwargs={},
            )
        },
    )


def _optimization_result() -> OptimizationResult:
    return OptimizationResult(
        kind="map",
        x=np.array([0.31], dtype=np.float64),
        theta={"a": float64(0.31)},
        success=True,
        message="ok",
        fun=float64(1.0),
        loglik=float64(-1.0),
        logprior=float64(-0.5),
        logpost=float64(-1.5),
        nfev=12,
        nit=4,
        optimizer_config={
            "method": "L-BFGS-B",
            "bounds": [[-1.0, 1.0]],
            "options": {"maxiter": 20},
        },
    )


def test_facade_flattens_optimization_run() -> None:
    builder = BundleBuilder().add_estimation(
        _estimator(), result=_optimization_result()
    )
    _, files = builder.build()

    spec = json.loads(files["estimation/spec.json"])
    assert spec["method"] == "map"
    # method kwargs folded from the result's optimizer_config (bounds excluded)
    assert spec["method_kwargs"] == {"method": "L-BFGS-B", "options": {"maxiter": 20}}
    param = spec["parameters"][0]
    assert param["name"] == "a"
    assert param["initial"] == 0.3  # from calibration
    assert (param["lower"], param["upper"]) == (-1.0, 1.0)  # folded from bounds
    assert param["prior"]["distribution"] == "normal"  # reversed from live Prior

    # result + observed members written
    result_doc = json.loads(files["estimation/result.json"])
    assert result_doc["type"] == "optimization"
    assert "estimation/observed.parquet" in files


def test_facade_flattens_mcmc_run_with_posterior() -> None:
    rng = np.random.default_rng(0)
    mcmc = MCMCResult(
        param_names=["a"],
        samples=rng.standard_normal((10, 1)),
        logpost_trace=rng.standard_normal(10),
        accept_rate=float64(0.4),
        n_draws=10,
        burn_in=2,
        thin=1,
        sampler_config={"adapt": True, "proposal_scale": 0.2, "random_state": 7},
    )

    builder = BundleBuilder().add_estimation(_estimator(), result=mcmc)
    _, files = builder.build()

    spec = json.loads(files["estimation/spec.json"])
    assert spec["method"] == "mcmc"
    # n_draws/burn_in/thin + sampler tuning all surface as method kwargs
    assert spec["method_kwargs"]["n_draws"] == 10
    assert spec["method_kwargs"]["burn_in"] == 2
    assert spec["method_kwargs"]["proposal_scale"] == 0.2
    assert spec["method_kwargs"]["random_state"] == 7

    # posterior auto-extracted from the live result
    assert "estimation/posterior.parquet" in files
    posterior = collapse_columns(
        from_parquet_columns(files["estimation/posterior.parquet"])
    )
    np.testing.assert_allclose(posterior["samples"], mcmc.samples)


def test_facade_requires_result_for_estimator() -> None:
    with pytest.raises(ValueError, match="requires a live"):
        BundleBuilder().add_estimation(_estimator())


def test_facade_rejects_unknown_source() -> None:
    with pytest.raises(TypeError, match="EstimationSpec or Estimator"):
        BundleBuilder().add_estimation(object())  # type: ignore[arg-type]
