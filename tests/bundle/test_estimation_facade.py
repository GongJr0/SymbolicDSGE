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
from SymbolicDSGE.estimation.results import MCMCResult, MAPResult


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
    if not hasattr(compiled, "n_var"):
        compiled.n_var = len(compiled.var_names)
    if getattr(compiled.kalman, "P0", None) is None:
        compiled.kalman.P0 = np.eye(len(compiled.var_names), dtype=np.float64)
    if not hasattr(compiled.kalman, "R_param_names"):
        compiled.kalman.R_param_names = None
    if not hasattr(compiled.kalman, "R_std_param_map"):
        compiled.kalman.R_std_param_map = None
    if getattr(compiled.kalman, "R", None) is None:
        compiled.kalman.R = np.eye(len(compiled.observable_names), dtype=np.float64)
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


def _optimization_result() -> MAPResult:
    return MAPResult(
        x=np.array([0.31], dtype=np.float64),
        theta={"a": float64(0.31)},
        success=True,
        message="ok",
        fun=float64(1.0),
        nfev=12,
        nit=4,
        optimizer_config={
            "method": "L-BFGS-B",
            "bounds": [[-1.0, 1.0]],
            "options": {"maxiter": 20},
        },
        logpost=float64(-1.5),
        logprior=float64(-0.5),
    )


def test_facade_flattens_optimization_run() -> None:
    builder = BundleBuilder().add_estimation(
        _estimator(), result=_optimization_result()
    )
    _, files = builder.build()

    # the spec carries construction state only
    spec = json.loads(files["estimation/spec.json"])
    assert spec["estimated_params"] == ["a"]
    assert spec["observables"] == ["y"]
    assert spec["priors"]["a"]["distribution"] == "normal"  # reversed from live Prior
    assert "method" not in spec and "method_kwargs" not in spec

    # the run's own arguments ride the result
    result_doc = json.loads(files["estimation/result.json"])
    assert result_doc["type"] == "map"
    config = result_doc["data"]["optimizer_config"]
    assert config["method"] == "L-BFGS-B"
    assert config["bounds"] == [[-1.0, 1.0]]
    assert config["options"]["maxiter"] == 20

    assert "estimation/observed.parquet" in files


def test_facade_flattens_mcmc_run_with_posterior() -> None:
    rng = np.random.default_rng(0)
    mcmc = MCMCResult(
        param_names=["a"],
        samples=rng.standard_normal((10, 1)),
        logpost_trace=rng.standard_normal(10),
        logjac_trace=rng.standard_normal(10),
        accept_rate=float64(0.4),
        n_draws=10,
        burn_in=2,
        thin=1,
        sampler_config={"adapt": True, "proposal_scale": 0.2, "random_state": 7},
    )

    builder = BundleBuilder().add_estimation(_estimator(), result=mcmc)
    _, files = builder.build()

    spec = json.loads(files["estimation/spec.json"])
    assert spec["estimated_params"] == ["a"]

    # draw counts and sampler tuning ride the result's meta, not the spec
    meta = json.loads(files["estimation/result.json"])["data"]
    assert meta["n_draws"] == 10
    assert meta["burn_in"] == 2
    assert meta["sampler_config"]["proposal_scale"] == 0.2
    assert meta["sampler_config"]["random_state"] == 7

    # posterior auto-extracted from the live result
    assert "estimation/posterior.parquet" in files
    posterior = collapse_columns(
        from_parquet_columns(files["estimation/posterior.parquet"])
    )
    np.testing.assert_allclose(posterior["samples"], mcmc.samples)


def test_facade_rejects_unknown_source() -> None:
    with pytest.raises(TypeError, match="EstimationSpec or Estimator"):
        BundleBuilder().add_estimation(object())  # type: ignore[arg-type]
