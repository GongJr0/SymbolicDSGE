from __future__ import annotations

import json

import numpy as np
import pytest
from numpy import float64

from SymbolicDSGE.bayesian.priors import make_prior
from SymbolicDSGE.bundle.builder import BundleBuilder
from SymbolicDSGE.bundle.parquet import collapse_columns, from_parquet_columns
from SymbolicDSGE.estimation import Estimator
from SymbolicDSGE.estimation.results import MCMCResult, MAPResult

#: The one estimated parameter, and the observables of the reference model.
PARAM = "rho_u"
OBSERVABLES = ["Infl", "Rate"]


@pytest.fixture
def estimator(compiled_test) -> Estimator:
    """An estimator over the model the bundle carries, with one estimated parameter.

    ``test.yaml`` declares no ``calibration.measurement_noise``, so R is the
    explicit override the constructor requires, and it rides the spec as one.
    """
    return Estimator(
        compiled=compiled_test,
        y=np.zeros((6, len(OBSERVABLES)), dtype=np.float64),
        observables=OBSERVABLES,
        estimated_params={
            PARAM: make_prior(
                distribution="normal",
                parameters={"mean": 0.7, "std": 0.1},
                transform="identity",
                transform_kwargs={},
            )
        },
        R=np.eye(len(OBSERVABLES), dtype=np.float64),
    )


def _optimization_result() -> MAPResult:
    return MAPResult(
        x=np.array([0.71], dtype=np.float64),
        theta={PARAM: float64(0.71)},
        success=True,
        message="ok",
        fun=float64(1.0),
        nfev=12,
        nit=4,
        optimizer_config={
            "method": "L-BFGS-B",
            "bounds": {PARAM: [-1.0, 1.0]},
            "options": {"maxiter": 20},
        },
        logpost=float64(-1.5),
        logprior=float64(-0.5),
    )


def test_facade_flattens_optimization_run(estimator, test_model_yaml) -> None:
    builder = (
        BundleBuilder()
        .add_model("reference", test_model_yaml)
        .add_estimation(estimator, result=_optimization_result())
    )
    _, files = builder.build()

    # the spec carries construction state only
    spec = json.loads(files["estimation/spec.json"])
    assert spec["estimated_params"] == [PARAM]
    assert spec["observables"] == OBSERVABLES
    assert spec["priors"][PARAM]["distribution"] == "normal"  # reversed from live Prior
    assert np.array_equal(spec["R"], np.eye(len(OBSERVABLES)).tolist())
    assert "method" not in spec and "method_kwargs" not in spec

    # the run's own arguments ride the result
    result_doc = json.loads(files["estimation/result.json"])
    assert result_doc["type"] == "map"
    config = result_doc["data"]["optimizer_config"]
    assert config["method"] == "L-BFGS-B"
    assert config["bounds"] == {PARAM: [-1.0, 1.0]}
    assert config["options"]["maxiter"] == 20

    assert "estimation/observed.parquet" in files


def test_facade_flattens_mcmc_run_with_posterior(estimator, test_model_yaml) -> None:
    rng = np.random.default_rng(0)
    mcmc = MCMCResult(
        param_names=[PARAM],
        samples=rng.standard_normal((10, 1)),
        logpost_trace=rng.standard_normal(10),
        logjac_trace=rng.standard_normal(10),
        accept_rate=float64(0.4),
        n_draws=10,
        burn_in=2,
        thin=1,
        sampler_config={"adapt": True, "proposal_scale": 0.2, "random_state": 7},
    )

    builder = (
        BundleBuilder()
        .add_model("reference", test_model_yaml)
        .add_estimation(estimator, result=mcmc)
    )
    _, files = builder.build()

    spec = json.loads(files["estimation/spec.json"])
    assert spec["estimated_params"] == [PARAM]

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
