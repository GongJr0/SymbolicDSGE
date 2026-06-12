from __future__ import annotations

import pytest

from SymbolicDSGE.estimation.spec import (
    EstimationParameterSpec,
    EstimationSpec,
    MCMCResultMeta,
    OptimizationResultMeta,
    PriorSpec,
)


def _baseline_spec() -> EstimationSpec:
    return EstimationSpec(
        method="map",
        parameters=[
            EstimationParameterSpec(
                name="beta",
                initial=0.99,
                estimate=True,
                lower=0.0,
                upper=1.0,
                prior=PriorSpec(
                    distribution="beta",
                    parameters={"alpha": 5.0, "beta": 1.0},
                    transform="identity",
                ),
            ),
            EstimationParameterSpec(name="rho", initial=0.5, estimate=False),
        ],
        observables=["y", "pi"],
        method_kwargs={"options": {"maxiter": 100}},
        compile_kwargs={"linearize": True},
        steady_state=[0.0, 0.0],
        posterior_point="map",
    )


def test_prior_spec_round_trips() -> None:
    prior = PriorSpec(
        distribution="normal",
        parameters={"mu": 0.0, "sigma": 1.0},
        transform="logit",
        transform_kwargs={"low": 0.0, "high": 1.0},
    )
    as_dict = prior.to_dict()
    assert PriorSpec.from_dict(as_dict).to_dict() == as_dict


def test_estimation_parameter_round_trips_with_and_without_prior() -> None:
    with_prior = EstimationParameterSpec(
        name="phi",
        initial=1.5,
        estimate=True,
        lower=0.0,
        prior=PriorSpec(distribution="gamma", parameters={"k": 2.0, "theta": 1.0}),
    )
    without_prior = EstimationParameterSpec(name="rho", initial=0.5)

    for spec in (with_prior, without_prior):
        as_dict = spec.to_dict()
        assert EstimationParameterSpec.from_dict(as_dict).to_dict() == as_dict


def test_estimation_parameter_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        EstimationParameterSpec(name="", initial=0.0)


def test_estimation_spec_round_trips() -> None:
    spec = _baseline_spec()
    as_dict = spec.to_dict()
    assert EstimationSpec.from_dict(as_dict).to_dict() == as_dict
    assert EstimationSpec.from_json(spec.to_json()).to_dict() == as_dict


def test_estimation_spec_omits_unset_optionals() -> None:
    spec = EstimationSpec(
        method="mle",
        parameters=[EstimationParameterSpec(name="a", initial=0.0, estimate=True)],
    )
    as_dict = spec.to_dict()
    assert "observables" not in as_dict
    assert "steady_state" not in as_dict


def test_estimation_spec_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        EstimationSpec(
            method="bogus",
            parameters=[EstimationParameterSpec(name="a", initial=0.0)],
        )


def test_estimation_spec_rejects_unknown_posterior_point() -> None:
    with pytest.raises(ValueError):
        EstimationSpec(
            method="mcmc",
            parameters=[EstimationParameterSpec(name="a", initial=0.0)],
            posterior_point="median",
        )


def test_estimation_spec_rejects_empty_parameters() -> None:
    with pytest.raises(ValueError):
        EstimationSpec(method="mle", parameters=[])


def test_estimation_spec_rejects_duplicate_active_names() -> None:
    with pytest.raises(ValueError):
        EstimationSpec(
            method="mle",
            parameters=[
                EstimationParameterSpec(name="a", initial=0.0, estimate=True),
                EstimationParameterSpec(name="a", initial=1.0, estimate=True),
            ],
        )


def test_estimation_spec_allows_duplicate_inactive_names() -> None:
    # Only estimated names must be unique; inactive duplicates are permitted.
    EstimationSpec(
        method="mle",
        parameters=[
            EstimationParameterSpec(name="a", initial=0.0, estimate=True),
            EstimationParameterSpec(name="a", initial=1.0, estimate=False),
        ],
    )


def test_optimization_result_meta_round_trips() -> None:
    meta = OptimizationResultMeta(
        kind="mle",
        theta={"beta": 0.99, "rho": 0.8},
        success=True,
        message="Optimization terminated successfully.",
        fun=-12.3,
        loglik=-10.0,
        logprior=-2.3,
        logpost=-12.3,
        nfev=42,
        nit=15,
    )
    as_dict = meta.to_dict()
    assert OptimizationResultMeta.from_dict(as_dict).to_dict() == as_dict


def test_optimization_result_meta_nit_optional() -> None:
    meta = OptimizationResultMeta(
        kind="map",
        theta={"a": 1.0},
        success=False,
        message="",
        fun=0.0,
        loglik=0.0,
        logprior=0.0,
        logpost=0.0,
        nfev=1,
        nit=None,
    )
    as_dict = meta.to_dict()
    assert as_dict["nit"] is None
    assert OptimizationResultMeta.from_dict(as_dict).nit is None


def test_mcmc_result_meta_round_trips() -> None:
    meta = MCMCResultMeta(
        param_names=["beta", "rho"],
        accept_rate=0.34,
        n_draws=1000,
        burn_in=200,
        thin=2,
    )
    as_dict = meta.to_dict()
    assert MCMCResultMeta.from_dict(as_dict).to_dict() == as_dict


@pytest.mark.parametrize(
    "kwargs",
    [
        {"param_names": []},
        {"n_draws": 0},
        {"burn_in": -1},
        {"thin": 0},
    ],
)
def test_mcmc_result_meta_validates(kwargs: dict) -> None:
    base = dict(
        param_names=["a"],
        accept_rate=0.3,
        n_draws=10,
        burn_in=0,
        thin=1,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        MCMCResultMeta(**base)


def test_estimation_run_request_to_core_drops_ui_fields() -> None:
    from SymbolicDSGE.ui.schemas import EstimationRunRequest

    request = EstimationRunRequest(
        role="reference",
        method="map",
        y=[[0.1, 0.2], [0.3, 0.4]],
        observables=["y", "pi"],
        parameters=[
            {
                "name": "beta",
                "estimate": True,
                "initial": 0.99,
                "lower": 0.0,
                "upper": 1.0,
                "prior": {
                    "distribution": "beta",
                    "parameters": {"alpha": 5.0, "beta": 1.0},
                },
            },
            {"name": "rho", "estimate": False, "initial": 0.5},
        ],
        method_kwargs={"options": {"maxiter": 100}},
        compile_kwargs={"linearize": True},
        steady_state=[0.0, 0.0],
        posterior_point="map",
        estimate_and_solve=True,
    )

    core = request.to_core()
    as_dict = core.to_dict()

    assert "role" not in as_dict
    assert "y" not in as_dict
    assert "estimate_and_solve" not in as_dict

    assert as_dict["method"] == "map"
    assert as_dict["observables"] == ["y", "pi"]
    assert as_dict["posterior_point"] == "map"
    assert as_dict["steady_state"] == [0.0, 0.0]
    assert as_dict["method_kwargs"] == {"options": {"maxiter": 100}}
    assert as_dict["compile_kwargs"] == {"linearize": True}
    assert as_dict["parameters"][0]["name"] == "beta"
    assert as_dict["parameters"][0]["prior"]["distribution"] == "beta"
    assert "lower" not in as_dict["parameters"][1]  # unset optional dropped
