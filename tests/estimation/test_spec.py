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
        ss_seed=[0.0, 0.0],
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
    assert "ss_seed" not in as_dict


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


def test_from_targets_flags_only_listed_params() -> None:
    spec = EstimationSpec.from_targets(
        ["psi_pi", "psi_x"],
        method="map",
        initial={"psi_pi": 1.5, "psi_x": 0.5},
        priors={
            "psi_pi": PriorSpec(
                distribution="normal", parameters={"loc": 1.5, "scale": 0.25}
            )
        },
        bounds={"psi_pi": (1.0, 3.0)},
        observables=["Infl", "Rate"],
    )

    assert [p.name for p in spec.parameters] == ["psi_pi", "psi_x"]
    assert all(p.estimate for p in spec.parameters)  # never typed by the user
    psi_pi, psi_x = spec.parameters
    assert psi_pi.initial == 1.5
    assert (psi_pi.lower, psi_pi.upper) == (1.0, 3.0)
    assert psi_pi.prior is not None and psi_pi.prior.distribution == "normal"
    assert psi_x.prior is None  # no prior supplied for this target
    assert (psi_x.lower, psi_x.upper) == (None, None)
    assert spec.method == "map"
    assert spec.observables == ["Infl", "Rate"]


def test_from_targets_defaults_initial_to_zero() -> None:
    spec = EstimationSpec.from_targets(["a"])
    assert spec.parameters[0].initial == 0.0
    assert spec.parameters[0].estimate is True


def test_from_targets_round_trips_through_json() -> None:
    spec = EstimationSpec.from_targets(
        ["a", "b"], method="mcmc", initial={"a": 0.1, "b": 0.2}
    )
    restored = EstimationSpec.from_json(spec.to_json())
    assert restored.to_dict() == spec.to_dict()


def test_from_targets_rejects_empty() -> None:
    with pytest.raises(ValueError):
        EstimationSpec.from_targets([])


def test_from_targets_rejects_duplicate_targets() -> None:
    with pytest.raises(ValueError):
        EstimationSpec.from_targets(["a", "a"])


def test_to_estimator_inputs_lowers_targets_priors_and_bounds() -> None:
    spec = EstimationSpec.from_targets(
        ["beta", "rho"],
        method="map",
        initial={"beta": 0.99, "rho": 0.5},
        bounds={"beta": (0.9, 1.0)},
        priors={
            "beta": PriorSpec(
                distribution="normal", parameters={"mean": 0.99, "std": 0.01}
            ),
            "rho": PriorSpec(
                distribution="normal", parameters={"mean": 0.5, "std": 0.1}
            ),
        },
    )

    inputs = spec.to_estimator_inputs()

    assert inputs.estimated_params == ["beta", "rho"]
    assert inputs.theta0 == {"beta": 0.99, "rho": 0.5}
    assert inputs.bounds == [(0.9, 1.0), (None, None)]
    assert inputs.priors is not None and set(inputs.priors) == {"beta", "rho"}
    # priors are built Prior objects, not specs
    assert all(hasattr(prior, "logpdf") for prior in inputs.priors.values())


def test_to_estimator_inputs_skips_non_estimated_and_omits_bounds() -> None:
    spec = EstimationSpec(
        method="mle",
        parameters=[
            EstimationParameterSpec(name="beta", initial=0.99, estimate=True),
            EstimationParameterSpec(name="rho", initial=0.5, estimate=False),
        ],
    )

    inputs = spec.to_estimator_inputs()

    assert inputs.estimated_params == ["beta"]
    assert inputs.priors is None  # MLE needs none
    assert inputs.bounds is None  # no bounds set on the active parameter


def test_to_estimator_inputs_requires_active_parameter() -> None:
    spec = EstimationSpec(
        method="mle",
        parameters=[EstimationParameterSpec(name="beta", initial=0.99, estimate=False)],
    )
    with pytest.raises(ValueError, match="no estimated parameters or matrix priors"):
        spec.to_estimator_inputs()


def test_to_estimator_inputs_requires_prior_for_map() -> None:
    spec = EstimationSpec(
        method="map",
        parameters=[EstimationParameterSpec(name="beta", initial=0.99, estimate=True)],
    )
    with pytest.raises(ValueError, match="requires a prior for MAP"):
        spec.to_estimator_inputs()


def test_optimization_result_meta_round_trips_with_config() -> None:
    meta = OptimizationResultMeta(
        kind="mle",
        theta={"a": 1.0},
        success=True,
        message="",
        fun=0.0,
        loglik=0.0,
        logprior=0.0,
        logpost=0.0,
        nfev=1,
        nit=None,
        optimizer_config={
            "method": "L-BFGS-B",
            "bounds": [[0.0, 1.0]],
            "options": {"maxiter": 10},
        },
    )
    as_dict = meta.to_dict()
    assert as_dict["optimizer_config"]["method"] == "L-BFGS-B"
    assert OptimizationResultMeta.from_dict(as_dict).to_dict() == as_dict


def test_mcmc_result_meta_round_trips_with_config() -> None:
    meta = MCMCResultMeta(
        param_names=["a"],
        accept_rate=0.3,
        n_draws=10,
        burn_in=0,
        thin=1,
        sampler_config={"adapt": True, "proposal_scale": 0.1, "random_state": 7},
    )
    as_dict = meta.to_dict()
    assert as_dict["sampler_config"]["random_state"] == 7
    assert MCMCResultMeta.from_dict(as_dict).to_dict() == as_dict


def _lkj_prior_spec() -> PriorSpec:
    return PriorSpec(
        distribution="lkj_chol",
        parameters={"eta": 2.0, "K": 3},
        transform="cholesky_corr",
        transform_kwargs={"K": 3},
    )


def test_matrix_priors_round_trip() -> None:
    spec = EstimationSpec(
        method="map",
        parameters=[EstimationParameterSpec(name="beta", initial=0.99, estimate=True)],
        matrix_priors={"R_corr": _lkj_prior_spec()},
    )
    as_dict = spec.to_dict()
    assert "R_corr" in as_dict["matrix_priors"]
    assert EstimationSpec.from_dict(as_dict).to_dict() == as_dict
    assert EstimationSpec.from_json(spec.to_json()).to_dict() == as_dict


def test_matrix_only_spec_is_valid() -> None:
    # No scalar parameters, only a block prior — must be accepted.
    spec = EstimationSpec(method="mcmc", matrix_priors={"Q_corr": _lkj_prior_spec()})
    assert spec.parameters == []


def test_to_estimator_inputs_lowers_matrix_priors() -> None:
    spec = EstimationSpec(
        method="map",
        parameters=[
            EstimationParameterSpec(
                name="beta",
                initial=0.99,
                estimate=True,
                prior=PriorSpec(
                    distribution="normal", parameters={"mean": 0.99, "std": 0.1}
                ),
            )
        ],
        matrix_priors={"R_corr": _lkj_prior_spec()},
    )
    inputs = spec.to_estimator_inputs()

    assert inputs.estimated_params == ["beta", "R_corr"]
    assert inputs.priors is not None and set(inputs.priors) == {"beta", "R_corr"}
    assert hasattr(inputs.priors["R_corr"], "logpdf")  # built Prior object
    # matrix members' initials come from calibration -> theta0 deferred entirely
    assert inputs.theta0 is None


def test_matrix_priors_require_bayesian_method() -> None:
    spec = EstimationSpec(method="mle", matrix_priors={"R_corr": _lkj_prior_spec()})
    with pytest.raises(ValueError, match="require method 'map' or 'mcmc'"):
        spec.to_estimator_inputs()


def test_from_targets_accepts_matrix_priors() -> None:
    spec = EstimationSpec.from_targets(
        [],
        method="mcmc",
        matrix_priors={"R_corr": _lkj_prior_spec()},
    )
    assert spec.parameters == []
    assert "R_corr" in spec.matrix_priors


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
        ss_seed=[0.0, 0.0],
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
    assert as_dict["ss_seed"] == [0.0, 0.0]
    assert as_dict["method_kwargs"] == {"options": {"maxiter": 100}}
    assert "compile_kwargs" not in as_dict  # now a model-level concern, dropped
    assert as_dict["parameters"][0]["name"] == "beta"
    assert as_dict["parameters"][0]["prior"]["distribution"] == "beta"
    assert "lower" not in as_dict["parameters"][1]  # unset optional dropped
