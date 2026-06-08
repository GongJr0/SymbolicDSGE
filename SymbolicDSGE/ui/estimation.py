from __future__ import annotations

import inspect
from typing import Any, Mapping

import numpy as np

from SymbolicDSGE.bayesian.distributions.param_builder import DIST_PARAMS_DISPATCH
from SymbolicDSGE.bayesian.transforms.transform_dispatch import (
    TRANSFORM_METHOD_DISPATCH,
)
from SymbolicDSGE.estimation.estimator import Estimator
from SymbolicDSGE.estimation.results import MCMCResult, OptimizationResult

from .schemas import EstimationParameterSpec, PriorSpec


def estimation_catalog() -> dict[str, Any]:
    return {
        "distributions": {
            family.value: {
                key: _json_value(value)
                for key, value in defaults.items()
                if key != "random_state"
            }
            for family, defaults in DIST_PARAMS_DISPATCH.items()
            if family.value != "lkj_chol"
        },
        "transforms": {
            method.value: {
                name: (
                    _json_value(param.default)
                    if param.default is not inspect.Parameter.empty
                    else None
                )
                for name, param in inspect.signature(transform).parameters.items()
            }
            for method, transform in TRANSFORM_METHOD_DISPATCH.items()
            if method.value != "cholesky_corr"
        },
        "optimizer_methods": [
            "L-BFGS-B",
            "Nelder-Mead",
            "Powell",
            "BFGS",
            "CG",
            "SLSQP",
        ],
        "posterior_points": ["mean", "map", "last"],
    }


def build_estimation_inputs(
    parameters: list[EstimationParameterSpec],
    *,
    method: str,
) -> tuple[
    list[str],
    dict[str, float],
    dict[str, Any] | None,
    list[tuple[float | None, float | None]] | None,
]:
    active = [parameter for parameter in parameters if parameter.estimate]
    if not active:
        raise ValueError("Select at least one parameter to estimate.")

    names = [parameter.name for parameter in active]
    if len(set(names)) != len(names):
        raise ValueError("Estimated parameter names must be unique.")

    theta0 = {parameter.name: parameter.initial for parameter in active}
    bounds = [(parameter.lower, parameter.upper) for parameter in active]
    bound_arg = (
        bounds
        if any(low is not None or high is not None for low, high in bounds)
        else None
    )

    priors: dict[str, Any] | None = None
    if method in {"map", "mcmc"}:
        priors = {}
        for parameter in active:
            if parameter.prior is None:
                raise ValueError(
                    f"Parameter '{parameter.name}' requires a prior for {method.upper()}."
                )
            priors[parameter.name] = _make_prior(parameter.prior)
    return names, theta0, priors, bound_arg


def serialize_estimation_result(result: Any) -> dict[str, Any]:
    if isinstance(result, OptimizationResult):
        return {
            "kind": result.kind,
            "success": result.success,
            "message": result.message,
            "theta": {name: float(value) for name, value in result.theta.items()},
            "fun": float(result.fun),
            "loglik": float(result.loglik),
            "logprior": float(result.logprior),
            "logpost": float(result.logpost),
            "nfev": result.nfev,
            "nit": result.nit,
        }
    if isinstance(result, MCMCResult):
        samples = np.asarray(result.samples, dtype=np.float64)
        logpost = np.asarray(result.logpost_trace, dtype=np.float64)
        return {
            "kind": "mcmc",
            "param_names": list(result.param_names),
            "posterior_mean": {
                name: float(samples[:, index].mean())
                for index, name in enumerate(result.param_names)
            },
            "posterior_std": {
                name: float(samples[:, index].std())
                for index, name in enumerate(result.param_names)
            },
            "samples": {
                name: samples[:, index].tolist()
                for index, name in enumerate(result.param_names)
            },
            "logpost_trace": logpost.tolist(),
            "accept_rate": float(result.accept_rate),
            "n_draws": result.n_draws,
            "burn_in": result.burn_in,
            "thin": result.thin,
            "logpost_mean": float(logpost.mean()),
            "logpost_min": float(logpost.min()),
            "logpost_max": float(logpost.max()),
        }
    raise TypeError(f"Unsupported estimation result type: {type(result).__name__}")


def _make_prior(spec: PriorSpec) -> Any:
    return Estimator.make_prior(
        distribution=spec.distribution,
        parameters=dict(spec.parameters),
        transform=spec.transform,
        transform_kwargs=dict(spec.transform_kwargs),
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value
