from __future__ import annotations

import inspect
from typing import Any, Mapping

import numpy as np

from SymbolicDSGE.bayesian.distributions.param_builder import DIST_PARAMS_DISPATCH
from SymbolicDSGE.bayesian.transforms.transform_dispatch import (
    TRANSFORM_METHOD_DISPATCH,
)
from SymbolicDSGE.estimation.results import MCMCResult, OptimizationResult
from SymbolicDSGE.estimation.spec import (
    EstimationParameterSpec as CoreEstimationParameterSpec,
)
from SymbolicDSGE.estimation.spec import (
    EstimationSpec,
    MCMCResultMeta,
    OptimizationResultMeta,
)

from .schemas import EstimationParameterSpec


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
    """Lower UI estimation parameters to Estimator inputs.

    Thin adapter: converts the pydantic request models to the core
    :class:`~SymbolicDSGE.estimation.spec.EstimationSpec` and delegates the
    compilation to :meth:`EstimationSpec.to_estimator_inputs`. The empty-selection
    guard keeps the GUI-facing message.
    """
    if not any(parameter.estimate for parameter in parameters):
        raise ValueError("Select at least one parameter to estimate.")

    spec = EstimationSpec(
        method=method,
        parameters=[
            CoreEstimationParameterSpec.from_dict(parameter.model_dump())
            for parameter in parameters
        ],
    )
    inputs = spec.to_estimator_inputs()
    return inputs.estimated_params, inputs.theta0, inputs.priors, inputs.bounds


def emit_estimation_wire(
    obj: OptimizationResult | OptimizationResultMeta | MCMCResult | MCMCResultMeta,
    *,
    traces: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render an estimation result (live or loaded-bundle metadata) to the wire dict.

    Single source of truth for the estimation tab's wire shape. Accepts:

    - a live :class:`OptimizationResult` / :class:`MCMCResult` (in-process path);
    - or an :class:`OptimizationResultMeta` / :class:`MCMCResultMeta` from a
      loaded ``.sdsge`` bundle (repaint path). For MCMC, the bundle path must
      supply ``traces`` carrying the ``samples`` (2-D) and ``logpost_trace``
      / ``logpost`` (1-D) arrays decoded from the Parquet member.

    Dispatches by direct ``isinstance`` so mypy can narrow within each branch;
    the meta dataclasses and live result classes overlap perfectly on the
    scalar slice each helper reads.
    """
    if isinstance(obj, (OptimizationResult, OptimizationResultMeta)):
        return _emit_optimization_wire(obj)
    if isinstance(obj, (MCMCResult, MCMCResultMeta)):
        return _emit_mcmc_wire(obj, traces)
    raise TypeError(f"Unsupported estimation result type: {type(obj).__name__}")


def _emit_optimization_wire(
    obj: OptimizationResult | OptimizationResultMeta,
) -> dict[str, Any]:
    return {
        "kind": obj.kind,
        "success": bool(obj.success),
        "message": obj.message,
        "theta": {name: float(value) for name, value in obj.theta.items()},
        "fun": float(obj.fun),
        "loglik": float(obj.loglik),
        "logprior": float(obj.logprior),
        "logpost": float(obj.logpost),
        "nfev": int(obj.nfev),
        "nit": obj.nit,
    }


def _emit_mcmc_wire(
    obj: MCMCResult | MCMCResultMeta,
    traces: Mapping[str, Any] | None,
) -> dict[str, Any]:
    samples_src = getattr(obj, "samples", None)
    logpost_src = getattr(obj, "logpost_trace", None)
    if samples_src is None and traces is not None:
        samples_src = traces.get("samples")
    if logpost_src is None and traces is not None:
        # Bundle authoring convention uses "logpost" (natural column name);
        # the live MCMCResult class exposes "logpost_trace". Accept either.
        logpost_src = traces.get("logpost_trace", traces.get("logpost"))
    if samples_src is None or logpost_src is None:
        raise ValueError(
            "MCMC wire emission requires 'samples' and 'logpost_trace'/'logpost' "
            "— supply them on the object or via the 'traces' mapping."
        )
    samples = np.asarray(samples_src, dtype=np.float64)
    logpost = np.asarray(logpost_src, dtype=np.float64)
    return {
        "kind": "mcmc",
        "param_names": list(obj.param_names),
        "posterior_mean": {
            name: float(samples[:, index].mean())
            for index, name in enumerate(obj.param_names)
        },
        "posterior_std": {
            name: float(samples[:, index].std())
            for index, name in enumerate(obj.param_names)
        },
        "samples": {
            name: samples[:, index].tolist()
            for index, name in enumerate(obj.param_names)
        },
        "logpost_trace": logpost.tolist(),
        "accept_rate": float(obj.accept_rate),
        "n_draws": int(obj.n_draws),
        "burn_in": int(obj.burn_in),
        "thin": int(obj.thin),
        "logpost_mean": float(logpost.mean()),
        "logpost_min": float(logpost.min()),
        "logpost_max": float(logpost.max()),
    }


def serialize_estimation_result(result: Any) -> dict[str, Any]:
    """Backwards-compatible thin wrapper around :func:`emit_estimation_wire`."""
    return emit_estimation_wire(result)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value
