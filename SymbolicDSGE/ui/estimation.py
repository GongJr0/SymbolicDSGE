from __future__ import annotations

import inspect
from typing import Any, Mapping, Sequence

import numpy as np

from ..bayesian.distributions.param_builder import DIST_PARAMS_DISPATCH
from ..bayesian.transforms.transform_dispatch import (
    TRANSFORM_METHOD_DISPATCH,
)
from ..bayesian.priors import Prior, make_prior
from ..core.compiled_model import CompiledModel
from ..estimation.backend import extract_base_params
from ..estimation.results import MLEResult, MAPResult, MCMCResult, OptimizationResult
from ..estimation.spec import EstimatorSpec

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
        # The two the native driver implements; anything else raises there.
        "optimizer_methods": ["L-BFGS-B", "Nelder-Mead"],
        "posterior_points": ["mean", "map", "last"],
    }


def build_estimation_inputs(
    parameters: list[EstimationParameterSpec],
    *,
    routine: str,
) -> tuple[
    list[str],
    dict[str, float],
    dict[str, Prior] | None,
    list[tuple[float | None, float | None]] | None,
]:
    """Lower the GUI parameter table to ``Estimator`` arguments.

    Selects the rows the user ticked and splits them into the four arguments a
    run takes: the estimated names, their starting values, the built priors
    (MAP/MCMC only, where every selected row must carry one), and the bounds.
    ``bounds`` stays ``None`` unless some row sets one, matching what
    :meth:`SymbolicDSGE.core.solver.DSGESolver.estimate` expects. The GUI never
    offers the reserved matrix keys, so every row here is a scalar parameter.
    """
    active = [parameter for parameter in parameters if parameter.estimate]
    if not active:
        raise ValueError("Select at least one parameter to estimate.")

    names = [parameter.name for parameter in active]
    if len(set(names)) != len(names):
        raise ValueError("Estimated parameter names must be unique.")

    theta0 = {parameter.name: float(parameter.initial) for parameter in active}

    bounds = [(parameter.lower, parameter.upper) for parameter in active]
    bound_arg = (
        bounds
        if any(low is not None or high is not None for low, high in bounds)
        else None
    )

    priors: dict[str, Prior] | None = None
    if routine in {"map", "mcmc"}:
        priors = {}
        for parameter in active:
            if parameter.prior is None:
                raise ValueError(
                    f"Parameter '{parameter.name}' requires a prior for "
                    f"{routine.upper()}."
                )
            priors[parameter.name] = make_prior(
                distribution=parameter.prior.distribution,
                parameters=dict(parameter.prior.parameters),
                transform=parameter.prior.transform,
                transform_kwargs=dict(parameter.prior.transform_kwargs),
            )

    return names, theta0, priors, bound_arg


def emit_estimation_wire(
    obj: MLEResult | MAPResult | MCMCResult,
    *,
    traces: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render an estimation result to the wire dict.

    Single source of truth for the estimation tab's wire shape, shared by the
    in-process run and a result rebuilt from a ``.sdsge`` bundle, which arrives
    as the same live class. ``traces`` remains accepted so an MCMC result whose
    bulk columns were read separately can supply ``samples`` and the
    ``logpost_trace``/``logpost`` and ``logjac_trace``/``logjac`` arrays.

    Dispatches by direct ``isinstance`` so mypy can narrow within each branch.
    """
    if isinstance(obj, MLEResult):
        return _emit_mle_wire(obj)
    if isinstance(obj, MAPResult):
        return _emit_map_wire(obj)
    if isinstance(obj, MCMCResult):
        return _emit_mcmc_wire(obj, traces)
    raise TypeError(f"Unsupported estimation result type: {type(obj).__name__}")


def _emit_optimization_wire(
    obj: OptimizationResult,
) -> dict[str, Any]:
    return {
        "success": bool(obj.success),
        "message": obj.message,
        "theta": {name: float(value) for name, value in obj.theta.items()},
        "fun": float(obj.fun),
        "nfev": int(obj.nfev),
        "nit": obj.nit,
        "se": _emit_se(obj.se),
        "cov_status": int(obj.cov_status),
    }


def _emit_se(se: Mapping[str, Any] | None) -> dict[str, float | None] | None:
    """Standard errors as JSON, a non-finite entry rendered as ``null``.

    A covariance that failed, and a negative variance on the diagonal of one
    that did not, both leave NaN in place rather than a status. The response
    encoder rejects NaN, so it becomes ``null`` and reads as "unavailable"
    alongside ``cov_status``.
    """
    if se is None:
        return None
    return {
        name: (float(value) if np.isfinite(value) else None)
        for name, value in se.items()
    }


def _emit_mle_wire(obj: MLEResult) -> dict[str, Any]:
    optim = _emit_optimization_wire(obj)
    return optim | {"loglik": float(obj.loglik)}


def _emit_map_wire(obj: MAPResult) -> dict[str, Any]:
    optim = _emit_optimization_wire(obj)
    return optim | {"logpost": float(obj.logpost), "logprior": float(obj.logprior)}


def _emit_mcmc_wire(
    obj: MCMCResult,
    traces: Mapping[str, Any] | None,
) -> dict[str, Any]:
    samples_src = getattr(obj, "samples", None)
    logpost_src = getattr(obj, "logpost_trace", None)
    logjac_src = getattr(obj, "logjac_trace", None)

    if samples_src is None and traces is not None:
        samples_src = traces.get("samples")
    if logpost_src is None and traces is not None:
        # Bundle authoring convention uses "logpost" (natural column name);
        # the live MCMCResult class exposes "logpost_trace". Accept either.
        logpost_src = traces.get("logpost_trace", traces.get("logpost"))
    if logjac_src is None and traces is not None:
        logjac_src = traces.get("logjac_trace", traces.get("logjac"))
    if samples_src is None or logpost_src is None or logjac_src is None:
        raise ValueError(
            "MCMC wire emission requires 'samples', 'logpost_trace'/'logpost', "
            "and 'logjac_trace'/'logjac'. Supply them on the object or via the "
            "'traces' mapping."
        )
    samples = np.asarray(samples_src, dtype=np.float64)
    logpost = np.asarray(logpost_src, dtype=np.float64)
    logjac = np.asarray(logjac_src, dtype=np.float64)
    return {
        "kind": "mcmc",
        "param_names": list(obj.param_names),
        "posterior_mean": {
            name: float(samples[:, index].mean())
            for index, name in enumerate(obj.param_names)
        },
        "samples": {
            name: samples[:, index].tolist()
            for index, name in enumerate(obj.param_names)
        },
        "logpost_trace": logpost.tolist(),
        "logjac_trace": logjac.tolist(),
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


def _method_from_result(
    result: MLEResult | MAPResult | MCMCResult | None,
) -> str | None:
    """The estimation method (``mle``/``map``/``mcmc``) implied by a result."""
    if result is None:
        return None
    if isinstance(result, MCMCResult):
        return "mcmc"
    if isinstance(result, MLEResult):
        return "mle"
    if isinstance(result, MAPResult):
        return "map"
    raise TypeError(f"Unsupported estimation result type: {type(result).__name__}")


def _bounds_from_result(
    result: MLEResult | MAPResult | MCMCResult | None,
    param_names: Sequence[str],
) -> dict[str, tuple[float | None, float | None]]:
    """Per-parameter bounds recorded on an optimization result (empty otherwise)."""
    if isinstance(result, OptimizationResult):
        raw = result.optimizer_config.get("bounds")
        if raw:
            return {name: (pair[0], pair[1]) for name, pair in zip(param_names, raw)}
    return {}


def estimator_spec_wire(spec: EstimatorSpec) -> dict[str, Any]:
    """An :class:`EstimatorSpec` as JSON, verbatim.

    The dataclass holds a plain list and a TypedDict, so this is a shape
    change and nothing else. It stays free of anything the GUI added, which
    is what lets a bundle take this slot as it stands.
    """
    return {"y": spec.y, "params": dict(spec.params)}


def build_estimation_prefill(
    spec: EstimatorSpec,
    result: MLEResult | MAPResult | MCMCResult | None,
    compiled: CompiledModel,
) -> dict[str, Any]:
    """Seed the estimation form from a bundle's stored run.

    The GUI's parameter table spans every calibration parameter, ticked or not,
    which no single stored artifact carries: the spec names only what was
    estimated and holds the priors, the model supplies the full roster and each
    row's starting value, and the run's optimizer config is where bounds were
    recorded. This assembles the three into the shape the run request posts back,
    observed data included, so the tab repaints without re-running.
    """
    params = spec.params
    estimated = list(params["estimated_params"] or [])
    priors = dict(params["priors"] or {})
    bounds = _bounds_from_result(result, estimated)
    base = extract_base_params(compiled)

    rows: list[dict[str, Any]] = []
    for name, value in base.items():
        low, high = bounds.get(name, (None, None))
        rows.append(
            {
                "name": name,
                "estimate": name in estimated,
                "initial": float(value),
                "lower": low,
                "upper": high,
                "prior": priors.get(name),
            }
        )

    # Reserved matrix keys are block targets, not calibration parameters, so they
    # have no row of their own and ride alongside the table.
    matrix_priors = {
        target: prior for target, prior in priors.items() if target not in base
    }

    return {
        "method": _method_from_result(result) or "mle",
        "y": spec.y,
        "observables": params["observables"],
        "parameters": rows,
        "matrix_priors": matrix_priors or None,
        "ss_seed": params["ss_seed"],
    }
