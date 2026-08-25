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
    """Everything the run produced, in the key set ``*ResultSpec`` reads back.

    ``x`` is the optimum itself and ``vcov`` the covariance ``cov=True`` paid
    for on the way there; both are carried rather than recomputed, and with
    ``optimizer_config`` they complete the set a result is rebuilt from.
    """
    return {
        "x": _emit_vector(obj.x),
        "theta": {name: _emit_scalar(value) for name, value in obj.theta.items()},
        "success": bool(obj.success),
        "message": obj.message,
        "fun": _emit_scalar(obj.fun),
        "nfev": int(obj.nfev),
        "nit": obj.nit,
        "vcov": _emit_matrix(obj.vcov),
        "se": _emit_se(obj.se),
        "cov_status": int(obj.cov_status),
        "optimizer_config": dict(obj.optimizer_config),
    }


def _emit_scalar(value: Any) -> float | None:
    """A float as JSON, ``null`` where it is not finite.

    Non-finite is how the estimator says "unavailable" in place: a covariance
    that failed leaves NaN throughout, a negative diagonal variance leaves one
    behind, and a driver that never ran leaves ``fun`` NaN. The response
    encoder rejects NaN outright, so a single one would cost the whole
    payload; ``null`` reads the same and survives the trip.
    """
    out = float(value)
    return out if np.isfinite(out) else None


def _emit_vector(values: Any) -> list[float | None]:
    return [_emit_scalar(value) for value in np.asarray(values, dtype=np.float64)]


def _emit_matrix(values: Any) -> list[list[float | None]] | None:
    if values is None:
        return None
    return [_emit_vector(row) for row in np.asarray(values, dtype=np.float64)]


def _emit_se(se: Mapping[str, Any] | None) -> dict[str, float | None] | None:
    if se is None:
        return None
    return {name: _emit_scalar(value) for name, value in se.items()}


def _emit_mle_wire(obj: MLEResult) -> dict[str, Any]:
    optim = _emit_optimization_wire(obj)
    return optim | {"loglik": _emit_scalar(obj.loglik)}


def _emit_map_wire(obj: MAPResult) -> dict[str, Any]:
    optim = _emit_optimization_wire(obj)
    return optim | {
        "logpost": _emit_scalar(obj.logpost),
        "logprior": _emit_scalar(obj.logprior),
    }


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
        "accept_rate": _emit_scalar(obj.accept_rate),
        "n_draws": int(obj.n_draws),
        "burn_in": int(obj.burn_in),
        "thin": int(obj.thin),
        # The sampler's own call arguments, the MCMC counterpart of an
        # optimization run's optimizer_config.
        "sampler_config": dict(getattr(obj, "sampler_config", {}) or {}),
        "logpost_mean": _emit_scalar(logpost.mean()),
        "logpost_min": _emit_scalar(logpost.min()),
        "logpost_max": _emit_scalar(logpost.max()),
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


#: View field <- the option key an optimization run recorded it under.
#: ``cov`` and the two ``cov_fd_*`` scalars have no control on a fresh form;
#: they are carried so a bundled run re-runs as it ran.
_OPTIMIZER_VIEW_KNOBS = {
    "maxIter": "maxiter",
    "maxFun": "maxfun",
    "m": "m",
    "maxLs": "maxls",
    "factr": "factr",
    "pgtol": "pgtol",
    "fdStep": "fd_step",
    "xatol": "xatol",
    "fatol": "fatol",
    "cov": "cov",
    "jacobian": "jacobian",
    "covFdStepScale": "cov_fd_step_scale",
    "covFdAbsoluteFloor": "cov_fd_absolute_floor",
}

#: View field <- the argument an MCMC run recorded it under. ``mapOptions``
#: and ``proposalCov`` are the two that cannot be a scalar control.
_SAMPLER_VIEW_KNOBS = {
    "seed": "random_state",
    "proposalScale": "proposal_scale",
    "adapt": "adapt",
    "adaptStart": "adapt_start",
    "adaptEpsilon": "adapt_epsilon",
    "computeMap": "compute_map",
    "mapOptions": "map_options",
    "proposalCov": "proposal_cov",
    "covFdStepScale": "cov_fd_step_scale",
    "covFdAbsoluteFloor": "cov_fd_absolute_floor",
}


def _run_config(
    result: MLEResult | MAPResult | MCMCResult | None,
) -> Mapping[str, Any]:
    """The call arguments a run recorded, whichever routine made it."""
    if isinstance(result, MCMCResult):
        return result.sampler_config or {}
    if isinstance(result, OptimizationResult):
        return result.optimizer_config or {}
    return {}


def _knobs_from_result(
    result: MLEResult | MAPResult | MCMCResult | None,
) -> dict[str, Any]:
    """A run's own settings, in the view's field names.

    A bundle reproduces only if the form re-posts what the run was made with,
    down to the options it renders no control for. A key the run did not
    record is left out, so the form falls back to its own default rather than
    inventing a value the run never used.
    """
    config = _run_config(result)
    if isinstance(result, MCMCResult):
        knobs = {
            view: config[key]
            for view, key in _SAMPLER_VIEW_KNOBS.items()
            if key in config
        }
        # Draw counts live on the result itself, not among the call arguments.
        return knobs | {
            "nDraws": int(result.n_draws),
            "burnIn": int(result.burn_in),
            "thin": int(result.thin),
        }
    options = config.get("options") or {}
    knobs = {
        view: options[key]
        for view, key in _OPTIMIZER_VIEW_KNOBS.items()
        if key in options
    }
    if (method := config.get("method")) is not None:
        knobs["optimizer"] = method
    return knobs


def _theta0_from_result(
    result: MLEResult | MAPResult | MCMCResult | None,
    param_names: Sequence[str],
) -> dict[str, float]:
    """The starting point the run used, per estimated parameter.

    Recorded as a vector ordered like the estimated names, or as the mapping
    it was given as. Without it the form would seed from the model's
    calibration, which is not where the stored run started.
    """
    raw = _run_config(result).get("theta0")
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(name): float(value) for name, value in raw.items()}
    return {name: float(value) for name, value in zip(param_names, raw)}


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
    recorded. This assembles the three into the tab's own view shape, so a
    bundle launch and a client update deliver the same thing to the same slot.

    Only the fields a bundle can speak to are filled; the form merges what
    arrives over its own defaults, so a key a run never recorded is left out
    rather than invented here.
    """
    params = spec.params
    estimated = list(params["estimated_params"] or [])
    priors = dict(params["priors"] or {})
    bounds = _bounds_from_result(result, estimated)
    theta0 = _theta0_from_result(result, estimated)
    base = extract_base_params(compiled)

    rows: list[dict[str, Any]] = []
    for name, value in base.items():
        low, high = bounds.get(name, (None, None))
        rows.append(
            {
                "name": name,
                "estimate": name in estimated,
                # Where the run started, falling back to the model's
                # calibration for a row the run did not estimate.
                "initial": float(theta0.get(name, value)),
                "lower": low,
                "upper": high,
                "prior": priors.get(name),
            }
        )

    # ``ss_seed`` and the reserved matrix keys' priors stay on the spec: the
    # form renders neither, and the view does not restate what it cannot edit.
    observable_names = list(params["observables"] or [])
    return {
        "method": _method_from_result(result) or "mle",
        "parameters": rows,
        "selected": rows[0]["name"] if rows else None,
        "observables": ", ".join(observable_names),
        "dataVectors": _data_vectors(spec.y, observable_names),
        "modeFolded": False,
        **_knobs_from_result(result),
    }


def _data_vectors(
    y: Sequence[Sequence[float]], observables: Sequence[str]
) -> dict[str, str]:
    """Observed data as the per-column text the form's textareas hold.

    The form edits one newline-separated column per observable, which is what
    it posts back as a matrix, so the seed has to arrive already split.
    """
    return {
        name: "\n".join(repr(float(row[index])) for row in y)
        for index, name in enumerate(observables)
    }
