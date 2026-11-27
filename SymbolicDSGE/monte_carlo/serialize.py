"""Serialization for Monte Carlo pipeline results."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .mc_constructs import (
    MCDataGenResult,
    MCFilterResult,
    MCPipelineResult,
    MCSmootherResult,
)
from .._diag_tests.result import MCTestResult
from ..regression.result import MCRegressionResult
from .spec import (
    MCDataGenResultMeta,
    MCFilterResultMeta,
    MCRegressionResultMeta,
    MCRunMeta,
    MCTestResultMeta,
    MCTransformResultMeta,
)

NDF = NDArray[np.float64]


def serialize_run_meta(result: MCPipelineResult) -> MCRunMeta:
    """The run's own metadata, with the arguments that produced it.

    The step kinds each carry their own meta; this is what is left over, and it
    is what :meth:`MCPipeline.run` needs beside the models to reproduce the run.
    """
    meta = result.meta
    return MCRunMeta(
        n_rep=result.n_rep,
        n_successful=result.n_successful,
        n_retained_by_step=dict(meta.n_retained_by_step),
        elapsed_s=meta.elapsed_s,
        step_elapsed_s=dict(meta.step_elapsed_s),
        step_counts=dict(meta.step_counts),
        step_failures=dict(meta.step_failures),
        failures=[f.to_meta() for f in result.failures],
        run_config=dict(result.run_config),
    )


def serialize_test_results(
    tests: Mapping[str, MCTestResult],
) -> dict[str, tuple[MCTestResultMeta, dict[str, NDArray[Any]]]]:
    """Each test's ``(meta, traces)`` halves, arrays passed through untouched."""
    out: dict[str, tuple[MCTestResultMeta, dict[str, NDArray[Any]]]] = {}
    for name, test in tests.items():
        spec = test.to_spec()
        traces: dict[str, NDArray[Any]] = {
            "statistic_trace": spec.statistic_trace,
            "status_trace": spec._raw_status,
            "retained_reps": spec.retained_reps,
        }
        out[name] = (spec.meta, traces)
    return out


def serialize_regression_results(
    regressions: Mapping[str, MCRegressionResult],
) -> dict[str, tuple[MCRegressionResultMeta, dict[str, NDArray[Any]]]]:
    """Each regression's ``(meta, traces)`` halves.

    A regression without standard errors omits ``se_trace`` rather than
    carrying a null one.
    """
    out: dict[str, tuple[MCRegressionResultMeta, dict[str, NDArray[Any]]]] = {}
    for name, regression in regressions.items():
        spec = regression.to_spec()
        traces: dict[str, NDArray[Any]] = {
            "coef_trace": spec.coef_trace,
            "ssr_trace": spec.ssr_trace,
            "sst_trace": spec.sst_trace,
            "status_trace": spec._raw_status,
            "retained_reps": spec.retained_reps,
        }
        if spec._se_trace is not None:
            traces["se_trace"] = spec._se_trace
        out[name] = (spec.meta, traces)
    return out


def serialize_datagen_results(
    datagens: Mapping[str, MCDataGenResult],
) -> dict[str, tuple[MCDataGenResultMeta, dict[str, NDArray[Any]]]]:
    """Each datagen's ``(meta, traces)`` halves, keyed by step name.

    Arrays are handed over as the run's own, and the metadata records their shapes.
    """
    out = {}
    for name, datagen in datagens.items():
        fields: dict[str, NDArray[Any]] = {
            "retained_reps": datagen.retained_reps,
            "states": datagen.X,
            "shocks": datagen.eps,
        }
        if datagen.y.shape[-1]:
            fields["observables"] = datagen.y
        meta = MCDataGenResultMeta(
            n_rep=int(datagen.n_rep),
            n_retained=int(datagen.n_retained),
            var_names=list(datagen.var_names),
            shock_names=list(datagen.shock_names),
            observable_names=list(datagen.observable_names),
            shapes={name: list(arr.shape) for name, arr in fields.items()},
        )
        out[name] = (meta, fields)
    return out


def _serialize_filter_result(
    result: MCFilterResult,
) -> tuple[MCFilterResultMeta, dict[str, NDArray[Any]]]:
    fields: dict[str, NDArray[Any]] = {
        "retained_reps": result.retained_reps,
        "x_pred": result.x_pred,
        "x_filt": result.x_filt,
        "y_pred": result.y_pred,
        "y_filt": result.y_filt,
        "P_pred": result.P_pred,
        "P_filt": result.P_filt,
        "S": result.S,
        "eps_hat": result.eps_hat,
        "innov": result.innov,
        "std_innov": result.std_innov,
        "loglik": result.loglik,
    }
    if result.mode == "unscented":
        # Write through the property to narrow the type.
        # Populate private fields to restore exact behavior on load
        fields["_x1_pred"] = result.x1_pred
        fields["_x1_filt"] = result.x1_filt
        fields["_x2_pred"] = result.x2_pred
        fields["_x2_filt"] = result.x2_filt

    meta = MCFilterResultMeta(
        n_rep=int(result.n_rep),
        n_retained=int(result.n_retained),
        filter_mode=result.mode,
        shapes={name: list(arr.shape) for name, arr in fields.items()},
    )
    return (meta, fields)


def _serialize_smoother_result(
    result: MCSmootherResult,
) -> tuple[MCFilterResultMeta, dict[str, NDArray[Any]]]:
    meta, fwd_fields = _serialize_filter_result(result.filter_result)
    smoother_fields: dict[str, NDArray[Any]] = {
        "x_smooth": result.x_smooth,
        "P_smooth": result.P_smooth,
        "eps_smooth": result.eps_smooth,
    }
    if result.mode == "unscented":
        smoother_fields["_x1_smooth"] = result.x1_smooth
        smoother_fields["_x2_smooth"] = result.x2_smooth
    fields = {**fwd_fields, **smoother_fields}
    meta["shapes"] = {
        **meta["shapes"],
        **{name: list(arr.shape) for name, arr in smoother_fields.items()},
    }

    return (meta, fields)


def serialize_filter_results(
    filters: Mapping[str, MCFilterResult],
) -> dict[str, tuple[MCFilterResultMeta, dict[str, NDArray[Any]]]]:
    out = {}
    for name, result in filters.items():
        out[name] = _serialize_filter_result(result)
    return out


def serialize_smoother_results(
    smoothers: Mapping[str, MCSmootherResult],
) -> dict[str, tuple[MCFilterResultMeta, dict[str, NDArray[Any]]]]:
    out = {}
    for name, result in smoothers.items():
        out[name] = _serialize_smoother_result(result)
    return out


def serialize_transform_results(
    transforms: Mapping[str, NDF],
) -> dict[str, tuple[MCTransformResultMeta, dict[str, NDArray[Any]]]]:
    """Each transform's ``(meta, traces)`` halves.

    A payload is ``(n_retained, *output_shape)``, so its shape travels in the
    meta and the retained rep indices travel beside the values. The arenas that
    held those indices do not outlive the run, so they are rebuilt from the
    retained row count through the same routine the run allocated with.
    """
    if not transforms:
        return {}
    out: dict[str, tuple[MCTransformResultMeta, dict[str, NDArray[Any]]]] = {}
    for name, arr in transforms.items():
        meta = MCTransformResultMeta(shape=list(arr.shape))
        traces: dict[str, NDArray[Any]] = {
            "value": arr,
        }
        out[name] = (meta, traces)
    return out


def _json_float(value: Any) -> float | None:
    scalar = float(value)
    return scalar if np.isfinite(scalar) else None


def json_safe(value: Any) -> Any:
    """Recursively convert numpy containers and scalars to JSON-native values.

    The writer-side seam: a serializer hands back its meta typed and its traces
    as arrays, and whoever writes the meta to JSON runs it through here. Traces
    never pass through it.
    """
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, tuple | list):
        return [json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return _json_float(value)
    return value
