"""Serialization for Monte Carlo pipeline results.

Lifted out of ``SymbolicDSGE.ui.mc`` so the result wire format is reusable by the
``.sdsge`` bundle without depending on the HTTP layer. Two shapes live here:

- ``serialize_pipeline_result`` -> the single flat JSON document the UI consumes,
- ``serialize_run_meta`` plus one ``serialize_*_results`` per step kind -> the
  bundle's split, each returning a typed meta beside the trace arrays themselves.

Only the meta half is ever JSON, through :func:`json_safe`, and the writer is
what calls it. Traces are handed over as the run's own ndarrays, never copied.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .mc_constructs import MCPipelineResult
from .postproc import Artifact, Summary, Raw
from .._ckernels.monte_carlo._arenas import resolve_retention
from .._diag_tests.result import MCTestResult
from ..regression.ols.ols_result import MCRegressionResult
from .spec import (
    MCFailureSpec,
    MCPostprocResultMeta,
    MCRegressionResultMeta,
    MCRunMeta,
    MCTestResultMeta,
    MCTransformResultMeta,
)

NDF = NDArray[np.float64]


def serialize_pipeline_result(
    result: MCPipelineResult, *, run_id: str
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "kind": "mc",
        "n_rep": result.n_rep,
        "n_retained_by_step": dict(result.meta.n_retained_by_step),
        "n_successful": result.n_successful,
        "succeeded": result.succeeded,
        "elapsed_s": result.meta.elapsed_s,
        "it_s": result.meta.it_s,
        "step_elapsed_s": dict(result.meta.step_elapsed_s),
        "step_it_s": dict(result.meta.step_it_s),
        "step_worker_it_s": dict(result.meta.step_worker_it_s),
        "step_wall_it_s": dict(result.meta.step_wall_it_s),
        "step_counts": dict(result.meta.step_counts),
        "step_failures": dict(result.meta.step_failures),
        "postproc_elapsed_s": dict(result.meta.postproc_elapsed_s),
        "failures": [
            {
                "rep_idx": failure.rep_idx,
                "step_name": failure.step_name,
                "error_type": failure.error_type,
                "message": failure.message,
            }
            for failure in result.failures
        ],
        "test_summaries": {
            name: {
                "test_name": summary.test_name,
                "n_rep": summary.n_rep,
                "n_retained": summary.n_retained,
                "retained_reps": json_safe(summary.retained_reps),
                "alpha": float(summary.alpha),
                "distribution": summary.dist.value,
                "df": json_safe(summary.df),
                "pval_method": summary.pval_method.value,
                "mean_statistic": float(summary.mean_statistic),
                "mean_pval": float(summary.mean_pval),
                "rejection_rate": float(summary.rejection_rate),
                "statistic_se": _json_float(summary.statistic_se),
                "pval_se": _json_float(summary.pval_se),
                "statistic_ci": json_safe(summary.statistic_confidence_interval()),
                "rejection_ci": json_safe(summary.pval_confidence_interval()),
                "statistic_trace": json_safe(summary.statistic_trace),
                "pval_trace": json_safe(summary.pval_trace),
                "status_trace": [int(status) for status in summary.status_trace],
                "status_counts": _status_counts(summary.status_trace),
                "statistic_summary": _trace_summary(summary.statistic_trace),
                "pval_summary": _trace_summary(summary.pval_trace),
            }
            for name, summary in result.test_summaries.items()
        },
        "regression_summaries": {
            name: _serialize_regression_summary(summary)
            for name, summary in result.regression_summaries.items()
        },
        "postproc": {
            name: _serialize_artifact(artifact)
            for name, artifact in result.postproc.items()
        },
    }


def _serialize_artifact(artifact: Artifact) -> dict[str, Any]:
    """One step's two slots, each serialized on its own terms.

    ``raw`` is bulk, so it records its shape and the value is stripped to a
    parquet member by :func:`result_document`. ``summary`` is inline whatever it
    holds. Either slot may be absent.
    """
    raw = artifact["raw"]
    summary = artifact["summary"]
    out: dict[str, Any] = {"raw": None, "summary": None}
    if raw is not None:
        arr = np.asarray(raw.value)
        out["raw"] = {"shape": list(arr.shape), "value": json_safe(arr)}
    if summary is not None:
        out["summary"] = {"value": _summary_value(summary.value)}
    return out


def _summary_value(value: Any) -> Any:
    """A ``Summary``'s value as JSON.

    A DataFrame rides pandas' own table schema, which carries its columns,
    dtypes and index and reads back through ``read_json(orient="table")``.
    """
    import pandas as pd

    if isinstance(value, pd.DataFrame):
        return cast(
            dict[str, Any],
            json.loads(value.to_json(orient="table", double_precision=15)),
        )
    return json_safe(value)


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
        postproc_elapsed_s=dict(meta.postproc_elapsed_s),
        failures=[
            MCFailureSpec(
                rep_idx=failure.rep_idx,
                step_name=failure.step_name,
                error_type=failure.error_type,
                message=failure.message,
            )
            for failure in result.failures
        ],
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


def serialize_transform_results(
    transforms: Mapping[str, NDF],
    n_rep: int,
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
        retained_reps, _ = resolve_retention(int(arr.shape[0]), n_rep)
        meta = MCTransformResultMeta(step_name=name, shape=list(arr.shape))
        traces: dict[str, NDArray[Any]] = {
            "value": arr,
            "retained_reps": retained_reps,
        }
        out[name] = (meta, traces)
    return out


def serialize_postproc_results(
    postprocs: Mapping[str, Artifact],
) -> dict[str, tuple[MCPostprocResultMeta, dict[str, NDArray[Any]]]]:
    """Each post-loop step's ``(meta, traces)`` halves.

    The ``summary`` slot is aggregate, so it rides the meta inline. A
    summary-only step contributes no traces at all.
    """
    out: dict[str, tuple[MCPostprocResultMeta, dict[str, NDArray[Any]]]] = {}
    for name, artifact in postprocs.items():
        raw = artifact["raw"]
        summary = artifact["summary"]
        value = np.asarray(raw.value) if raw is not None else None
        meta = MCPostprocResultMeta(
            step_name=name,
            shape=list(value.shape) if value is not None else None,
            summary=_summary_value(summary.value) if summary is not None else None,
        )
        traces: dict[str, NDArray[Any]] = {}
        if value is not None:
            traces["value"] = value
        out[name] = (meta, traces)
    return out


def _serialize_regression_summary(summary: Any) -> dict[str, Any]:
    coefficient_summaries = [
        {
            "variable": variable,
            **_trace_summary(summary.coef_trace[:, index]),
        }
        for index, variable in enumerate(summary.variables)
    ]
    metrics = {
        "r2": _trace_summary(summary.r2_trace),
        "adjusted_r2": _trace_summary(summary.r2_adj_trace),
        "rmse": _trace_summary(summary.rmse_trace),
        "mse": _trace_summary(summary.mse_trace),
        "ssr": _trace_summary(summary.ssr_trace),
    }
    out = {
        "variables": summary.variables,
        "n_rep": summary.n_rep,
        "n_retained": summary.n_retained,
        "retained_reps": json_safe(summary.retained_reps),
        "n": summary.n,
        "k": summary.k,
        "coef_trace": json_safe(summary.coef_trace),
        "r2_trace": json_safe(summary.r2_trace),
        "status_trace": [int(status) for status in summary.status_trace],
        "status_counts": _status_counts(summary.status_trace),
        "coefficient_summaries": coefficient_summaries,
        "metrics": metrics,
        "ols": None,
    }
    if summary.kind == "ols":
        out["ols"] = {
            "mean_standard_errors": json_safe(np.mean(summary.se_trace, axis=0)),
            "mean_t_statistics": json_safe(np.mean(summary.t_stat_trace, axis=0)),
            "mean_pvalues": json_safe(np.mean(summary.pval_trace, axis=0)),
            "mean_partial_r2": json_safe(np.mean(summary.partial_r2_trace, axis=0)),
            "f_statistic": _trace_summary(summary.F_stat_trace),
            "f_pvalue": _trace_summary(summary.F_pval_trace),
        }
    return out


def _status_counts(status_trace: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in status_trace:
        counts[status.name] = counts.get(status.name, 0) + 1
    return counts


def _trace_summary(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "n": int(arr.size),
            "n_finite": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "q025": None,
            "q975": None,
        }
    return {
        "n": int(arr.size),
        "n_finite": int(finite.size),
        "mean": _json_float(finite.mean()),
        "std": _json_float(finite.std()),
        "min": _json_float(finite.min()),
        "max": _json_float(finite.max()),
        "q025": _json_float(np.quantile(finite, 0.025)),
        "q975": _json_float(np.quantile(finite, 0.975)),
    }


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
