"""Serialization for Monte Carlo pipeline results.

Lifted out of ``SymbolicDSGE.ui.mc`` so the result wire format is reusable by the
``.sdsge`` bundle without depending on the HTTP layer. ``serialize_pipeline_result``
remains the canonical (unchanged) wire shape consumed by the UI; the bundle path uses
the parquet-friendly split below:

- ``result_document`` -> JSON-safe metadata + summaries (no bulk trace arrays),
- ``run_traces`` -> the bulk numeric trace columns as ndarrays (no I/O here),
- ``pipeline_result_wire`` -> re-merges the two back into the UI wire shape (hydration).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from SymbolicDSGE.monte_carlo.core import MCPipeline

from .mc_constructs import MCPipelineResult
from .postproc import Artifact
from .traces import regression_trace_keys, test_trace_keys, payload_trace_key


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
                "retained_reps": _json_value(summary.retained_reps),
                "alpha": float(summary.alpha),
                "distribution": summary.dist.value,
                "df": _json_value(summary.df),
                "pval_method": summary.pval_method.value,
                "mean_statistic": float(summary.mean_statistic),
                "mean_pval": float(summary.mean_pval),
                "rejection_rate": float(summary.rejection_rate),
                "statistic_se": _json_float(summary.statistic_se),
                "pval_se": _json_float(summary.pval_se),
                "statistic_ci": _json_value(summary.statistic_confidence_interval()),
                "rejection_ci": _json_value(summary.pval_confidence_interval()),
                "statistic_trace": _json_value(summary.statistic_trace),
                "pval_trace": _json_value(summary.pval_trace),
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


# Bulk trace keys stripped from the JSON document and carried as ndarray columns.
_TEST_TRACE_KEYS = ("statistic_trace", "pval_trace", "status_trace")
_REGRESSION_TRACE_KEYS = ("coef_trace", "r2_trace", "status_trace")


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
        out["raw"] = {"shape": list(arr.shape), "value": _json_value(arr)}
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
    return _json_value(value)


def result_postproc_arrays(result: MCPipelineResult) -> dict[str, NDArray[Any]]:
    """Each step's bulk ``Raw`` array, keyed by step name.

    Unlike :func:`run_traces` (uniform ``R``-length columns), these are
    arbitrary-shape payloads (e.g. a KDE ``N x 2`` curve), each serialized to
    its own shape-manifest parquet member by the bundle builder.
    """
    return {
        name: np.asarray(artifact["raw"].value)
        for name, artifact in result.postproc.items()
        if artifact["raw"] is not None
    }


def result_document(result: MCPipelineResult, *, run_id: str = "") -> dict[str, Any]:
    """JSON-safe metadata + summaries with the bulk trace arrays removed.

    Pairs with :func:`run_traces`; recombine via :func:`pipeline_result_wire`.
    """
    document = serialize_pipeline_result(result, run_id=run_id)
    for entry in document["test_summaries"].values():
        for key in _TEST_TRACE_KEYS:
            entry.pop(key, None)
    for entry in document["regression_summaries"].values():
        for key in _REGRESSION_TRACE_KEYS:
            entry.pop(key, None)
    for entry in document["postproc"].values():
        if entry["raw"] is not None:  # bulk -> shape-manifest parquet member
            entry["raw"].pop("value", None)
    return document


def pipeline_result_wire(
    document: dict[str, Any],
    traces: dict[str, NDArray[Any]],
    postproc_arrays: Mapping[str, NDArray[Any]] | None = None,
) -> dict[str, Any]:
    """Re-merge a trace-free :func:`result_document` with :func:`run_traces`
    (and :func:`result_postproc_arrays`) into the canonical UI wire shape (used
    for hydration).

    A float trace column that is degenerate across *every* replication (e.g. a
    test that returns an undefined-variance NaN statistic in all reps) is dropped
    by the Parquet encoder, since an all-null column carries no values. Such a
    column is reconstructed here as a null-filled trace of the summary's length —
    which is exactly what the canonical wire reports for an all-NaN trace — so
    hydration stays robust instead of raising ``KeyError`` on the missing key.
    The same null-from-``shape`` fallback applies to a dropped POSTPROC array.
    """
    arrays = postproc_arrays or {}
    wire = copy.deepcopy(document)
    for name, entry in wire.get("postproc", {}).items():
        raw = entry["raw"]
        if raw is None:  # a summary-only step keeps its inline value
            continue
        arr = arrays.get(name)
        if arr is not None:
            raw["value"] = _json_value(arr)
        else:
            shape = tuple(int(d) for d in raw.get("shape", []))
            raw["value"] = _json_value(np.full(shape, np.nan)) if shape else None
    for name, entry in wire["test_summaries"].items():
        n = int(entry.get("n_retained", entry.get("n_rep", 0)))
        entry["statistic_trace"] = _trace_or_null(traces, f"test.{name}.statistic", n)
        entry["pval_trace"] = _trace_or_null(traces, f"test.{name}.pval", n)
        entry["status_trace"] = _status_trace(traces, f"test.{name}.status")
    for name, entry in wire["regression_summaries"].items():
        n_retained = int(entry.get("n_retained", entry.get("n_rep", 0)))
        k = int(entry.get("k", 0))
        coef = traces.get(f"regression.{name}.coef")
        entry["coef_trace"] = (
            _json_value(coef)
            if coef is not None
            else [[None] * k for _ in range(n_retained)]
        )
        entry["r2_trace"] = _trace_or_null(traces, f"regression.{name}.r2", n_retained)
        entry["status_trace"] = _status_trace(traces, f"regression.{name}.status")
    return wire


def _trace_or_null(traces: dict[str, NDArray[Any]], key: str, n: int) -> list[Any]:
    """A trace column as a JSON-safe list, or ``n`` nulls if it was dropped."""
    arr = traces.get(key)
    if arr is not None:
        return cast(list[Any], _json_value(arr))
    return [None] * n


def _status_trace(traces: dict[str, NDArray[Any]], key: str) -> list[int]:
    """Status traces are integer-valued and never all-null, so a missing column
    only occurs for an empty run; fall back to an empty list."""
    arr = traces.get(key)
    return [int(x) for x in arr] if arr is not None else []


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
        "retained_reps": _json_value(summary.retained_reps),
        "n": summary.n,
        "k": summary.k,
        "coef_trace": _json_value(summary.coef_trace),
        "r2_trace": _json_value(summary.r2_trace),
        "status_trace": [int(status) for status in summary.status_trace],
        "status_counts": _status_counts(summary.status_trace),
        "coefficient_summaries": coefficient_summaries,
        "metrics": metrics,
        "ols": None,
    }
    if summary.kind == "ols":
        out["ols"] = {
            "mean_standard_errors": _json_value(np.mean(summary.se_trace, axis=0)),
            "mean_t_statistics": _json_value(np.mean(summary.t_stat_trace, axis=0)),
            "mean_pvalues": _json_value(np.mean(summary.pval_trace, axis=0)),
            "mean_partial_r2": _json_value(np.mean(summary.partial_r2_trace, axis=0)),
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


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return _json_float(value)
    return value


def run_traces(result: MCPipelineResult) -> dict[str, NDArray]:
    """The retained across-rep columns a bundle stores, keyed as traces.

    Narrower than the registry a post-loop op receives
    (:func:`SymbolicDSGE.monte_carlo.traces.traces_from_summaries`): ``pval`` and
    ``r2`` are recomputed from the columns beside them, so they are not written.
    ``se`` appears only where the regression carries one. POSTPROC artifacts are
    not traces; they ride their own shape-manifest members.
    """
    test_summaries = result.test_summaries
    regression_summaries = result.regression_summaries
    payload_columns: Mapping[str, NDArray] = result.transform_outputs or {}

    traces: dict[str, NDArray] = {}
    for name, test_summary in test_summaries.items():
        keys = test_trace_keys(name)
        traces[keys["statistic"]] = test_summary.statistic_trace
        traces[keys["status"]] = test_summary._raw_status

    for name, reg_summary in regression_summaries.items():
        keys = regression_trace_keys(name)
        traces[keys["coef"]] = reg_summary.coef_trace
        traces[keys["ssr"]] = reg_summary.ssr_trace
        traces[keys["sst"]] = reg_summary.sst_trace
        traces[keys["status"]] = reg_summary._raw_status

        if reg_summary._se_trace is not None:
            traces[keys["se"]] = reg_summary._se_trace

    for name, arr in (payload_columns or {}).items():
        traces[payload_trace_key(name)] = arr
    return traces
