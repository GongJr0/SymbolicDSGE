"""Serialization for Monte Carlo pipeline results.

Lifted out of ``SymbolicDSGE.ui.mc`` so the result wire format is reusable by the
``.sdsge`` bundle without depending on the HTTP layer. ``serialize_pipeline_result``
remains the canonical (unchanged) wire shape consumed by the UI; the bundle path uses
the parquet-friendly split below:

- ``result_document`` -> JSON-safe metadata + summaries (no bulk trace arrays),
- ``result_traces`` -> the bulk numeric trace columns as ndarrays (no I/O here),
- ``pipeline_result_wire`` -> re-merges the two back into the UI wire shape (hydration).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..regression.ols import OLSResult
from .mc_constructs import MCContext, MCPipelineResult
from .postproc import Artifact, Raw, Summary, normalize_artifacts
from .traces import regression_trace_keys, test_trace_keys

#: Scalar artifact-value types that ride the JSON document inline; ndarray values
#: go to the parquet side-channel, anything else is a tabular artifact (#181).
_SCALAR_TYPES = (bool, int, float, str, np.integer, np.floating, np.bool_)


def _normalized_postproc(result: MCPipelineResult) -> dict[str, Artifact]:
    """POSTPROC artifacts, keyed for the wire.

    ``result.postproc`` holds each op's *plain* return value (float / ndarray /
    DataFrame / mapping). Normalize on arrival: a mapping fans out into
    ``{"<step>.<key>": artifact}``, a bare value wraps under the step name
    (ndarray -> ``Raw``, else -> ``Summary``). Downstream serialization keeps its
    ``Summary``/``Raw`` isinstance dispatch unchanged.
    """
    out: dict[str, Artifact] = {}
    for name, value in result.postproc.items():
        out.update(normalize_artifacts(value, name))
    return out


def serialize_pipeline_result(
    result: MCPipelineResult, *, run_id: str
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "kind": "mc",
        "n_rep": result.n_rep,
        "payloads_retained": result.meta.payloads_retained,
        "test_results_retained": result.meta.test_results_retained,
        "contexts_retained": result.meta.contexts_retained,
        "n_successful": result.n_successful,
        "succeeded": result.succeeded,
        "elapsed_s": result.meta.elapsed_s,
        "it_s": result.meta.it_s,
        "step_elapsed_s": dict(result.meta.step_elapsed_s),
        "step_it_s": dict(result.meta.step_it_s),
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
                "n": summary.n,
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
        "data_summaries": _summarize_context_data(result.contexts or ()),
        "postproc": {
            name: _serialize_artifact(artifact)
            for name, artifact in _normalized_postproc(result).items()
        },
    }


# Bulk trace keys stripped from the JSON document and carried as ndarray columns.
_TEST_TRACE_KEYS = ("statistic_trace", "pval_trace", "status_trace")
_REGRESSION_TRACE_KEYS = ("coef_trace", "r2_trace", "status_trace")


def traces_from_summaries(
    test_summaries: Mapping[str, Any],
    regression_summaries: Mapping[str, Any],
) -> dict[str, NDArray[Any]]:
    """Bulk numeric trace columns from the test/regression summaries (no I/O).

    Keys: per test ``"test.<name>.{statistic,pval,status}"``; per regression
    ``"regression.<name>.{coef,r2,status}"`` (``coef`` is 2D ``n_rep x k``). The
    single source of truth for trace shaping — shared by :func:`result_traces`
    (the wire) and the post-loop ``OpType.POSTPROC`` trace registry.
    """
    traces: dict[str, NDArray[Any]] = {}
    for name, test_summary in test_summaries.items():
        keys = test_trace_keys(name)
        traces[keys["statistic"]] = np.asarray(
            test_summary.statistic_trace, dtype=np.float64
        )
        traces[keys["pval"]] = np.asarray(test_summary.pval_trace, dtype=np.float64)
        traces[keys["status"]] = np.asarray(
            [int(status) for status in test_summary.status_trace], dtype=np.int64
        )
    for name, reg_summary in regression_summaries.items():
        keys = regression_trace_keys(name)
        traces[keys["coef"]] = np.asarray(reg_summary.coef_trace, dtype=np.float64)
        traces[keys["r2"]] = np.asarray(reg_summary.r2_trace, dtype=np.float64)
        traces[keys["status"]] = np.asarray(
            [int(status) for status in reg_summary.status_trace], dtype=np.int64
        )
    return traces


def result_traces(result: MCPipelineResult) -> dict[str, NDArray[Any]]:
    """Bulk numeric trace columns for a later parquet writer (no I/O)."""
    return traces_from_summaries(result.test_summaries, result.regression_summaries)


def _artifact_array(artifact: Raw | Summary) -> NDArray[Any] | None:
    """The ndarray a POSTPROC artifact carries as bulk data, or ``None``.

    ``Raw`` is always bulk; a ``Summary`` is bulk only when its value is an
    ndarray (a scalar Summary rides the JSON document inline).
    """
    if isinstance(artifact, Raw):
        return np.asarray(artifact.value)
    if isinstance(artifact, Summary) and isinstance(artifact.value, np.ndarray):
        return artifact.value
    return None


def _serialize_artifact(artifact: Raw | Summary) -> dict[str, Any]:
    """Wire entry for one POSTPROC artifact (arrays inlined for the UI wire).

    Scalars live inline; ndarrays carry a ``shape`` + inlined ``value`` (the
    ``value`` is stripped by :func:`result_document` and re-merged from the
    parquet side-channel). Tabular/DataFrame artifacts are out of scope (#181).
    """
    if isinstance(artifact, Raw):
        arr = np.asarray(artifact.value)
        return {
            "kind": "raw",
            "artifact": "array",
            "shape": list(arr.shape),
            "value": _json_value(arr),
        }
    entry: dict[str, Any] = {
        "kind": "summary",
        "title": artifact.title,
        "render": artifact.render,
    }
    value = artifact.value
    if isinstance(value, np.ndarray):
        entry["artifact"] = "array"
        entry["shape"] = list(value.shape)
        entry["value"] = _json_value(value)
    elif value is None or isinstance(value, _SCALAR_TYPES):
        entry["artifact"] = "scalar"
        entry["value"] = _json_value(value)
    else:
        frame = _artifact_frame(artifact)
        if frame is None:
            raise TypeError(
                f"POSTPROC Summary value of type {type(value).__name__!r} is not "
                "a scalar, ndarray, or DataFrame and cannot be serialized."
            )
        entry.update(_frame_to_wire(frame))  # bulk `data` stripped by result_document
    return entry


def result_postproc_arrays(result: MCPipelineResult) -> dict[str, NDArray[Any]]:
    """The bulk ndarray POSTPROC artifacts, keyed by artifact name (no I/O).

    Unlike :func:`result_traces` (uniform ``R``-length columns), these are
    arbitrary-shape *payloads* (e.g. a KDE ``N x 2`` curve), each serialized to
    its own shape-manifest parquet member by the bundle builder.
    """
    out: dict[str, NDArray[Any]] = {}
    for name, artifact in _normalized_postproc(result).items():
        arr = _artifact_array(artifact)
        if arr is not None:
            out[name] = arr
    return out


_INDEX_COL = "__index__"  # reserved data key carrying a labeled index's values


def _artifact_frame(artifact: Raw | Summary) -> Any | None:
    """The pandas DataFrame a ``Summary`` carries as a tabular artifact, else None."""
    if not isinstance(artifact, Summary):
        return None
    import pandas as pd

    value = artifact.value
    return value if isinstance(value, pd.DataFrame) else None


def _wire_dtype(dtype: Any) -> str:
    """Normalize a pandas dtype to a wire dtype tag.

    Only ``numeric`` (``int``/``float``), ``bool``, and ``string`` are carried;
    everything else (object, categorical, datetime) round-trips as ``string`` —
    dictionary-encoded Parquet collapses repeated category labels anyway.
    """
    import pandas as pd

    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    return "string"


def _frame_column_cells(series: Any, wire_dtype: str) -> list[Any]:
    """A column's cells as JSON-safe scalars (NaN/NaT/None -> ``None``)."""
    import pandas as pd

    if wire_dtype == "string":
        return [None if pd.isna(v) else str(v) for v in series.tolist()]
    if wire_dtype == "bool":
        return [bool(v) for v in series.to_numpy()]
    return [_json_value(v) for v in series.to_numpy()]


def _is_default_range_index(index: Any) -> bool:
    import pandas as pd

    return (
        isinstance(index, pd.RangeIndex)
        and index.start == 0
        and index.step == 1
        and index.name is None
    )


def _frame_to_columns(frame: Any) -> dict[str, list[Any]]:
    """The table's columnar payload (the bulk ``data``), index column first.

    A labeled (non-default-range) index is carried as the reserved
    :data:`_INDEX_COL` column; a default ``RangeIndex`` is metadata-only.
    """
    data: dict[str, list[Any]] = {}
    if not _is_default_range_index(frame.index):
        data[_INDEX_COL] = _frame_column_cells(
            frame.index.to_series(), _wire_dtype(frame.index.dtype)
        )
    for col in frame.columns:
        data[str(col)] = _frame_column_cells(frame[col], _wire_dtype(frame[col].dtype))
    return data


def _frame_to_wire(frame: Any) -> dict[str, Any]:
    """Tabular artifact wire entry: metadata + inlined columnar ``data``."""
    labeled = not _is_default_range_index(frame.index)
    dtypes = {str(col): _wire_dtype(frame[col].dtype) for col in frame.columns}
    if labeled:
        dtypes[_INDEX_COL] = _wire_dtype(frame.index.dtype)
    return {
        "artifact": "table",
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": [str(col) for col in frame.columns],
        "dtypes": dtypes,
        "index": {
            "kind": "labeled" if labeled else "range",
            "name": frame.index.name,
        },
        "data": _frame_to_columns(frame),
    }


def result_postproc_tables(result: MCPipelineResult) -> dict[str, dict[str, list[Any]]]:
    """The columnar payloads of tabular POSTPROC artifacts, keyed by name (no I/O).

    Each table is mixed-dtype, so — unlike :func:`result_postproc_arrays` (float
    payloads) — it rides the columnar NDJSON parquet seam (``frame_to_json`` +
    ``to_parquet``), one member per table.
    """
    out: dict[str, dict[str, list[Any]]] = {}
    for name, artifact in _normalized_postproc(result).items():
        frame = _artifact_frame(artifact)
        if frame is not None:
            out[name] = _frame_to_columns(frame)
    return out


def result_document(result: MCPipelineResult, *, run_id: str = "") -> dict[str, Any]:
    """JSON-safe metadata + summaries with the bulk trace arrays removed.

    Pairs with :func:`result_traces`; recombine via :func:`pipeline_result_wire`.
    """
    document = serialize_pipeline_result(result, run_id=run_id)
    for entry in document["test_summaries"].values():
        for key in _TEST_TRACE_KEYS:
            entry.pop(key, None)
    for entry in document["regression_summaries"].values():
        for key in _REGRESSION_TRACE_KEYS:
            entry.pop(key, None)
    for entry in document["postproc"].values():
        artifact = entry.get("artifact")
        if artifact == "array":  # bulk -> shape-manifest parquet member
            entry.pop("value", None)
        elif artifact == "table":  # bulk -> columnar parquet member
            entry.pop("data", None)
    return document


def pipeline_result_wire(
    document: dict[str, Any],
    traces: dict[str, NDArray[Any]],
    postproc_arrays: Mapping[str, NDArray[Any]] | None = None,
    postproc_tables: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
) -> dict[str, Any]:
    """Re-merge a trace-free :func:`result_document` with :func:`result_traces`
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
    tables = postproc_tables or {}
    wire = copy.deepcopy(document)
    for name, entry in wire.get("postproc", {}).items():
        artifact = entry.get("artifact")
        if artifact == "array":
            arr = arrays.get(name)
            if arr is not None:
                entry["value"] = _json_value(arr)
            else:
                shape = tuple(int(d) for d in entry.get("shape", []))
                entry["value"] = _json_value(np.full(shape, np.nan)) if shape else None
        elif artifact == "table":
            entry["data"] = _table_data_or_null(entry, tables.get(name, {}))
        # scalar artifacts keep their inline value from the document
    for name, entry in wire["test_summaries"].items():
        n = int(entry.get("n", 0))
        entry["statistic_trace"] = _trace_or_null(traces, f"test.{name}.statistic", n)
        entry["pval_trace"] = _trace_or_null(traces, f"test.{name}.pval", n)
        entry["status_trace"] = _status_trace(traces, f"test.{name}.status")
    for name, entry in wire["regression_summaries"].items():
        n_rep = int(entry.get("n_rep", 0))
        k = int(entry.get("k", 0))
        coef = traces.get(f"regression.{name}.coef")
        entry["coef_trace"] = (
            _json_value(coef)
            if coef is not None
            else [[None] * k for _ in range(n_rep)]
        )
        entry["r2_trace"] = _trace_or_null(traces, f"regression.{name}.r2", n_rep)
        entry["status_trace"] = _status_trace(traces, f"regression.{name}.status")
    return wire


def _table_data_or_null(
    entry: Mapping[str, Any], columns: Mapping[str, Sequence[Any]]
) -> dict[str, list[Any]]:
    """Rebuild a table's columnar ``data`` from its decoded parquet columns.

    A column that was all-null is dropped by the Parquet encoder; it is rebuilt
    as ``n`` nulls (matching the trace-column convention). A labeled index rides
    the reserved :data:`_INDEX_COL` column.
    """
    n = int(entry.get("shape", [0, 0])[0])
    keys = list(entry.get("columns", []))
    if entry.get("index", {}).get("kind") == "labeled":
        keys = [_INDEX_COL, *keys]
    return {key: list(columns[key]) if key in columns else [None] * n for key in keys}


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
    if all(isinstance(item, OLSResult) for item in summary.results):
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


def _summarize_context_data(contexts: Sequence[MCContext]) -> dict[str, Any]:
    arrays: dict[str, list[np.ndarray]] = {}
    for context in contexts:
        if context.data is None:
            continue
        data = context.data
        if data.states is not None:
            arrays.setdefault("states", []).append(np.asarray(data.states))
        if data.observables is not None:
            arrays.setdefault("observables", []).append(np.asarray(data.observables))
        for name, value in data.raw.items():
            if name != "_X":
                arrays.setdefault(f"raw:{name}", []).append(np.asarray(value))
    return {name: _array_collection_summary(values) for name, values in arrays.items()}


def _array_collection_summary(values: Sequence[np.ndarray]) -> dict[str, Any]:
    n_total = sum(int(arr.size) for arr in values)
    n_finite = 0
    value_sum = 0.0
    square_sum = 0.0
    minimum = np.inf
    maximum = -np.inf
    for arr in values:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        n_finite += int(finite.size)
        value_sum += float(finite.sum())
        square_sum += float(np.square(finite).sum())
        minimum = min(minimum, float(finite.min()))
        maximum = max(maximum, float(finite.max()))
    if n_finite == 0:
        return {
            "n_rep": len(values),
            "shape": list(values[0].shape),
            "n_values": n_total,
            "n_finite": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    mean = value_sum / n_finite
    variance = max(0.0, square_sum / n_finite - mean**2)
    return {
        "n_rep": len(values),
        "shape": list(values[0].shape),
        "n_values": n_total,
        "n_finite": n_finite,
        "mean": _json_float(mean),
        "std": _json_float(variance**0.5),
        "min": _json_float(minimum),
        "max": _json_float(maximum),
    }


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return _json_float(value)
    return value
