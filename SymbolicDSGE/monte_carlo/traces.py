"""Across-replication trace addressing.

The post-loop (``OpType.POSTPROC``) phase exposes every producer's across-rep
output as a keyed ndarray (a *trace*). This module is the single source of truth
for those key strings and for enumerating, from a pipeline *spec* alone, which
trace keys a run will produce — so a POSTPROC op's trace references can be
offered in the GUI and validated before the (potentially long) run.

Key format:

- tests -> ``test.<name>.{statistic,pval,status}``
- regressions -> ``regression.<name>.{coef,ssr,sst,se,r2,status}``
- transforms -> ``payload.<name>`` (the step's stacked per-rep ndarray output)

:func:`traces_from_summaries` builds the registry a post-loop op receives from
these same primitives, so the static view here can't drift from what a run
actually emits. The bundle writes its own projection per step kind, where
``pval`` and ``r2`` are derived from the columns beside them and are not stored.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np
from numpy.typing import NDArray

from SymbolicDSGE._diag_tests.result import MCTestResult
from SymbolicDSGE.regression.result import MCRegressionResult

from .mc_constructs import MCStep, OpType
from .spec import PipelineSpec

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import MCPipeline

_TEST_SUBKEYS = ("statistic", "pval", "status")
_REGRESSION_SUBKEYS = ("coef", "ssr", "sst", "se", "r2", "status")

#: The sub-channels each trace-producing op kind emits, keyed by the prefix its
#: keys carry. A payload is a single array, so it has no sub-channel.
_TRACE_OUTPUTS: dict[str, frozenset[str]] = {
    "test": frozenset(_TEST_SUBKEYS),
    "regression": frozenset(_REGRESSION_SUBKEYS),
    "payload": frozenset(),
}


def test_trace_keys(name: str) -> dict[str, str]:
    """Trace keys a test step named ``name`` produces, by sub-channel."""
    return {sub: f"test.{name}.{sub}" for sub in _TEST_SUBKEYS}


def regression_trace_keys(name: str) -> dict[str, str]:
    """Trace keys a regression step named ``name`` produces, by sub-channel."""
    return {sub: f"regression.{name}.{sub}" for sub in _REGRESSION_SUBKEYS}


def payload_trace_key(name: str) -> str:
    """Trace key for a transform's stacked per-rep payload."""
    return f"payload.{name}"


def trace_keys_for(op_type: str, name: str, kind: str | None = None) -> list[str]:
    """The across-rep trace keys a producer of ``op_type`` named ``name`` emits.

    ``kind`` is the producer's step kind where that narrows its outputs: only an
    OLS regression carries a standard error.
    """
    if op_type == OpType.REGRESSION:
        keys = regression_trace_keys(name)
        if (kind or "ols") != "ols":  # a regression defaults to OLS
            keys.pop("se", None)
        return list(keys.values())
    if op_type == OpType.TEST:
        return list(test_trace_keys(name).values())
    if op_type == OpType.TRANSFORM:
        return [payload_trace_key(name)]
    return []  # datagen / filter / postproc produce no consumable trace


def trace_keys_for_step(step: MCStep) -> list[str]:
    """The across-rep trace keys a live per-rep step emits."""
    return trace_keys_for(step.op_type, step.name, step.kwargs.get("kind"))


def is_trace_ref(value: object) -> bool:
    """Whether a parameter value is spelled like a trace key.

    A postproc names the trace it reads, so a reference is recognized by its own
    spelling rather than by the parameter it was passed under. That covers a
    custom op's references as well as a catalogue op's.
    """
    return isinstance(value, str) and value.split(".", 1)[0] in _TRACE_OUTPUTS


def trace_ref_error(ref: str, available: Collection[str]) -> str | None:
    """Why ``ref`` is unusable as a trace key, or ``None`` when it is fine.

    The producer's name is not checked on its own; it is covered by testing the
    whole key against what the pipeline emits.
    """
    parts = ref.split(".")
    outputs = _TRACE_OUTPUTS[parts[0]]
    if len(parts) != (3 if outputs else 2):
        return f"{ref!r} is not a well-formed {parts[0]} trace key"
    if outputs and parts[2] not in outputs:
        return (
            f"{parts[2]!r} is not an output of a {parts[0]} step "
            f"(one of: {', '.join(sorted(outputs))})"
        )
    if ref not in available:
        return f"no step in the pipeline produces {ref!r}"
    return None


def _trace_keys(spec: PipelineSpec) -> list[str]:
    """Every across-rep trace key the pipeline's producers will emit (in node order).

    The set a POSTPROC op may reference; used to populate the GUI trace picker and
    to validate trace references before a run.
    """
    keys: list[str] = []
    for node in spec["nodes"]:
        keys.extend(
            trace_keys_for(node["op_type"], node["name"], node["params"].get("kind"))
        )
    return keys


def available_traces(pipeline: MCPipeline) -> dict[str, list[str]]:
    """The trace keys each per-rep step in the pipeline will emit.
    Lists the available traces a POSTPROC may refer to.
    """
    return {step.name: trace_keys_for_step(step) for step in pipeline.per_rep_steps}


def traces_from_summaries(
    test_summaries: Mapping[str, MCTestResult],
    regression_summaries: Mapping[str, MCRegressionResult],
) -> dict[str, NDArray]:
    """Bulk numeric trace columns from the test/regression summaries (no I/O).

    Keys: per test ``"test.<name>.{statistic,pval,status}"``; per regression
    ``"regression.<name>.{coef,ssr,sst,se,r2,status}"`` (``coef`` and ``se`` are
    2D ``n_rep x k``). ``se`` appears only where the regression carries one. The
    registry a post-loop ``OpType.POSTPROC`` op receives, and it must agree with
    :func:`available_traces`, which reads the same subkeys. The bundle writes its
    own projection per step kind; see ``SymbolicDSGE.monte_carlo.serialize``.
    """
    traces: dict[str, NDArray] = {}
    for name, test_summary in test_summaries.items():
        keys = test_trace_keys(name)
        traces[keys["statistic"]] = test_summary.statistic_trace
        traces[keys["pval"]] = test_summary.pval_trace
        traces[keys["status"]] = test_summary._raw_status

    for name, reg_summary in regression_summaries.items():
        keys = regression_trace_keys(name)
        traces[keys["coef"]] = reg_summary.coef_trace
        traces[keys["ssr"]] = reg_summary.ssr_trace
        traces[keys["sst"]] = reg_summary.sst_trace
        traces[keys["r2"]] = reg_summary.r2_trace
        traces[keys["status"]] = reg_summary._raw_status
        if reg_summary._se_trace is not None:
            traces[keys["se"]] = reg_summary._se_trace
    return traces
