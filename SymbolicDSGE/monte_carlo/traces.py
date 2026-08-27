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

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from SymbolicDSGE._diag_tests.result import MCTestResult
from SymbolicDSGE.regression.ols.ols_result import MCRegressionResult

from .catalog import TERMINAL_STEP_TYPES, TRANSFORM_STEP_TYPES
from .mc_constructs import MCStep, OpType
from .spec import PipelineSpec

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import MCPipeline

_TEST_SUBKEYS = ("statistic", "pval", "status")
_REGRESSION_SUBKEYS = ("coef", "ssr", "sst", "se", "r2", "status")


def test_trace_keys(name: str) -> dict[str, str]:
    """Trace keys a test step named ``name`` produces, by sub-channel."""
    return {sub: f"test.{name}.{sub}" for sub in _TEST_SUBKEYS}


def regression_trace_keys(name: str) -> dict[str, str]:
    """Trace keys a regression step named ``name`` produces, by sub-channel."""
    return {sub: f"regression.{name}.{sub}" for sub in _REGRESSION_SUBKEYS}


def payload_trace_key(name: str) -> str:
    """Trace key for a transform's stacked per-rep payload."""
    return f"payload.{name}"


def trace_keys_for(step_type: str, name: str) -> list[str]:
    """The across-rep trace keys a producer of ``step_type`` named ``name`` emits."""
    if step_type == "regression":
        return list(regression_trace_keys(name).values())
    if step_type in TERMINAL_STEP_TYPES:  # remaining terminals are tests
        return list(test_trace_keys(name).values())
    if step_type in TRANSFORM_STEP_TYPES | {"payload", "transform:custom"}:
        return [payload_trace_key(name)]
    return []  # datagen / filter / postproc produce no consumable trace


def trace_keys_for_step(step: MCStep) -> list[str]:
    """The across-rep trace keys a live per-rep step emits, by its role.

    The step counterpart of :func:`trace_keys_for`, dispatching on ``op_type``
    rather than ``step_type``, which a hand-built step may leave unset.
    """
    if step.op_type is OpType.REGRESSION:
        return list(regression_trace_keys(step.name).values())
    if step.op_type is OpType.TEST:
        return list(test_trace_keys(step.name).values())
    if step.op_type is OpType.TRANSFORM:
        return [payload_trace_key(step.name)]
    return []  # datagen / filter / postproc produce no consumable trace


def _trace_keys(spec: PipelineSpec) -> list[str]:
    """Every across-rep trace key the pipeline's producers will emit (in node order).

    The set a POSTPROC op may reference; used to populate the GUI trace picker and
    to validate trace references before a run.
    """
    keys: list[str] = []
    for node in spec["nodes"]:
        keys.extend(trace_keys_for(node["step_type"], node["name"]))
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
