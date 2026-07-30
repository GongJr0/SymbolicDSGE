"""Across-replication trace addressing.

The post-loop (``OpType.POSTPROC``) phase exposes every producer's across-rep
output as a keyed ndarray (a *trace*). This module is the single source of truth
for those key strings and for enumerating, from a pipeline *spec* alone, which
trace keys a run will produce — so a POSTPROC op's trace references can be
offered in the GUI and validated before the (potentially long) run.

Key format:

- tests -> ``test.<name>.{statistic,pval,status}``
- regressions -> ``regression.<name>.{coef,r2,status}``
- transforms -> ``payload.<name>`` (the step's stacked per-rep ndarray output)

``serialize.traces_from_summaries`` and the engine's payload stacking build the
runtime registry from these same primitives, so the static view here can't drift
from what a run actually emits.
"""

from __future__ import annotations

from .catalog import TERMINAL_STEP_TYPES, TRANSFORM_STEP_TYPES
from .spec import PipelineSpec

_TEST_SUBKEYS = ("statistic", "pval", "status")
_REGRESSION_SUBKEYS = ("coef", "r2", "status")


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
    if step_type in TRANSFORM_STEP_TYPES or step_type == "transform:custom":
        return [payload_trace_key(name)]
    return []  # datagen / filter / postproc produce no consumable trace


def available_traces(spec: PipelineSpec) -> list[str]:
    """Every across-rep trace key the pipeline's producers will emit (in node order).

    The set a POSTPROC op may reference; used to populate the GUI trace picker and
    to validate trace references before a run.
    """
    keys: list[str] = []
    for node in spec.nodes:
        keys.extend(trace_keys_for(node.step_type, node.name))
    return keys
