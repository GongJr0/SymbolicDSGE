"""HTTP-facing Monte-Carlo adapters.

The catalogue, graph validation, and pipeline compilation now live in the core
:mod:`SymbolicDSGE.monte_carlo` package (UI-independent). This module is a thin
seam that accepts the pydantic request models and delegates to the core API via
``MCPipelineSpec.to_core()``.
"""

from __future__ import annotations

from typing import Any

from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipelineResult, NodeSpec
from SymbolicDSGE.monte_carlo import build_pipeline as build_pipeline
from SymbolicDSGE.monte_carlo import catalog_payload
from SymbolicDSGE.monte_carlo import run_pipeline as _run_pipeline
from SymbolicDSGE.monte_carlo import validate_pipeline_spec as _validate_pipeline_spec
from SymbolicDSGE.monte_carlo.custom_op import (
    CustomOpValidationError,
    NumpyCustomFunc,
)
from SymbolicDSGE.monte_carlo.serialize import (
    serialize_pipeline_result as serialize_pipeline_result,
)

from .mc_schemas import MCPipelineSpec

#: Pre-fill for the custom-op Monaco editor. numpy is available as ``np`` inside
#: the safe namespace, so no imports are needed (and the validator rejects them).
MC_CUSTOM_OP_TEMPLATE = '''@custom_operation
def transform(*, context, reference, dgp, rep_idx, **kwargs):
    """Custom Monte-Carlo transform. Runs once per replication."""
    # numpy is available as `np` (no imports needed). Read this replication's
    # data; `context.require_data()` returns the current MCData:
    #   .states (T x k) / .observables (T x m) / .raw (dict) / .observable_names
    data = context.require_data()
    arr = np.asarray(data.observables, dtype=float)
    # Return a 2-D ndarray (T x k). It is stored under this step's name, so a
    # downstream step can read it with source="payload".
    return (arr - arr.mean(axis=0)) / arr.std(axis=0)
'''


def mc_catalog() -> dict[str, Any]:
    """The step catalogue payload served at ``/api/mc/catalog``."""
    return catalog_payload()


def mc_custom_op_template() -> dict[str, str]:
    """The starter source served to the custom-op editor."""
    return {"template": MC_CUSTOM_OP_TEMPLATE}


def validate_custom_op(code: str) -> dict[str, Any]:
    """Validate a single custom-op source for live editor feedback.

    Returns ``{"valid": True, "name": ...}`` or ``{"valid": False, "error": ...}``
    (a 200 either way) so the editor can render the message inline.
    """
    try:
        func = NumpyCustomFunc.from_source(code)
    except CustomOpValidationError as exc:
        return {"valid": False, "error": str(exc)}
    return {"valid": True, "name": func.name}


def compile_custom_resources(spec: MCPipelineSpec) -> dict[str, Any]:
    """Compile each ``custom`` node's source into a callable, keyed by node name.

    Feeds ``build_pipeline``/``run_pipeline`` via their ``resources`` seam. Raises
    ``ValueError`` (node-scoped) on missing or invalid source so the validate/run
    endpoints report which step failed.
    """
    resources: dict[str, Any] = {}
    for node in spec.nodes:
        if node.step_type != "custom":
            continue
        code = node.params.get("code", "")
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"Custom step '{node.name}' has no source code.")
        try:
            resources[node.name] = NumpyCustomFunc.from_source(code)
        except CustomOpValidationError as exc:
            raise ValueError(f"Custom step '{node.name}': {exc}") from exc
    return resources


def validate_pipeline_spec(
    spec: MCPipelineSpec,
    *,
    has_reference: bool,
    has_dgp: bool,
) -> list[NodeSpec]:
    """Graph-validate a UI pipeline request and return its ordered core nodes."""
    return _validate_pipeline_spec(
        spec.to_core(), has_reference=has_reference, has_dgp=has_dgp
    )


def run_pipeline(
    spec: MCPipelineSpec,
    *,
    reference: SolvedModel | None,
    dgp: SolvedModel | None,
    n_rep: int,
    fail_fast: bool,
) -> MCPipelineResult:
    """Validate, compile, and run a UI pipeline request (custom ops included)."""
    return _run_pipeline(
        spec.to_core(),
        reference=reference,
        dgp=dgp,
        n_rep=n_rep,
        fail_fast=fail_fast,
        resources=compile_custom_resources(spec),
    )
