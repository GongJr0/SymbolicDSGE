"""HTTP-facing Monte-Carlo adapters.

The catalogue, graph validation, and pipeline compilation now live in the core
:mod:`SymbolicDSGE.monte_carlo` package (UI-independent). This module is a thin
seam that accepts the pydantic request models and delegates to the core API via
``MCPipelineSpec.to_core()``.
"""

from __future__ import annotations

from typing import Any

from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipelineResult, NodeSpec, PostprocSpec
from SymbolicDSGE.monte_carlo import available_traces as _available_traces
from SymbolicDSGE.monte_carlo import build_pipeline as build_pipeline
from SymbolicDSGE.monte_carlo import catalog_payload
from SymbolicDSGE.monte_carlo import run_pipeline as _run_pipeline
from SymbolicDSGE.monte_carlo.custom_op import (
    CustomFunc,
    CustomOpValidationError,
    NumbaCustomFunc,
    PandasCustomFunc,
)
from SymbolicDSGE.monte_carlo.serialize import (
    serialize_pipeline_result as serialize_pipeline_result,
)

from .mc_schemas import MCNodeSpec, MCPipelineSpec, MCPostprocSpec

#: Pre-fill for the custom-op Monaco editor. numpy is available as ``np`` inside
#: the safe namespace, so no imports are needed (and the validator rejects them).
MC_CUSTOM_OP_TEMPLATE = '''@custom_transform
def transform(sample, output):
    """Custom Monte-Carlo transform. Runs once per replication."""
    # `sample` is the selected source array. `output` has the shape declared
    # on this step and must be written in full. Both are 2-D float64 arrays.
    output[:, :] = sample
    return 0
'''


def mc_catalog() -> dict[str, Any]:
    """The step catalogue payload served at ``/api/mc/catalog``."""
    return catalog_payload()


def mc_custom_op_template() -> dict[str, str]:
    """The starter source served to the custom-op editor."""
    return {"template": MC_CUSTOM_OP_TEMPLATE}


def mc_available_traces(spec: MCPipelineSpec) -> dict[str, list[str]]:
    """The across-rep trace keys the pipeline's producers will emit.

    Feeds the post-loop trace picker (a ``type="trace"`` field) so a POSTPROC op
    can select which test/regression/transform producer it consumes.
    """
    return {"traces": _available_traces(spec.to_core())}


def _custom_func_class(step_type: str) -> type[CustomFunc]:
    """The wrapper class for a custom step kind: pandas for post-loop, else numba."""
    return PandasCustomFunc if step_type == "postproc:custom" else NumbaCustomFunc


def validate_custom_op(
    code: str, *, step_type: str = "transform:custom"
) -> dict[str, Any]:
    """Validate a single custom-op source for live editor feedback.

    Returns ``{"valid": True, "name": ...}`` or ``{"valid": False, "error": ...}``
    (a 200 either way) so the editor can render the message inline. ``step_type``
    selects the namespace (``"postproc:custom"`` gets the pandas namespace).
    """
    try:
        func = _custom_func_class(step_type).from_source(code)
    except CustomOpValidationError as exc:
        return {"valid": False, "error": str(exc)}
    return {"valid": True, "name": func.name}


def compile_custom_resources(spec: MCPipelineSpec) -> dict[str, Any]:
    """Compile each ``custom`` node's source into a callable, keyed by node name.

    Feeds ``build_pipeline``/``run_pipeline`` via their ``resources`` seam. The
    namespace is phase-based: ``postproc:custom`` nodes compile under the pandas
    namespace, ``transform:custom`` under Numba. Raises ``ValueError``
    (node-scoped) on missing or invalid source so the validate/run endpoints
    report which step failed.
    """
    resources: dict[str, Any] = {}
    # transform:custom lives in nodes; postproc:custom in postprocs.
    steps: list[MCNodeSpec | MCPostprocSpec] = [*spec.nodes, *spec.postprocs]
    for node in steps:
        if node.step_type not in ("transform:custom", "postproc:custom"):
            continue
        code = node.params.get("code", "")
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"Custom step '{node.name}' has no source code.")
        try:
            resources[node.name] = _custom_func_class(node.step_type).from_source(code)
        except CustomOpValidationError as exc:
            raise ValueError(f"Custom step '{node.name}': {exc}") from exc
    return resources


def run_pipeline(
    spec: MCPipelineSpec,
    *,
    reference: SolvedModel | None,
    dgp: SolvedModel | None,
    n_rep: int,
    fail_fast: bool,
    n_jobs: int | None = None,
    verbosity: int = 0,
) -> MCPipelineResult:
    """Validate, compile, and run a UI pipeline request (custom ops included)."""
    return _run_pipeline(
        spec.to_core(),
        reference=reference,
        dgp=dgp,
        n_rep=n_rep,
        fail_fast=fail_fast,
        n_jobs=n_jobs,
        verbosity=verbosity,
        resources=compile_custom_resources(spec),
    )
