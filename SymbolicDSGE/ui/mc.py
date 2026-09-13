"""HTTP-facing Monte-Carlo adapters.

The catalogue, graph validation, and pipeline compilation now live in the core
:mod:`SymbolicDSGE.monte_carlo` package (UI-independent). This module is a thin
seam that accepts the pydantic request models and delegates to the core API via
``MCPipelineSpec.to_core()``.
"""

from __future__ import annotations

from typing import Any

from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo.core import MCPipeline
from SymbolicDSGE.monte_carlo.mc_constructs import MCPipelineResult
from SymbolicDSGE.monte_carlo.postproc import builtin_postproc
from SymbolicDSGE.monte_carlo.spec import PipelineSpec, StepMeta, StepSpec
from SymbolicDSGE.monte_carlo.traces import _trace_keys
from SymbolicDSGE.monte_carlo.custom_op import (
    CustomFunc,
    CustomOpValidationError,
    NumbaCustomFunc,
    PandasCustomFunc,
)
from SymbolicDSGE.monte_carlo.serialize import (
    serialize_pipeline_result as serialize_pipeline_result,
)

from .mc_schemas import MCPipelineSpec, MCStepSpec

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


def mc_custom_op_template() -> dict[str, str]:
    """The starter source served to the custom-op editor."""
    return {"template": MC_CUSTOM_OP_TEMPLATE}


def mc_available_traces(spec: MCPipelineSpec) -> dict[str, list[str]]:
    """The across-rep trace keys the pipeline's producers will emit.

    Feeds the post-loop trace picker (a ``type="trace"`` field) so a POSTPROC op
    can select which test/regression/transform producer it consumes.
    """
    return {"traces": _trace_keys(spec.to_core())}


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


def _step_func(step: MCStepSpec) -> Any | None:
    """The callable a step runs, or ``None`` when its kind names none.

    A custom op compiles from the source the editor submitted; a built-in
    post-loop op is looked up by kind. Nothing is inferred: a step whose kind
    owns no callable keeps an empty slot, which is what the library expects.
    """
    if step.step_type not in ("transform:custom", "postproc:custom"):
        return builtin_postproc(step.step_type)
    code = step.code or ""
    if not code.strip():
        raise ValueError(f"Custom step {step.name!r} has no source code.")
    try:
        return _custom_func_class(step.step_type).from_source(code)
    except CustomOpValidationError as exc:
        raise ValueError(f"Custom step {step.name!r}: {exc}") from exc


def build_pipeline(spec: MCPipelineSpec) -> MCPipeline:
    """Compile a UI pipeline request into a live :class:`MCPipeline`.

    The request is the pipeline as data, which is everything except the
    callables. Pairing each step's meta with the callable its kind implies is
    the last thing left before the library can take it.
    """
    funcs = {step.name: _step_func(step) for step in spec.steps}
    doc = spec.to_core()

    def step_spec(meta: StepMeta) -> StepSpec:
        return StepSpec(meta=meta, func=funcs.get(meta["name"]))

    return MCPipeline.from_spec(
        PipelineSpec(
            replication_steps=[step_spec(m) for m in doc["replication_steps"]],
            postproc_steps=[step_spec(m) for m in doc["postproc_steps"]],
        )
    )


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
    if reference is None:
        raise ValueError("A solved reference model is required.")
    return build_pipeline(spec).run(
        reference=reference,
        dgp=dgp,
        n_rep=n_rep,
        fail_fast=fail_fast,
        n_jobs=n_jobs,
        verbosity=verbosity,
    )
