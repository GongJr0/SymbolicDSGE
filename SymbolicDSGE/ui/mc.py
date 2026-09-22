"""HTTP-facing Monte-Carlo adapters.

The catalogue, graph validation, and pipeline compilation live in the core
:mod:`SymbolicDSGE.monte_carlo` package (UI-independent). This module is the
JSON boundary and nothing else: a request body arrives as untrusted data and
leaves as library objects, which is why nothing here returns a document.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo.core import MCPipeline
from SymbolicDSGE.monte_carlo.mc_constructs import MCPipelineResult, MCStep
from SymbolicDSGE.monte_carlo.postproc import builtin_postproc
from SymbolicDSGE.monte_carlo.spec import PipelineMeta, StepMeta, StepSpec
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


def _step_meta(entry: Mapping[str, Any]) -> StepMeta:
    """One authored step's data half, taken field by field.

    Field by field rather than cast, so a key the client invented has nowhere to
    land. ``code`` is dropped here: a step's source is not one of its kwargs, and
    the step that owns it reads it in :func:`_step`.
    """
    return StepMeta(
        name=entry["name"],
        op_type=entry["op_type"],
        step_type=entry["step_type"],
        kwargs=dict(entry.get("kwargs", {})),
        source_args=list(entry.get("source_args", [])),
        n_retain=int(entry.get("n_retain", -1)),
    )


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


def _step_func(step: StepMeta, code: str = "") -> Any | None:
    """The callable a step runs, or ``None`` when its kind names none.

    A custom op compiles from the source the editor submitted; a built-in
    post-loop op is looked up by kind. Nothing is inferred: a step whose kind
    owns no callable keeps an empty slot, which is what the library expects.
    """
    if step["step_type"] not in ("transform:custom", "postproc:custom"):
        return builtin_postproc(step["step_type"])
    if not code.strip():
        raise ValueError(f"Custom step {step['name']} has no source code.")
    try:
        return _custom_func_class(step["step_type"]).from_source(code)
    except CustomOpValidationError as exc:
        raise ValueError(f"Custom step {step['name']}: {exc}") from exc


def _step(entry: Mapping[str, Any]) -> MCStep:
    """One authored step as a live :class:`MCStep`, its source compiled in place.

    The entry dies here. A step's code arrives on the step that owns it, so
    there is no name-keyed rejoin to do and nothing downstream needs the
    document. :meth:`MCStep.from_spec` is what reads ``kwargs["shocks"]`` back
    into live entries, which is why the meta goes through a :class:`StepSpec`
    rather than into the constructor.
    """
    meta = _step_meta(entry)
    return MCStep.from_spec(
        StepSpec(meta=meta, func=_step_func(meta, entry.get("code") or ""))
    )


def build_pipeline(doc: Mapping[str, Any]) -> MCPipeline:
    """A posted pipeline as a live :class:`MCPipeline`.

    The only place JSON is crossed. The constructor orders the graph and
    validates the sources, so a pipeline that comes back from here is one the
    library has already accepted.
    """
    return MCPipeline(
        replication_steps=[_step(entry) for entry in doc["replication_steps"]],
        postproc_steps=[_step(entry) for entry in doc.get("postproc_steps", [])],
    )


def mc_available_traces(doc: Mapping[str, Any]) -> dict[str, list[str]]:
    """The across-rep trace keys the pipeline's producers will emit.

    Feeds the post-loop trace picker (a ``type="trace"`` field) so a POSTPROC op
    can select which test/regression/transform producer it consumes. Reads the
    steps as data instead of building them, since the picker runs while a
    pipeline is still being authored and may not yet be constructible.
    """
    return {
        "traces": _trace_keys(
            PipelineMeta(
                replication_steps=[
                    _step_meta(entry) for entry in doc["replication_steps"]
                ],
                postproc_steps=[
                    _step_meta(entry) for entry in doc.get("postproc_steps", [])
                ],
            )
        )
    }


def run_pipeline(
    pipeline: MCPipeline,
    *,
    models: Mapping[str, SolvedModel],
    n_rep: int,
    fail_fast: bool,
    n_jobs: int | None = None,
    verbosity: int = 0,
) -> MCPipelineResult:
    """Run a built pipeline against the session's models."""
    return pipeline.run(
        models=models,
        n_rep=n_rep,
        fail_fast=fail_fast,
        n_jobs=n_jobs,
        verbosity=verbosity,
    )
