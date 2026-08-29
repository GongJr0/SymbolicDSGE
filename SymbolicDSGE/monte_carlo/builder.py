"""Compile Monte-Carlo pipelines from the core :class:`PipelineSpec`.

Lifted out of ``ui.mc`` so a pipeline can be compiled into an :class:`MCPipeline`
and run without the ``[ui]`` extra. Operates on the pydantic-free core
``TypedDict`` specs; the UI keeps thin wrappers that convert its request models
via ``to_core()``.

A node records its op kind, its step kind, its plain kwargs and its source
bindings, so a step is rebuilt by constructing :class:`MCStep` from them. Only
ops carrying a callable branch: the two custom kinds take theirs from
``resources``, and ``kde`` is the one built-in that runs a library function.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .core import MCPipeline
from .custom_op import NumbaCustomFunc
from .mc_constructs import MCPipelineResult, MCStep, OpType
from .postproc import run_kde
from .spec import OP_TYPES, STEP_KINDS, NodeSpec, PipelineSpec, PostprocSpec
from .spec_compile import restore_kwargs, restore_sources

#: Built-ins whose step runs a library callable rather than a native kernel.
_BUILTIN_FUNCS = {"kde": run_kde}

if TYPE_CHECKING:
    from ..core.solved_model import SolvedModel


def build_pipeline(
    spec: PipelineSpec,
    *,
    resources: Mapping[str, Any] | None = None,
) -> MCPipeline:
    """Compile a spec into a runnable pipeline.

    Nodes are compiled in whatever order the spec lists them, since
    :class:`MCPipeline` sorts them; ``postprocs`` are the post-loop ops (a
    separate terminal phase). ``edges`` are not read. No model is needed: every
    step compiles purely from its parameters (simulation shocks come from the
    explicit registry, not a model), so a pipeline builds before any model is on
    hand. ``resources`` reattaches bulk side-channel data the JSON spec only
    references by key: a ``raw_model_data`` node's arrays (keyed by its
    ``data_ref``) and a ``custom`` op's callable (keyed by its ``func_ref``). The
    bundle loader supplies it; for an all-builtin pipeline it can be omitted.
    """
    resources = resources or {}
    per_rep_steps = [_build_per_rep_step(node, resources) for node in spec["nodes"]]

    postproc_steps = [_build_postproc_step(pp, resources) for pp in spec["postprocs"]]
    return MCPipeline(per_rep_steps, postproc_steps)


def _build_per_rep_step(node: NodeSpec, resources: Mapping[str, Any]) -> MCStep:
    name = node["name"]
    step_type = node["step_type"]
    if step_type not in STEP_KINDS:
        raise ValueError(f"Unsupported MC step type: {step_type}")
    # A node declares its own op kind; it does not get to disagree with what
    # its step kind is.
    if node["op_type"] != OP_TYPES[step_type]:
        raise ValueError(
            f"Step {name!r} declares op_type {node['op_type']!r}, but "
            f"{step_type!r} is {OP_TYPES[step_type]!r}."
        )
    params = dict(node["params"])
    n_retain = _pop_n_retain(params, name)
    func = None

    if step_type == "raw_model_data":
        params = _raw_model_data_kwargs(node, resources, params)
    elif step_type == "payload":
        if params.get("value") is None:
            raise ValueError(f"Payload step {name!r} requires a value.")
    elif step_type == "transform:custom":
        func = NumbaCustomFunc(_resource_callable(node, resources, params))

    return MCStep(
        name=name,
        op_type=OpType(node["op_type"]),
        func=func,
        kwargs=restore_kwargs(step_type, params),
        source_args=restore_sources(node["sources"]),
        step_type=step_type,
        n_retain=n_retain,
    )


def _build_postproc_step(spec: PostprocSpec, resources: Mapping[str, Any]) -> MCStep:
    name = spec["name"]
    step_type = spec["step_type"]
    params = dict(spec["params"])
    if step_type == "postproc:custom":
        func = _resource_callable(spec, resources, params)
    else:
        func = _BUILTIN_FUNCS.get(step_type)
        if func is None:
            raise ValueError(f"Unsupported MC postproc step type: {step_type}")
    return MCStep(
        name=name,
        op_type=OpType.POSTPROC,
        func=func,
        kwargs=params,
        step_type=step_type,
    )


def _raw_model_data_kwargs(
    node: NodeSpec,
    resources: Mapping[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Reattach a ``raw_model_data`` datagen's arrays from resources."""
    name = node["name"]
    ref = params.pop("data_ref", name)
    params.pop("data_shapes", None)
    arrays = resources.get(ref)
    if arrays is None:
        raise ValueError(
            f"raw_model_data step '{name}' references data '{ref}' that is not "
            "present in the supplied resources."
        )
    names = params.get("observable_names") or ()
    return {
        "states": arrays.get("states"),
        "observables": arrays.get("observables"),
        "observable_names": tuple(names),
    }


def _resource_callable(
    node: NodeSpec | PostprocSpec,
    resources: Mapping[str, Any],
    params: dict[str, Any],
) -> Any:
    """Pull a custom op's callable out of resources, consuming its reference."""
    name = node["name"]
    ref = params.pop("func_ref", name)
    # The authoring source rides in ``code`` (compiled into the resources
    # callable upstream); it is not a runtime kwarg of the op.
    params.pop("code", None)
    func = resources.get(ref)
    if func is None:
        raise ValueError(
            f"custom step '{name}' references callable '{ref}' that is not "
            "present in the supplied resources."
        )
    return func


def _pop_n_retain(params: dict[str, Any], step_name: str) -> int:
    value = params.pop("n_retain", -1)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Step {step_name!r} n_retain must be an integer.")
    if value < -1:
        raise ValueError(
            f"Step {step_name!r} n_retain must be -1 (retain all) or non-negative."
        )
    return value


def run_pipeline(
    spec: PipelineSpec,
    *,
    reference: SolvedModel | None,
    dgp: SolvedModel | None,
    n_rep: int,
    fail_fast: bool,
    n_jobs: int | None = None,
    verbosity: int = 0,
    resources: Mapping[str, Any] | None = None,
    check_memory_availability: bool = True,
) -> MCPipelineResult:
    """Compile and run ``spec`` against the reference and DGP models.

    ``resources`` reattaches bulk side-channels the spec references by key
    (``raw_model_data`` arrays, ``custom`` callables); see :func:`build_pipeline`.
    """
    if reference is None:
        raise ValueError("A solved reference model is required.")
    pipeline = build_pipeline(spec, resources=resources)
    return pipeline.run(
        reference=reference,
        dgp=dgp,
        n_rep=n_rep,
        fail_fast=fail_fast,
        n_jobs=n_jobs,
        verbosity=verbosity,
        check_memory_availability=check_memory_availability,
    )
