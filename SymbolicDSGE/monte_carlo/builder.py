"""Compile Monte-Carlo pipelines from the core :class:`PipelineSpec`.

Lifted out of ``ui.mc`` so a pipeline can be compiled into an :class:`MCPipeline`
and run without the ``[ui]`` extra. Operates on the pydantic-free core
``TypedDict`` specs; the UI keeps thin wrappers that convert its request models
via ``to_core()``.

Compilation is driven entirely by :data:`SymbolicDSGE.monte_carlo.catalog.STEP_CATALOG`
There is no per-step branching here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .catalog import STEP_CATALOG
from .core import MCPipeline
from .mc_constructs import MCPipelineResult
from .step_factories import (
    add_payload_step,
    postproc_step,
    raw_model_data_step,
    transform_step,
)
from .spec import NodeSpec, PipelineSpec, PostprocSpec

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

    postproc_steps = []
    for pp in spec["postprocs"]:
        step_type = pp["step_type"]
        if step_type == "postproc:custom":
            postproc_steps.append(_build_custom(pp, resources, postproc_step))
        else:
            definition = STEP_CATALOG.get(step_type)
            if definition is None:
                raise ValueError(f"Unsupported MC postproc step type: {step_type}")
            postproc_steps.append(definition.build(pp["name"], dict(pp["params"])))
    return MCPipeline(per_rep_steps, postproc_steps)


def _build_per_rep_step(node: NodeSpec, resources: Mapping[str, Any]) -> Any:
    name = node["name"]
    step_type = node["step_type"]
    params = dict(node["params"])
    n_retain = _pop_n_retain(params, name)
    if step_type == "raw_model_data":
        step = _build_raw_model_data(node, resources, params)
    elif step_type == "payload":
        value = params.pop("value", None)
        if value is None:
            raise ValueError(f"Payload step {name!r} requires a value.")
        step = add_payload_step(name, value)
    elif step_type == "transform:custom":
        step = _build_custom(node, resources, transform_step, params)
    else:
        definition = STEP_CATALOG.get(step_type)
        if definition is None:
            raise ValueError(f"Unsupported MC step type: {step_type}")
        step = definition.build(name, params)
    return replace(step, n_retain=n_retain)


def _build_raw_model_data(
    node: NodeSpec,
    resources: Mapping[str, Any],
    params: dict[str, Any],
) -> Any:
    """Rehydrate a ``raw_model_data`` datagen, injecting its arrays from resources."""
    name = node["name"]
    ref = params.pop("data_ref", name)
    params.pop("data_shapes", None)
    arrays = resources.get(ref)
    if arrays is None:
        raise ValueError(
            f"raw_model_data step '{name}' references data '{ref}' that is not "
            "present in the supplied resources."
        )
    kwargs: dict[str, Any] = {}
    if "states" in arrays:
        kwargs["states"] = arrays["states"]
    if "observables" in arrays:
        kwargs["observables"] = arrays["observables"]
    observable_names = params["observable_names"]
    if observable_names:
        kwargs["observable_names"] = tuple(observable_names)
    return raw_model_data_step(name, **kwargs)


def _build_custom(
    node: NodeSpec | PostprocSpec,
    resources: Mapping[str, Any],
    factory: Any,
    params: dict[str, Any] | None = None,
) -> Any:
    """Rehydrate a custom op, reattaching its callable from resources.

    ``factory`` is the step constructor for the op role (``transform_step`` for a
    ``transform:custom`` node, ``postproc_step`` for a ``postproc:custom`` spec).
    """
    name = node["name"]
    if params is None:
        params = dict(node["params"])
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
    return factory(name, func, **params)


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
