"""Compile and validate Monte-Carlo pipelines from the core :class:`PipelineSpec`.

Lifted out of ``ui.mc`` so a pipeline can be graph-validated, compiled into an
:class:`MCPipeline`, and run without the ``[ui]`` extra. Operates on the
pydantic-free core dataclasses (:class:`NodeSpec`/:class:`PipelineSpec`); the UI
keeps thin wrappers that convert its request models via ``to_core()``.

Compilation is driven entirely by :data:`SymbolicDSGE.monte_carlo.catalog.STEP_CATALOG`
There is no per-step branching here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .catalog import (
    DATAGEN_STEP_TYPES,
    SourceBinding,
    STEP_CATALOG,
    TERMINAL_STEP_TYPES,
    TRANSFORM_STEP_TYPES,
)
from .core import MCPipeline
from .mc_constructs import MCPipelineResult
from .operations.core import raw_model_data_step
from .operations.postproc import postproc_step
from .operations.transforms import transform_step
from .spec import NodeSpec, PipelineSpec, PostprocSpec
from .traces import available_traces

if TYPE_CHECKING:
    from ..core.solved_model import SolvedModel

#: Transform-role kinds at the spec level: the catalogue transforms plus the
#: user transform custom op (an ``OpType.TRANSFORM`` middle node shipped as a
#: member).
_TRANSFORM_KINDS = TRANSFORM_STEP_TYPES | {"transform:custom"}

#: ``source`` kinds a transform/terminal may legally link from.
_ROOT_SOURCE_TYPES = DATAGEN_STEP_TYPES | {"filter"}


def validate_pipeline_spec(
    spec: PipelineSpec,
    *,
    has_reference: bool,
    has_dgp: bool,
) -> tuple[list[NodeSpec], list[PostprocSpec]]:
    """Validate the pipeline and return ``(ordered per-rep nodes, postprocs)``.

    Enforces unique ids/names (across nodes *and* postprocs), well-formed edges,
    exactly one datagen (``simulation`` or ``raw_model_data``), and the terminal/filter
    linking rules, then orders the per-rep steps. Postprocs are a terminal phase:
    they carry no edges and are validated only for trace references.
    """
    nodes = {node.id: node for node in spec.nodes}
    if len(nodes) != len(spec.nodes):
        raise ValueError("Pipeline node IDs must be unique.")
    names = [node.name for node in spec.nodes] + [pp.name for pp in spec.postprocs]
    if len(set(names)) != len(names):
        raise ValueError("Pipeline step names must be unique.")

    incoming: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    seen_edges: set[tuple[str, str]] = set()
    for edge in spec.edges:
        pair = (edge.source, edge.target)
        if edge.source not in nodes or edge.target not in nodes:
            raise ValueError("Pipeline edges must reference existing nodes.")
        if edge.source == edge.target:
            raise ValueError("Pipeline steps cannot connect to themselves.")
        if pair in seen_edges:
            raise ValueError("Pipeline edges must be unique.")
        source = nodes[edge.source]
        target = nodes[edge.target]
        if source.step_type in TERMINAL_STEP_TYPES:
            raise ValueError(
                f"Terminal step '{source.name}' cannot link to another step."
            )
        if target.step_type in DATAGEN_STEP_TYPES:
            raise ValueError("The datagen step cannot have an incoming link.")
        if target.step_type == "filter" and source.step_type not in DATAGEN_STEP_TYPES:
            raise ValueError("Filter steps must link directly from the datagen.")
        if (
            target.step_type in _TRANSFORM_KINDS
            and source.step_type not in _ROOT_SOURCE_TYPES | _TRANSFORM_KINDS
        ):
            raise ValueError(
                f"Transform '{target.name}' must link from the datagen, a "
                "filter, or another transform."
            )
        if (
            target.step_type in TERMINAL_STEP_TYPES
            and source.step_type not in _ROOT_SOURCE_TYPES | _TRANSFORM_KINDS
        ):
            raise ValueError(
                "Tests and regressions must link from the datagen, a filter, "
                "or a transform."
            )
        seen_edges.add(pair)
        outgoing[edge.source].append(edge.target)
        incoming[edge.target].append(edge.source)

    datagens = [node for node in spec.nodes if node.step_type in DATAGEN_STEP_TYPES]
    if len(datagens) != 1:
        raise ValueError("Pipeline supports exactly one datagen step.")
    datagen = datagens[0]
    if incoming[datagen.id]:
        raise ValueError("The datagen step cannot have an incoming link.")
    for node in spec.nodes:
        if node.id == datagen.id:
            continue
        if node.step_type in TERMINAL_STEP_TYPES and outgoing[node.id]:
            raise ValueError(f"Terminal step '{node.name}' cannot link forward.")

    if not has_reference:
        raise ValueError("A solved reference model is required.")
    if datagen.step_type == "simulation":
        target_is_dgp = (
            "target" not in datagen.params or str(datagen.params["target"]) == "dgp"
        )
        if target_is_dgp and not has_dgp:
            raise ValueError("A solved DGP model is required by the simulation step.")

    filter_nodes = [node for node in spec.nodes if node.step_type == "filter"]
    # Ordering deps come from edges and explicit source step references.
    name_to_id = {node.name: node.id for node in spec.nodes}
    dep_ids = {
        node.id: set(incoming[node.id]) | _source_dep_ids(node, name_to_id)
        for node in spec.nodes
    }
    transform_nodes = _topological_transforms(
        [node for node in spec.nodes if node.step_type in _TRANSFORM_KINDS],
        dep_ids,
        placed_ids={datagen.id, *(node.id for node in filter_nodes)},
    )
    terminal_nodes = [
        node for node in spec.nodes if node.step_type in TERMINAL_STEP_TYPES
    ]
    # Post-loop ops run after everything, over the assembled across-rep traces.
    # They are a separate list (spec.postprocs), never graph nodes.
    _validate_postproc_trace_refs(spec.postprocs, available_traces(spec))
    ordered = [
        datagen,
        *filter_nodes,
        *transform_nodes,
        *terminal_nodes,
    ]

    bound: list[NodeSpec] = []
    bound_by_id: dict[str, NodeSpec] = {}
    prior_names: set[str] = set()
    for node in ordered:
        parents = [
            bound_by_id[parent_id]
            for parent_id in incoming[node.id]
            if parent_id in bound_by_id
        ]
        bound_node = _bind_graph_dependency(node, parents, datagen)
        _validate_dependency(bound_node, prior_names)
        bound.append(bound_node)
        bound_by_id[node.id] = bound_node
        prior_names.add(bound_node.name)
    return bound, list(spec.postprocs)


def _validate_postproc_trace_refs(
    postproc_nodes: Sequence[PostprocSpec], registry: Sequence[str]
) -> None:
    """Validate that each POSTPROC step's declared ``trace`` references resolve.

    Catalogue postproc steps mark trace-selector fields with ``type == "trace"``;
    their values must name a trace the pipeline produces. Custom postproc ops
    reference traces in opaque code, so they're not statically validated.
    """
    available = set(registry)
    for node in postproc_nodes:
        definition = STEP_CATALOG.get(node.step_type)
        if definition is None:
            continue
        for field in definition.fields:
            if field.type != "trace":
                continue
            if field.key not in node.params:
                raise ValueError(
                    f"POSTPROC step '{node.name}' must select a trace for "
                    f"'{field.key}' (available: {sorted(available)})."
                )
            ref = node.params[field.key]
            if not ref:
                raise ValueError(
                    f"POSTPROC step '{node.name}' must select a trace for "
                    f"'{field.key}' (available: {sorted(available)})."
                )
            if ref not in available:
                raise ValueError(
                    f"POSTPROC step '{node.name}' field '{field.key}' references "
                    f"trace {ref!r}, which no step in the pipeline produces "
                    f"(available: {sorted(available)})."
                )


def _source_dep_ids(node: NodeSpec, name_to_id: Mapping[str, str]) -> set[str]:
    """Producer node ids a node references by explicit source name."""
    out: set[str] = set()
    for binding in _source_bindings(node):
        if binding.source_key not in node.params:
            continue
        producer = node.params[binding.source_key]
        if producer and producer in name_to_id:
            out.add(name_to_id[producer])
    return out


def _topological_transforms(
    transforms: Sequence[NodeSpec],
    deps_by_id: Mapping[str, set[str]],
    *,
    placed_ids: set[str],
) -> list[NodeSpec]:
    """Order transform nodes so each comes after all of its producers.

    Transforms can chain, and a transform/custom op may depend on several
    producers. We Kahn-walk the subset, placing a node only once every
    dependency is already ordered.
    """
    remaining = list(transforms)
    ordered: list[NodeSpec] = []
    placed = set(placed_ids)
    while remaining:
        progress = False
        next_remaining: list[NodeSpec] = []
        for node in remaining:
            if all(dep_id in placed for dep_id in deps_by_id[node.id]):
                ordered.append(node)
                placed.add(node.id)
                progress = True
            else:
                next_remaining.append(node)
        if not progress:
            stuck = [node.name for node in next_remaining]
            raise ValueError(
                f"Transform dependency cycle or unresolved parent among {stuck}."
            )
        remaining = next_remaining
    return ordered


def build_pipeline(
    ordered: Sequence[NodeSpec],
    postprocs: Sequence[PostprocSpec] = (),
    *,
    resources: Mapping[str, Any] | None = None,
) -> MCPipeline:
    """Compile validated, ordered nodes (+ postprocs) into a runnable pipeline.

    ``ordered`` are the per-replication nodes in execution order; ``postprocs``
    are the post-loop ops (a separate terminal phase). No model is needed: every
    step compiles purely from its parameters (simulation shocks come from the
    explicit registry, not a model), so a pipeline builds before any model is on
    hand. ``resources`` reattaches bulk side-channel data the JSON spec only
    references by key: a ``raw_model_data`` node's arrays (keyed by its ``data_ref``)
    and a ``custom`` op's callable (keyed by its ``func_ref``). The bundle loader
    supplies it; for an all-builtin pipeline it can be omitted.
    """
    resources = resources or {}
    per_rep_steps = []
    for node in ordered:
        if node.step_type == "raw_model_data":
            per_rep_steps.append(_build_raw_model_data(node, resources))
        elif node.step_type == "transform:custom":
            per_rep_steps.append(_build_custom(node, resources, transform_step))
        else:
            definition = STEP_CATALOG.get(node.step_type)
            if definition is None:
                raise ValueError(f"Unsupported MC step type: {node.step_type}")
            per_rep_steps.append(
                definition.build(node.name, _clean_params(node.params))
            )

    postproc_steps = []
    for pp in postprocs:
        if pp.step_type == "postproc:custom":
            postproc_steps.append(_build_custom(pp, resources, postproc_step))
        else:
            definition = STEP_CATALOG.get(pp.step_type)
            if definition is None:
                raise ValueError(f"Unsupported MC postproc step type: {pp.step_type}")
            postproc_steps.append(definition.build(pp.name, _clean_params(pp.params)))
    return MCPipeline(per_rep_steps, postproc_steps)


def _build_raw_model_data(node: NodeSpec, resources: Mapping[str, Any]) -> Any:
    """Rehydrate a ``raw_model_data`` datagen, injecting its arrays from resources."""
    params = dict(node.params)
    ref = params.pop("data_ref", node.name)
    params.pop("data_shapes", None)
    arrays = resources.get(ref)
    if arrays is None:
        raise ValueError(
            f"raw_model_data step '{node.name}' references data '{ref}' that is not "
            "present in the supplied resources."
        )
    kwargs: dict[str, Any] = {}
    if "states" in arrays:
        kwargs["states"] = arrays["states"]
    if "observables" in arrays:
        kwargs["observables"] = arrays["observables"]
    raw = {
        key[len("raw:") :]: value
        for key, value in arrays.items()
        if key.startswith("raw:")
    }
    if raw:
        kwargs["raw"] = raw
    observable_names = params["observable_names"]
    if observable_names:
        kwargs["observable_names"] = tuple(observable_names)
    return raw_model_data_step(node.name, **kwargs)


def _build_custom(
    node: NodeSpec | PostprocSpec,
    resources: Mapping[str, Any],
    factory: Any,
) -> Any:
    """Rehydrate a custom op, reattaching its callable from resources.

    ``factory`` is the step constructor for the op role (``transform_step`` for a
    ``transform:custom`` node, ``postproc_step`` for a ``postproc:custom`` spec).
    """
    params = dict(node.params)
    ref = params.pop("func_ref", node.name)
    # The authoring source rides in ``code`` (compiled into the resources
    # callable upstream); it is not a runtime kwarg of the op.
    params.pop("code", None)
    func = resources.get(ref)
    if func is None:
        raise ValueError(
            f"custom step '{node.name}' references callable '{ref}' that is not "
            "present in the supplied resources."
        )
    return factory(node.name, func, **_clean_params(params))


def run_pipeline(
    spec: PipelineSpec,
    *,
    reference: SolvedModel | None,
    dgp: SolvedModel | None,
    n_rep: int,
    fail_fast: bool,
    resources: Mapping[str, Any] | None = None,
) -> MCPipelineResult:
    """Validate, compile, and run ``spec`` against the reference and DGP models.

    ``resources`` reattaches bulk side-channels the spec references by key
    (``raw_model_data`` arrays, ``custom`` callables); see :func:`build_pipeline`.
    """
    ordered, postprocs = validate_pipeline_spec(
        spec,
        has_reference=reference is not None,
        has_dgp=dgp is not None,
    )
    assert reference is not None
    pipeline = build_pipeline(ordered, postprocs, resources=resources)
    return pipeline.run(
        reference=reference,
        dgp=dgp,
        n_rep=n_rep,
        retain_payloads=False,
        retain_test_results=False,
        retain_contexts=True,
        fail_fast=fail_fast,
        verbosity=0,
    )


def _datagen_has_observables(datagen: NodeSpec) -> bool:
    """Whether the root datagen produces observables a filter can consume."""
    if datagen.step_type == "raw_model_data":
        return "observables" in dict(datagen.params["data_shapes"])
    return "observables" not in datagen.params or bool(datagen.params["observables"])


def _bind_graph_dependency(
    node: NodeSpec,
    parents: list[NodeSpec],
    datagen: NodeSpec,
) -> NodeSpec:
    """Validate graph-level requirements that do not mutate source params."""
    params = dict(node.params)
    if node.step_type == "filter":
        if not _datagen_has_observables(datagen):
            raise ValueError("Filter steps require the datagen to produce observables.")
    return replace(node, params=params)


def _validate_dependency(node: NodeSpec, prior_names: set[str]) -> None:
    params = node.params
    if node.step_type == "filter":
        return
    for binding in _source_bindings(node):
        source_key = binding.source_key
        field_key = binding.field_key
        if source_key not in params and field_key not in params:
            continue
        if source_key not in params:
            raise ValueError(
                f"Step '{node.name}' declares '{field_key}' without '{source_key}'."
            )
        if field_key not in params:
            raise ValueError(
                f"Step '{node.name}' declares '{source_key}' without '{field_key}'."
            )
        producer = str(params[source_key])
        if producer not in prior_names:
            raise ValueError(f"Step '{node.name}' requires prior source '{producer}'.")


def _source_bindings(node: NodeSpec) -> tuple[SourceBinding, ...]:
    definition = STEP_CATALOG.get(node.step_type)
    if definition is None:
        return ()
    return definition.source_bindings


def _clean_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if value == "" or value == [] or value is None:
            continue
        out[key] = value
    return out
