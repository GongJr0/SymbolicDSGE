"""Compile and validate Monte-Carlo pipelines from the core :class:`PipelineSpec`.

Lifted out of ``ui.mc`` so a pipeline can be graph-validated, compiled into an
:class:`MCPipeline`, and run without the ``[ui]`` extra. Operates on the
pydantic-free core dataclasses (:class:`NodeSpec`/:class:`PipelineSpec`); the UI
keeps thin wrappers that convert its request models via ``to_core()``.

Compilation is driven entirely by :data:`SymbolicDSGE.monte_carlo.catalog.STEP_CATALOG`
— there is no per-step branching here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .catalog import (
    FILTER_SOURCES,
    STEP_CATALOG,
    TERMINAL_STEP_TYPES,
    TRANSFORM_STEP_TYPES,
)
from .core import MCPipeline
from .mc_constructs import MCPipelineResult
from .spec import NodeSpec, PipelineSpec

if TYPE_CHECKING:
    from ..core.solved_model import SolvedModel

_DEPENDENCY_SOURCE_KEYS = (
    "source",
    "residual_source",
    "y_source",
    "X_source",
    "x_source",
)
_PAYLOAD_KEYS = (
    "payload_key",
    "residual_payload_key",
    "y_payload_key",
    "x_payload_key",
)


def validate_pipeline_spec(
    spec: PipelineSpec,
    *,
    has_reference: bool,
    has_dgp: bool,
) -> list[NodeSpec]:
    """Validate the pipeline graph and return its steps in execution order.

    Enforces unique ids/names, well-formed edges, exactly one simulation, the
    terminal/filter linking rules, and single-parent dependencies, then binds
    each terminal step to its upstream filter (recording ``filter_key``).
    """
    nodes = {node.id: node for node in spec.nodes}
    if len(nodes) != len(spec.nodes):
        raise ValueError("Pipeline node IDs must be unique.")
    names = [node.name for node in spec.nodes]
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
        if target.step_type == "simulation":
            raise ValueError("The simulation step cannot have an incoming link.")
        if target.step_type == "filter" and source.step_type != "simulation":
            raise ValueError("Filter steps must link directly from simulation.")
        if (
            target.step_type in TRANSFORM_STEP_TYPES
            and source.step_type
            not in {
                "simulation",
                "filter",
            }
            | TRANSFORM_STEP_TYPES
        ):
            raise ValueError(
                f"Transform '{target.name}' must link from simulation, a "
                "filter, or another transform."
            )
        if (
            target.step_type in TERMINAL_STEP_TYPES
            and source.step_type
            not in {
                "simulation",
                "filter",
            }
            | TRANSFORM_STEP_TYPES
        ):
            raise ValueError(
                "Tests and regressions must link from simulation, a filter, "
                "or a transform."
            )
        seen_edges.add(pair)
        outgoing[edge.source].append(edge.target)
        incoming[edge.target].append(edge.source)

    simulations = [node for node in spec.nodes if node.step_type == "simulation"]
    if len(simulations) != 1:
        raise ValueError("Pipeline supports exactly one simulation step.")
    simulation = simulations[0]
    if incoming[simulation.id]:
        raise ValueError("The simulation step cannot have an incoming link.")
    for node in spec.nodes:
        if node.id == simulation.id:
            continue
        if len(incoming[node.id]) != 1:
            raise ValueError(
                f"Step '{node.name}' must have exactly one incoming dependency link."
            )
        if node.step_type in TERMINAL_STEP_TYPES and outgoing[node.id]:
            raise ValueError(f"Terminal step '{node.name}' cannot link forward.")

    if not has_reference:
        raise ValueError("A solved reference model is required.")
    if not has_dgp:
        raise ValueError("A solved DGP model is required by the simulation step.")

    filter_nodes = [node for node in spec.nodes if node.step_type == "filter"]
    transform_nodes = _topological_transforms(
        [node for node in spec.nodes if node.step_type in TRANSFORM_STEP_TYPES],
        incoming,
        placed_ids={simulation.id, *(node.id for node in filter_nodes)},
    )
    terminal_nodes = [
        node for node in spec.nodes if node.step_type in TERMINAL_STEP_TYPES
    ]
    ordered = [simulation, *filter_nodes, *transform_nodes, *terminal_nodes]

    bound: list[NodeSpec] = []
    bound_by_id: dict[str, NodeSpec] = {}
    prior_names: set[str] = set()
    for node in ordered:
        parent_id = incoming[node.id][0] if incoming[node.id] else None
        parent = bound_by_id.get(parent_id) if parent_id is not None else None
        bound_node = _bind_graph_dependency(node, parent, simulation)
        _validate_dependency(bound_node, prior_names)
        bound.append(bound_node)
        bound_by_id[node.id] = bound_node
        prior_names.add(bound_node.name)
    return bound


def _topological_transforms(
    transforms: Sequence[NodeSpec],
    incoming: Mapping[str, list[str]],
    *,
    placed_ids: set[str],
) -> list[NodeSpec]:
    """Order transform nodes so each comes after its (single) parent.

    Transforms can chain (``transform_a`` -> ``transform_b``), so spec order
    isn't always topological. Single-parent constraint is already enforced
    upstream; here we just Kahn-walk the subset.
    """
    remaining = list(transforms)
    ordered: list[NodeSpec] = []
    placed = set(placed_ids)
    while remaining:
        progress = False
        next_remaining: list[NodeSpec] = []
        for node in remaining:
            parent_id = incoming[node.id][0] if incoming[node.id] else None
            if parent_id is None or parent_id in placed:
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
    *,
    dgp: SolvedModel,
) -> MCPipeline:
    """Compile validated, ordered nodes into a runnable :class:`MCPipeline`."""
    steps = []
    for node in ordered:
        definition = STEP_CATALOG.get(node.step_type)
        if definition is None:
            raise ValueError(f"Unsupported MC step type: {node.step_type}")
        steps.append(definition.build(node.name, _clean_params(node.params), dgp))
    return MCPipeline(steps)


def run_pipeline(
    spec: PipelineSpec,
    *,
    reference: SolvedModel | None,
    dgp: SolvedModel | None,
    n_rep: int,
    fail_fast: bool,
) -> MCPipelineResult:
    """Validate, compile, and run ``spec`` against the reference and DGP models."""
    ordered = validate_pipeline_spec(
        spec,
        has_reference=reference is not None,
        has_dgp=dgp is not None,
    )
    assert reference is not None
    assert dgp is not None
    pipeline = build_pipeline(ordered, dgp=dgp)
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


def _bind_graph_dependency(
    node: NodeSpec,
    parent: NodeSpec | None,
    simulation: NodeSpec,
) -> NodeSpec:
    params = dict(node.params)
    if node.step_type == "filter":
        if not bool(simulation.params.get("observables", True)):
            raise ValueError("Filter steps require simulation observables.")
    elif node.step_type in TRANSFORM_STEP_TYPES:
        _bind_consumer_payload(node, parent, params)
    elif node.step_type in TERMINAL_STEP_TYPES:
        _bind_consumer_payload(node, parent, params)
        # Filter-only-derived sources still need an upstream filter binding;
        # if the immediate parent is a filter it provides ``filter_key`` for
        # the FilterResult-typed sources (x_pred / x_filt / ...).
        if parent is not None and parent.step_type == "filter":
            params["filter_key"] = parent.name
        elif any(source in FILTER_SOURCES for source in _sources(params)):
            raise ValueError(
                f"Step '{node.name}' uses filter output and must link from a filter."
            )
    return replace(node, params=params)


def _bind_consumer_payload(
    node: NodeSpec,
    parent: NodeSpec | None,
    params: dict[str, Any],
) -> None:
    """Wire a node that depends on a transform parent.

    Rule:
      * No transform parent: ``source="payload"`` is rejected (legacy guard).
      * Transform parent + explicit ``source="payload"`` on any input leg
        (``source`` / ``residual_source`` / ``y_source`` / ``X_source`` /
        ``x_source``): bind the matching payload key to the parent's name.
      * Transform parent + single-source consumer (only ``source`` in the
        catalog) with no override: default ``source="payload"`` and bind.
      * Transform parent + multi-input consumer with no explicit ``payload``
        leg: raise, asking the user to mark the legs that should chain.
    """
    if parent is None or parent.step_type not in TRANSFORM_STEP_TYPES:
        if node.step_type in TERMINAL_STEP_TYPES and any(
            source == "payload" for source in _sources(params)
        ):
            raise ValueError(
                f"Step '{node.name}': payload sources require a transform "
                "parent in the graph."
            )
        return

    pairs = (
        ("source", "payload_key"),
        ("residual_source", "residual_payload_key"),
        ("y_source", "y_payload_key"),
        ("X_source", "x_payload_key"),
        ("x_source", "x_payload_key"),
    )
    bound = False
    for source_key, payload_key_key in pairs:
        if params.get(source_key) == "payload":
            # Respect an explicit override: the user can point this leg at any
            # earlier transform by name, not just the immediate parent.
            if payload_key_key not in params:
                params[payload_key_key] = parent.name
            bound = True
    if bound:
        return

    # No explicit "payload" leg yet. For a single-source consumer (the
    # catalog declares one of "source" / "residual_source" / ... and only
    # one of them), default that leg to consume the transform's output.
    # Multi-input nodes must opt in explicitly per leg — silently picking
    # which input chains to the transform would be a footgun.
    source_field_keys = {key for key, _ in pairs}
    definition = STEP_CATALOG.get(node.step_type)
    if definition is not None:
        declared_source_fields = [
            field.key for field in definition.fields if field.key in source_field_keys
        ]
    else:
        declared_source_fields = [key for key in source_field_keys if key in params]

    if declared_source_fields == ["source"]:
        params["source"] = "payload"
        params["payload_key"] = parent.name
        return

    raise ValueError(
        f"Step '{node.name}': linked from transform '{parent.name}' but no "
        "input leg is set to source='payload'. For multi-input steps "
        "(Breusch-Pagan / CUSUM / regression / etc.) explicitly mark each "
        "leg you want to read from the transform."
    )


def _validate_dependency(node: NodeSpec, prior_names: set[str]) -> None:
    params = node.params
    if node.step_type == "filter":
        return
    if any(source in FILTER_SOURCES for source in _sources(params)):
        filter_key = str(params.get("filter_key", "filter"))
        if filter_key not in prior_names:
            raise ValueError(
                f"Step '{node.name}' requires prior filter payload '{filter_key}'."
            )
    for payload_key in (params.get(key) for key in _PAYLOAD_KEYS if params.get(key)):
        if str(payload_key) not in prior_names:
            raise ValueError(
                f"Step '{node.name}' requires prior payload '{payload_key}'."
            )


def _sources(params: Mapping[str, Any]) -> list[Any]:
    return [
        params[key] for key in _DEPENDENCY_SOURCE_KEYS if params.get(key) is not None
    ]


def _clean_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if value == "" or value == [] or value is None:
            continue
        out[key] = value
    return out
