"""Dependency graph derived from an :class:`MCPipeline`'s steps.

A pipeline is *authored* as a flat, ordered list of :class:`MCStep`, but it
*is* a DAG: each step pulls inputs from upstream steps. Those dependencies are
encoded in the step kwargs — the input *channel* (``source`` /
``residual_source`` / ``y_source`` / ``X_source`` / ``x_source``) and the
*producer* (``filter_key`` / a ``*_payload_key`` / the implicit datagen). This
module resolves that into an explicit graph **owned by the pipeline** so graph
structure is computed in exactly one place: serializers (``to_spec``) and graph
consumers read :class:`PipelineGraph` instead of re-deriving from kwargs.

Channels resolve to producers as:

- ``states`` / ``observables`` -> the datagen (root);
- a filter channel (``std_innov`` / ``x_pred`` / ...) -> the step named by
  ``filter_key`` (default ``"filter"``);
- ``payload`` -> the step named by the leg's ``*_payload_key``.

Op defaults matter: a step authored as ``wald_test_step("w")`` carries no
``source`` kwarg yet reads ``std_innov`` by default, so the resolver merges the
catalogue field defaults for each leg before reading it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .catalog import FILTER_SOURCES, STEP_CATALOG
from .mc_constructs import MCStep, OpType

#: Input-channel kwarg -> the kwarg naming that leg's payload producer.
_LEG_PAYLOAD_KEYS: dict[str, str] = {
    "source": "payload_key",
    "residual_source": "residual_payload_key",
    "y_source": "y_payload_key",
    "X_source": "x_payload_key",
    "x_source": "x_payload_key",
}

_DATA_CHANNELS = frozenset({"states", "observables"})
_DEFAULT_FILTER_KEY = "filter"


@dataclass(frozen=True)
class InputEdge:
    """One resolved input dependency of a step.

    ``role`` is the consumer's input leg (``"source"``, ``"residual_source"``,
    ...); ``producer`` is the upstream step name; ``channel`` is what is read
    from it (``"observables"``, ``"std_innov"``, ``"payload"``, ...).
    """

    role: str
    producer: str
    channel: str


@dataclass(frozen=True)
class PipelineNode:
    """A step plus its resolved place in the dependency graph."""

    step: MCStep
    inputs: tuple[InputEdge, ...]
    children: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.step.name

    @property
    def step_type(self) -> str | None:
        return self.step.step_type

    @property
    def is_root(self) -> bool:
        """The datagen — the one node with no inputs."""
        return not self.inputs

    @property
    def is_leaf(self) -> bool:
        """A terminal — nothing downstream consumes it."""
        return not self.children

    @property
    def parents(self) -> tuple[str, ...]:
        """Producer names across all input legs, de-duplicated in leg order."""
        seen: dict[str, None] = {}
        for edge in self.inputs:
            seen.setdefault(edge.producer, None)
        return tuple(seen)

    @property
    def primary_parent(self) -> str | None:
        """The single structural parent for a spec edge.

        Prefers a filter producer (so the forward graph binder re-derives
        ``filter_key`` from the edge); otherwise the first input's producer.
        ``None`` for the root.
        """
        for edge in self.inputs:
            if edge.channel in FILTER_SOURCES:
                return edge.producer
        return self.inputs[0].producer if self.inputs else None


class PipelineGraph:
    """The dependency DAG of a pipeline, keyed by step name."""

    def __init__(self, nodes: dict[str, PipelineNode], root: str) -> None:
        self.nodes = nodes
        self.root = root
        #: authored (and validated) execution order — a valid topological order.
        self.order: tuple[str, ...] = tuple(nodes)

    @classmethod
    def from_steps(cls, steps: tuple[MCStep, ...]) -> PipelineGraph:
        if not steps:
            raise ValueError("Cannot build a graph from an empty pipeline.")
        index = {step.name: i for i, step in enumerate(steps)}
        root = steps[0].name

        inputs_by_name: dict[str, tuple[InputEdge, ...]] = {}
        for position, step in enumerate(steps):
            edges = _resolve_inputs(step, root_name=root)
            for edge in edges:
                if edge.producer not in index:
                    raise ValueError(
                        f"Step {step.name!r} depends on unknown producer "
                        f"{edge.producer!r}."
                    )
                if index[edge.producer] >= position:
                    raise ValueError(
                        f"Step {step.name!r} depends on {edge.producer!r}, which "
                        "does not appear earlier in the pipeline."
                    )
            inputs_by_name[step.name] = edges

        children: dict[str, list[str]] = {step.name: [] for step in steps}
        for name, edges in inputs_by_name.items():
            for producer in dict.fromkeys(edge.producer for edge in edges):
                children[producer].append(name)

        nodes = {
            step.name: PipelineNode(
                step=step,
                inputs=inputs_by_name[step.name],
                children=tuple(children[step.name]),
            )
            for step in steps
        }
        return cls(nodes, root)

    def __iter__(self) -> Any:
        return iter(self.nodes.values())

    @property
    def leaves(self) -> tuple[PipelineNode, ...]:
        return tuple(node for node in self.nodes.values() if node.is_leaf)

    def edges(self) -> list[tuple[str, str]]:
        """Structural ``(producer, consumer)`` edges (one per non-root node)."""
        out: list[tuple[str, str]] = []
        for node in self.nodes.values():
            parent = node.primary_parent
            if parent is not None:
                out.append((parent, node.name))
        return out


def _resolve_inputs(step: MCStep, *, root_name: str) -> tuple[InputEdge, ...]:
    if step.op_type is OpType.DATAGEN:
        return ()
    if step.op_type is OpType.POSTPROC:
        # Post-loop ops reference producers by trace key, not by an input
        # channel/edge, so they contribute no structural graph edges.
        return ()
    if step.op_type is OpType.FILTER:
        # Filters consume the datagen's observables implicitly (no source kwarg).
        return (InputEdge(role="source", producer=root_name, channel="observables"),)

    edges: list[InputEdge] = []
    for leg, payload_key in _LEG_PAYLOAD_KEYS.items():
        channel = _effective_channel(step, leg)
        if not channel:
            continue
        producer = _resolve_producer(
            step, channel=channel, payload_key=payload_key, root_name=root_name
        )
        edges.append(InputEdge(role=leg, producer=producer, channel=channel))
    return tuple(edges)


def _effective_channel(step: MCStep, leg: str) -> str | None:
    """The channel a leg reads, falling back to the catalogue field default."""
    if leg in step.kwargs:
        value = step.kwargs[leg]
        return str(value) if value else None
    default = _catalog_default(step.step_type, leg)
    return str(default) if default else None


def _catalog_default(step_type: str | None, leg: str) -> Any:
    if step_type is None:
        return None
    definition = STEP_CATALOG.get(step_type)
    if definition is None:
        return None
    for field in definition.fields:
        if field.key == leg:
            return field.default
    return None


def _resolve_producer(
    step: MCStep, *, channel: str, payload_key: str, root_name: str
) -> str:
    if channel in _DATA_CHANNELS:
        # states/observables are produced by the datagen (the graph root).
        return root_name
    if channel == "payload":
        producer = step.kwargs.get(payload_key)
        if not producer:
            raise ValueError(
                f"Step {step.name!r} reads source='payload' but has no "
                f"{payload_key!r} naming the producer."
            )
        return str(producer)
    if channel in FILTER_SOURCES:
        return str(step.kwargs.get("filter_key", _DEFAULT_FILTER_KEY))
    raise ValueError(
        f"Step {step.name!r} has unknown input channel {channel!r} on a "
        "dependency leg."
    )
