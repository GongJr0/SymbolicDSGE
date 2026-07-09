"""Dependency graph derived from an :class:`MCPipeline`'s compiled steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .catalog import FILTER_SOURCES
from .mc_constructs import MCStep, OpType


@dataclass(frozen=True)
class InputEdge:
    """One resolved input dependency of a step.

    ``role`` is the consumer's input leg (``"source"``, ``"residuals"``,
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

        Prefers a filter producer; otherwise the first input's producer.
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
    if step.op_type is OpType.FILTER:
        return (InputEdge(role="source", producer=root_name, channel="observables"),)

    return tuple(
        InputEdge(
            role=selector.arg,
            producer=selector.source_step,
            channel=selector.field,
        )
        for selector in step.source_args
    )
