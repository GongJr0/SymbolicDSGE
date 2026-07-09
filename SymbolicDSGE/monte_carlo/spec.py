"""Serializable Monte Carlo pipeline specification (graph form).

Stdlib dataclasses. The core ``monte_carlo`` module must stay pydantic-free
(pydantic is only present transitively under the ``[ui]`` extra). The UI keeps its
pydantic request models and converts via :meth:`PipelineSpec.from_dict`. This is the
text representation a ``.sdsge`` bundle stores for the MC pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

MCStepKind = Literal[
    # datagen / filter
    "simulation",
    "raw_model_data",
    "filter",
    # terminal: tests
    "wald",
    "ljung_box",
    "jarque_bera",
    "breusch_pagan",
    "breusch_godfrey",
    "cusum",
    "cusumsq",
    "chow",
    # terminal: regression
    "regression",
    # transforms
    "standardize",
    "log",
    "log_diff",
    "diff",
    "rolling_mean",
    "rolling_std",
    "rolling_var",
    # post-processing (post-loop ops over across-rep traces)
    "kde",
    # custom (user-supplied ops, shipped as cloudpickle bundle members); the
    # prefix records the op role since a custom op may be a transform or a postproc.
    "transform:custom",
    "postproc:custom",
]

#: Authoritative set of valid step-type strings. Must agree with the keys of
#: :data:`SymbolicDSGE.monte_carlo.catalog.STEP_CATALOG`. There's a regression
#: test in ``tests/monte_carlo/test_catalog_builder.py`` that enforces parity.
STEP_KINDS: frozenset[str] = frozenset(get_args(MCStepKind))

#: Post-loop step kinds. A postproc is a *terminal reduction* over the assembled
#: across-rep traces, not a graph node. It lives in ``PipelineSpec.postprocs``,
#: never in ``nodes``. Keep in sync with ``catalog.POSTPROC_STEP_TYPES`` + the
#: custom postproc kind (guarded by the catalog parity test).
PostprocStepKind = Literal["kde", "postproc:custom"]
POSTPROC_KINDS: frozenset[str] = frozenset(get_args(PostprocStepKind))

#: Per-replication step kinds (everything that is an actual graph node).
PER_REP_KINDS: frozenset[str] = STEP_KINDS - POSTPROC_KINDS


@dataclass
class NodeSpec:
    id: str
    step_type: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "step_type": self.step_type,
            "name": self.name,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NodeSpec:
        step_type = str(data["step_type"])
        if step_type not in STEP_KINDS:
            raise ValueError(f"Unknown MC step type: {step_type!r}")
        return cls(
            id=str(data["id"]),
            step_type=step_type,
            name=str(data["name"]),
            params=dict(data.get("params", {})),
        )


@dataclass
class EdgeSpec:
    source: str
    target: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EdgeSpec:
        return cls(source=str(data["source"]), target=str(data["target"]))


@dataclass
class PostprocSpec:
    """A post-loop op: a named, typed, parameterized terminal reduction over the
    assembled across-rep traces. Deliberately *not* a graph node. It has no
    ``id`` and no edges; its inputs are trace keys carried in ``params``.
    """

    name: str
    step_type: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "step_type": self.step_type,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PostprocSpec:
        step_type = str(data["step_type"])
        if step_type not in POSTPROC_KINDS:
            raise ValueError(f"{step_type!r} is not a post-processing step type.")
        return cls(
            name=str(data["name"]),
            step_type=step_type,
            params=dict(data.get("params", {})),
        )


@dataclass
class PipelineSpec:
    nodes: list[NodeSpec]
    edges: list[EdgeSpec] = field(default_factory=list)
    #: Post-loop ops, run once over the assembled traces. Kept separate from the
    #: per-rep DAG (``nodes``/``edges``). They are not graph participants.
    postprocs: list[PostprocSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "postprocs": [postproc.to_dict() for postproc in self.postprocs],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineSpec:
        nodes = [NodeSpec.from_dict(node) for node in data.get("nodes", [])]
        misplaced = [node.name for node in nodes if node.step_type in POSTPROC_KINDS]
        if misplaced:
            raise ValueError(
                "Post-processing steps must be listed under 'postprocs', not "
                f"'nodes': {misplaced}."
            )
        return cls(
            nodes=nodes,
            edges=[EdgeSpec.from_dict(edge) for edge in data.get("edges", [])],
            postprocs=[PostprocSpec.from_dict(pp) for pp in data.get("postprocs", [])],
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> PipelineSpec:
        return cls.from_dict(json.loads(text))
