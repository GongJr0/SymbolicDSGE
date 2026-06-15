"""Serializable Monte Carlo pipeline specification (graph form).

Stdlib dataclasses — the core ``monte_carlo`` module must stay pydantic-free
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
]

#: Authoritative set of valid step-type strings. Must agree with the keys of
#: :data:`SymbolicDSGE.monte_carlo.catalog.STEP_CATALOG` — there's a regression
#: test in ``tests/monte_carlo/test_catalog_builder.py`` that enforces parity.
STEP_KINDS: frozenset[str] = frozenset(get_args(MCStepKind))


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
class PipelineSpec:
    nodes: list[NodeSpec]
    edges: list[EdgeSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineSpec:
        return cls(
            nodes=[NodeSpec.from_dict(node) for node in data.get("nodes", [])],
            edges=[EdgeSpec.from_dict(edge) for edge in data.get("edges", [])],
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> PipelineSpec:
        return cls.from_dict(json.loads(text))
