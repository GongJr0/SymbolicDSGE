from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

MCStepKind = Literal[
    "simulation",
    "filter",
    "wald",
    "ljung_box",
    "jarque_bera",
    "breusch_pagan",
    "breusch_godfrey",
    "regression",
]


class MCNodeSpec(BaseModel):
    id: str = Field(min_length=1)
    step_type: MCStepKind
    name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class MCEdgeSpec(BaseModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)


class MCPipelineSpec(BaseModel):
    nodes: list[MCNodeSpec] = Field(min_length=1)
    edges: list[MCEdgeSpec] = Field(default_factory=list)


class MCRunRequest(BaseModel):
    pipeline: MCPipelineSpec
    n_rep: int = Field(default=100, gt=0)
    fail_fast: bool = True
