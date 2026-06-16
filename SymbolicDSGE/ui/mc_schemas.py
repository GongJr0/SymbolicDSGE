from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from SymbolicDSGE.monte_carlo.spec import MCStepKind, PipelineSpec


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

    def to_core(self) -> PipelineSpec:
        """Convert to the pydantic-free core spec (bundle/text serialization)."""
        return PipelineSpec.from_dict(self.model_dump())


class MCRunRequest(BaseModel):
    pipeline: MCPipelineSpec
    n_rep: int = Field(default=100, gt=0)
    fail_fast: bool = True


class MCCustomOpRequest(BaseModel):
    """A single custom-op source submission for live editor validation."""

    code: str = Field(min_length=1)
