from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, Field, field_validator

from SymbolicDSGE.monte_carlo.spec import MCStepKind, PipelineSpec, PostprocStepKind


def _drop_blank(params: dict[str, Any]) -> dict[str, Any]:
    """Drop the keys the editor submits empty so the step factory's default applies.

    An unwired source posts ``""`` and an untouched column list posts ``[]``;
    neither is a value a factory takes. Written specs are not cleaned: a blank
    there is an error, not an unfinished form.
    """
    return {
        key: value
        for key, value in params.items()
        if value != "" and value != [] and value is not None
    }


class MCNodeSpec(BaseModel):
    id: str = Field(min_length=1)
    step_type: MCStepKind
    name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)

    _clean_params = field_validator("params")(_drop_blank)


class MCEdgeSpec(BaseModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)


class MCPostprocSpec(BaseModel):
    """A post-loop op. Not a graph node -- no ``id``/edges; it references producers
    by trace key in ``params`` and runs once over the assembled traces."""

    step_type: PostprocStepKind
    name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)

    _clean_params = field_validator("params")(_drop_blank)


class MCPipelineSpec(BaseModel):
    nodes: list[MCNodeSpec] = Field(min_length=1)
    edges: list[MCEdgeSpec] = Field(default_factory=list)
    postprocs: list[MCPostprocSpec] = Field(default_factory=list)

    def to_core(self) -> PipelineSpec:
        """Convert to the pydantic-free core spec (bundle/text serialization)."""
        return cast(PipelineSpec, self.model_dump())


class MCRunRequest(BaseModel):
    pipeline: MCPipelineSpec
    n_rep: int = Field(default=100, gt=0)
    n_jobs: int | None = Field(default=None, gt=0)
    fail_fast: bool = True
    verbosity: int = Field(default=0, ge=0, le=2)


class MCCustomOpRequest(BaseModel):
    """A single custom-op source submission for live editor validation.

    ``step_type`` selects the validation namespace: ``postproc:custom`` validates
    under the pandas namespace, everything else under Numba.
    """

    code: str = Field(min_length=1)
    step_type: MCStepKind = "transform:custom"
