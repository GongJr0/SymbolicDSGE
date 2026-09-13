from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, Field

from SymbolicDSGE.monte_carlo.spec import (
    MCStepKind,
    PipelineMeta,
    SourceSpec,
    StepMeta,
)


class MCSourceSpec(BaseModel):
    """One authored source leg of a step, as the form resolved it.

    Mirrors :class:`SourceSpec`, the serialized form of a compiled selector.
    """

    arg: str = Field(min_length=1)
    source_step: str = Field(min_length=1)
    field: str = Field(min_length=1)
    #: Explicit indices as a list, a slice as its ``start``/``stop``/``step``,
    #: or null for all columns.
    columns: list[int] | dict[str, int | None] | None = None
    burn_in: int = Field(default=0, ge=0)

    def to_core(self) -> SourceSpec:
        """The selector as the pydantic-free mapping a step records."""
        return cast(SourceSpec, self.model_dump())


class MCStepSpec(BaseModel):
    """One authored step, as the form resolved it.

    Mirrors :class:`StepMeta`, so :meth:`to_core` drops the pydantic wrapper and
    nothing else. ``code`` is the exception: it is the authoring source of a
    custom op, which compiles into the step's callable rather than travelling as
    one of its kwargs.
    """

    name: str = Field(min_length=1)
    op_type: str = Field(min_length=1)
    step_type: MCStepKind
    kwargs: dict[str, Any] = Field(default_factory=dict)
    source_args: list[MCSourceSpec] = Field(default_factory=list)
    n_retain: int = Field(default=-1, ge=-1)
    #: Custom-op source. Compiled into the step's callable, never a kwarg.
    code: str | None = None

    def to_core(self) -> StepMeta:
        """The step as the pydantic-free mapping a bundle writes."""
        return cast(StepMeta, self.model_dump(exclude={"code"}))


class MCPipelineSpec(BaseModel):
    """A pipeline as the form resolved it: two lists of steps and nothing else."""

    replication_steps: list[MCStepSpec] = Field(min_length=1)
    postproc_steps: list[MCStepSpec] = Field(default_factory=list)

    @property
    def steps(self) -> tuple[MCStepSpec, ...]:
        """Every step, in the order the phases run."""
        return (*self.replication_steps, *self.postproc_steps)

    def to_core(self) -> PipelineMeta:
        """The pipeline as the pydantic-free document a bundle stores."""
        return PipelineMeta(
            replication_steps=[step.to_core() for step in self.replication_steps],
            postproc_steps=[step.to_core() for step in self.postproc_steps],
        )


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
