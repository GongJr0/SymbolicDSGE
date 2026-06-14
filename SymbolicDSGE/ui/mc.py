"""HTTP-facing Monte-Carlo adapters.

The catalogue, graph validation, and pipeline compilation now live in the core
:mod:`SymbolicDSGE.monte_carlo` package (UI-independent). This module is a thin
seam that accepts the pydantic request models and delegates to the core API via
``MCPipelineSpec.to_core()``.
"""

from __future__ import annotations

from typing import Any

from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipelineResult, NodeSpec
from SymbolicDSGE.monte_carlo import build_pipeline as build_pipeline
from SymbolicDSGE.monte_carlo import catalog_payload
from SymbolicDSGE.monte_carlo import run_pipeline as _run_pipeline
from SymbolicDSGE.monte_carlo import validate_pipeline_spec as _validate_pipeline_spec
from SymbolicDSGE.monte_carlo.serialize import (
    serialize_pipeline_result as serialize_pipeline_result,
)

from .mc_schemas import MCPipelineSpec


def mc_catalog() -> dict[str, Any]:
    """The step catalogue payload served at ``/api/mc/catalog``."""
    return catalog_payload()


def validate_pipeline_spec(
    spec: MCPipelineSpec,
    *,
    has_reference: bool,
    has_dgp: bool,
) -> list[NodeSpec]:
    """Graph-validate a UI pipeline request and return its ordered core nodes."""
    return _validate_pipeline_spec(
        spec.to_core(), has_reference=has_reference, has_dgp=has_dgp
    )


def run_pipeline(
    spec: MCPipelineSpec,
    *,
    reference: SolvedModel | None,
    dgp: SolvedModel | None,
    n_rep: int,
    fail_fast: bool,
) -> MCPipelineResult:
    """Validate, compile, and run a UI pipeline request."""
    return _run_pipeline(
        spec.to_core(),
        reference=reference,
        dgp=dgp,
        n_rep=n_rep,
        fail_fast=fail_fast,
    )
