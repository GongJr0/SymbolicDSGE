"""Monte-Carlo pipelines: public API.

Step factories live in :mod:`SymbolicDSGE.monte_carlo.operations` and are reached
through their group (``operations.tests``, ``operations.transforms``,
``operations.regressions``, ``operations.core``). Result serialization helpers
live in :mod:`SymbolicDSGE.monte_carlo.serialize`. This namespace exposes the
types and entry points needed to build, run, and inspect pipelines in code.
"""

from .builder import build_pipeline, run_pipeline, validate_pipeline_spec
from .catalog import (
    STEP_CATALOG,
    TERMINAL_STEP_TYPES,
    TRANSFORM_STEP_TYPES,
    FieldSpec,
    StepDefinition,
    catalog_payload,
)
from .core import MCPipeline
from .custom_op import NumpyCustomFunc, custom_operation
from .mc_constructs import (
    MCContext,
    MCData,
    MCPipelineResult,
    MCStep,
    OpType,
)
from .spec import EdgeSpec, MCStepKind, NodeSpec, PipelineSpec

__all__ = [
    # pipeline + execution
    "MCPipeline",
    "MCPipelineResult",
    "build_pipeline",
    "run_pipeline",
    "validate_pipeline_spec",
    # step constructs (custom-op authoring surface)
    "MCStep",
    "MCContext",
    "MCData",
    "OpType",
    "custom_operation",
    "NumpyCustomFunc",
    # graph spec (serialization / bundle)
    "PipelineSpec",
    "NodeSpec",
    "EdgeSpec",
    "MCStepKind",
    # catalogue
    "STEP_CATALOG",
    "StepDefinition",
    "FieldSpec",
    "TERMINAL_STEP_TYPES",
    "TRANSFORM_STEP_TYPES",
    "catalog_payload",
]
