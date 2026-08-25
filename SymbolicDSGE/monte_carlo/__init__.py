"""Monte-Carlo pipelines: public API.

Step factories live in :mod:`SymbolicDSGE.monte_carlo.step_factories`. Result serialization helpers
live in :mod:`SymbolicDSGE.monte_carlo.serialize`. This namespace exposes the
types and entry points needed to build, run, and inspect pipelines in code.
"""

from .builder import build_pipeline, run_pipeline
from .catalog import (
    STEP_CATALOG,
    TERMINAL_STEP_TYPES,
    TRANSFORM_STEP_TYPES,
    FieldSpec,
    StepDefinition,
    catalog_payload,
)
from .core import MCPipeline
from .custom_op import (
    NumbaCustomFunc,
    PandasCustomFunc,
    custom_transform,
    pandas_operation,
)
from .postproc import Raw, Summary
from .mc_constructs import (
    MCPipelineResult,
    MCStep,
    OpType,
)
from .shock_native import replication_shocks
from .spec import EdgeSpec, MCStepKind, NodeSpec, PipelineSpec, PostprocSpec
from .traces import available_traces

__all__ = [
    # pipeline + execution
    "MCPipeline",
    "MCPipelineResult",
    "build_pipeline",
    "run_pipeline",
    # step constructs (custom-op authoring surface)
    "MCStep",
    "OpType",
    "custom_transform",
    "pandas_operation",
    "NumbaCustomFunc",
    "PandasCustomFunc",
    "Summary",
    "Raw",
    # graph spec (serialization / bundle)
    "PipelineSpec",
    "NodeSpec",
    "EdgeSpec",
    "PostprocSpec",
    "MCStepKind",
    "available_traces",
    # reproducing one replication
    "replication_shocks",
    # catalogue
    "STEP_CATALOG",
    "StepDefinition",
    "FieldSpec",
    "TERMINAL_STEP_TYPES",
    "TRANSFORM_STEP_TYPES",
    "catalog_payload",
]
