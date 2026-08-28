"""Monte-Carlo pipelines: public API.

Step factories live in :mod:`SymbolicDSGE.monte_carlo.step_factories`. Result serialization helpers
live in :mod:`SymbolicDSGE.monte_carlo.serialize`. This namespace exposes the
types and entry points needed to build, run, and inspect pipelines in code.
"""

from .core import MCPipeline
from .custom_op import (
    NumbaCustomFunc,
    PandasCustomFunc,
    custom_transform,
    pandas_operation,
)
from .postproc import Raw, Summary
from .mc_constructs import (
    MCStep,
    OpType,
)
from .shock_native import replication_shocks
from .traces import available_traces

__all__ = [
    # pipeline
    "MCPipeline",
    # step constructs (custom-op authoring surface)
    "MCStep",
    "OpType",
    "custom_transform",
    "pandas_operation",
    "NumbaCustomFunc",
    "PandasCustomFunc",
    "Summary",
    "Raw",
    # reproducing one replication
    "replication_shocks",
    # Trace list for postproc
    "available_traces",
]
