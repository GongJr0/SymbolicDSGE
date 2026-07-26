"""Native Monte Carlo kernels (per-replication sample transforms).

Re-exports the compiled ``_transforms`` extension. Nothing in the library
imports this yet: the Python transform ops in
``SymbolicDSGE.monte_carlo.operations.transforms`` remain the implementation,
and these kernels exist for the native replication loop and its parity tests.
"""

from ._transforms import (
    BAD_ARG as BAD_ARG,
    diff,
    log,
    log_diff,
    rolling_mean,
    rolling_std,
    rolling_var,
    standardize_ax0,
)

__all__ = [
    "BAD_ARG",
    "diff",
    "log",
    "log_diff",
    "rolling_mean",
    "rolling_std",
    "rolling_var",
    "standardize_ax0",
]
