"""Native Monte Carlo kernels (per-replication sample transforms).

Re-exports the compiled ``_transforms`` extension, which backs the transform ops
in ``SymbolicDSGE.monte_carlo.operations.transforms``. The parity tests check
these kernels against ``tests/_oracles/mc_transforms``, never against those ops,
which now call straight through to here.
"""

from ._transforms import (
    BAD_ARG as BAD_ARG,
    diff_transform,
    log_transform,
    log_diff_transform,
    rolling_mean,
    rolling_std,
    rolling_var,
    standardize_ax0,
)

__all__ = [
    "BAD_ARG",
    "diff_transform",
    "log_transform",
    "log_diff_transform",
    "rolling_mean",
    "rolling_std",
    "rolling_var",
    "standardize_ax0",
]
