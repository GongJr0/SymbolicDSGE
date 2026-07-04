"""Per-replication Monte Carlo series transforms.

Shared contract for the built-in ``*_step`` factories in this group:

- name: unique step name; also the payload key (see output below).
- Possible inputs (``source``): "observables", "states", "raw", "filter", "payload".
- Select/trim: ``columns`` to pick columns, ``burn_in`` to drop leading rows,
  ``drop_initial`` to drop the initial x0 row.
- Shared kwargs: ``filter_key="filter"``, ``payload_key=None``.
- Output location: stored as the step's payload; downstream steps read it with
  ``source="payload"`` and it is stacked into ``traces["payload.<name>"]``.

``transform_step`` wraps an arbitrary user callable and has its own contract.
"""

from .builtins import (
    diff_step,
    log_diff_step,
    log_step,
    rolling_mean_step,
    rolling_std_step,
    rolling_var_step,
    standardize_step,
    transform_step,
)

__all__ = [
    "transform_step",
    "standardize_step",
    "log_step",
    "log_diff_step",
    "diff_step",
    "rolling_mean_step",
    "rolling_std_step",
    "rolling_var_step",
]
