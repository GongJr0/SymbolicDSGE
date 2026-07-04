"""Per-replication Monte Carlo regressions.

Shared contract for ``regression_step``:

- name: unique step name; also the trace-key prefix (see output below).
- Inputs: dependent ``y_source`` and regressors ``X_source``, each in
  {"observables", "states", "raw", "filter", "payload"}; select with
  ``y_column``/``X_columns``, trim with ``burn_in``/``drop_initial``.
- Shared kwargs: ``filter_key="filter"``, ``intercept=True``.
- Output location: ``traces["regression.<name>.coef"]`` plus OLS diagnostics
  (``.r2``, ``.se``, ``.t_stat``, ``.pval``, ...).
"""

from .builtins import regression_step

__all__ = [
    "regression_step",
]
