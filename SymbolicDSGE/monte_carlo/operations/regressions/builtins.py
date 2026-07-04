from __future__ import annotations

from typing import Any
from ...mc_constructs import MCStep, OpType
from .._docs import with_base_doc
from .ops import run_regression

_BASE_DOC = """
Per-replication Monte Carlo regressions.

- name: unique step name; also the trace-key prefix (see output below).
- Inputs: dependent ``y_source`` and regressors ``X_source``, each in
  {"observables", "states", "raw", "filter", "payload"}; select with
  ``y_column``/``X_columns``, trim with ``burn_in``/``drop_initial``.
- Shared kwargs: ``filter_key="filter"``, ``intercept=True``.
- Output location: ``traces["regression.<name>.coef"]`` plus OLS diagnostics
  (``.r2``, ``.se``, ``.t_stat``, ``.pval``, ...).
"""


@with_base_doc(_BASE_DOC)
def regression_step(name: str, **kwargs: Any) -> MCStep:
    """Fit a per-replication regression of ``y_source`` on ``X_source``.

    Signature: ``regression_step(name, *, y_source, X_source, kind="ols",
    y_column=None, X_columns=None, intercept=True, variables=None,
    **kind_kwargs)``.

    ``kind`` selects the estimator ("ols", "ridge", "lasso", "elastic_net", and
    their "_gs" grid-searched variants); penalty hyperparameters pass through
    ``**kind_kwargs`` (e.g. ``alpha=...``).

    Example:
        >>> regression_step("r", y_source="observables", X_source="states")
        >>> regression_step("rg", y_source="observables", X_source="states",
        ...                  kind="ridge", alpha=0.1)
    """
    return MCStep(
        name=name,
        op_type=OpType.REGRESSION,
        func=run_regression,
        kwargs=kwargs,
        step_type="regression",
    )
