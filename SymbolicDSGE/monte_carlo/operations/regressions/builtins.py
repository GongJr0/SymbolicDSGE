from __future__ import annotations

from typing import Any
from ...mc_constructs import MCStep, OpType
from .ops import run_regression


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

    See ``operations.regressions`` for the shared input / selection / output contract.
    """
    return MCStep(
        name=name,
        op_type=OpType.REGRESSION,
        func=run_regression,
        kwargs=kwargs,
        step_type="regression",
    )
