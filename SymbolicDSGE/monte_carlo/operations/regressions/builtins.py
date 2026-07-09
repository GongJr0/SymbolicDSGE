from __future__ import annotations

from typing import Any
from ...mc_constructs import ColumnSelector, MCStep, OpType, _compile_source_args
from .ops import run_regression


def regression_step(
    name: str,
    *,
    y_source: str,
    y_field: str,
    X_source: str,
    X_field: str,
    y_column: ColumnSelector = None,
    X_columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """Fit a per-replication regression of ``y`` on ``X``.

    Signature: ``regression_step(name, *, y_source, y_field, X_source, X_field, kind="ols",
    y_column=None, X_columns=None, intercept=True, variables=None,
    **kind_kwargs)``.

    ``kind`` selects the estimator ("ols", "ridge", "lasso", "elastic_net", and
    their "_gs" grid-searched variants); penalty hyperparameters pass through
    ``**kind_kwargs`` (e.g. ``alpha=...``).

    Example:
        >>> regression_step("r", y_source="datagen", y_field="observables", X_source="datagen", X_field="states")
        >>> regression_step("rg", y_source="datagen", y_field="observables", X_source="datagen", X_field="states",
        ...                  kind="ridge", alpha=0.1)

    See ``operations.regressions`` for the shared input / selection / output contract.
    """
    source_args = (
        _compile_source_args(
            arg="y",
            source=y_source,
            field=y_field,
            columns=y_column,
            burn_in=burn_in,
            drop_initial=drop_initial,
        ),
        _compile_source_args(
            arg="X",
            source=X_source,
            field=X_field,
            columns=X_columns,
            burn_in=burn_in,
            drop_initial=drop_initial,
        ),
    )
    return MCStep(
        name=name,
        op_type=OpType.REGRESSION,
        func=run_regression,
        kwargs=step_kwargs,
        source_args=source_args,
        step_type="regression",
    )
