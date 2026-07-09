from __future__ import annotations

from typing import Any
from ...mc_constructs import MCStep, OpType, _pop_source_arg, _pop_source_controls
from .ops import run_regression


def regression_step(name: str, **kwargs: Any) -> MCStep:
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
    params = dict(kwargs)
    burn_in, drop_initial = _pop_source_controls(params)
    source_args = (
        _pop_source_arg(
            params,
            source_key="y_source",
            field_key="y_field",
            arg="y",
            columns_key="y_column",
            burn_in=burn_in,
            drop_initial=drop_initial,
        ),
        _pop_source_arg(
            params,
            source_key="X_source",
            field_key="X_field",
            arg="X",
            columns_key="X_columns",
            burn_in=burn_in,
            drop_initial=drop_initial,
        ),
    )
    return MCStep(
        name=name,
        op_type=OpType.REGRESSION,
        func=run_regression,
        kwargs=params,
        source_args=source_args,
        step_type="regression",
    )
