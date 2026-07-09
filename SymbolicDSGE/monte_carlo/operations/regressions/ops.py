from __future__ import annotations

from typing import Any, Callable, Sequence, Literal
import numpy as np

from ..types import NDF

from ....core.solved_model import SolvedModel
from ...mc_constructs import MCContext

from ....regression import RegressionResult, RegressionKind
from ....regression.ols import ols
from ....regression.ridge import ridge, ridge_gs
from ....regression.lasso import lasso, lasso_gs
from ....regression.elastic_net import elastic_net, elastic_net_gs


def run_regression(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    kind: Literal[
        "ols",
        "ridge",
        "ridge_gs",
        "lasso",
        "lasso_gs",
        "elastic_net",
        "elastic_net_gs",
    ] = "ols",
    y: NDF,
    X: NDF,
    intercept: bool = True,
    variables: Sequence[str] | None = None,
    **kind_kwargs: Any,
) -> RegressionResult:
    del context, reference, dgp, rep_idx

    if y.shape[1] != 1:
        raise ValueError(
            "Regression response must resolve to exactly one column. "
            f"Got shape {y.shape}."
        )
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            "Regression response and design matrix must have the same number "
            f"of rows. Got y={y.shape[0]} and X={X.shape[0]}."
        )
    if variables is not None and len(variables) != X.shape[1]:
        raise ValueError(
            "Regression variable names must match the number of design columns. "
            f"Got {len(variables)} names for {X.shape[1]} columns."
        )

    y_vec = np.ascontiguousarray(y[:, 0], dtype=np.float64)
    variable_names = list(variables) if variables is not None else None

    fun: Callable[..., RegressionResult]
    match RegressionKind(kind):
        case RegressionKind.OLS:
            fun = ols
        case RegressionKind.RIDGE:
            fun = ridge
        case RegressionKind.RIDGE_GS:
            fun = ridge_gs
        case RegressionKind.LASSO:
            fun = lasso
        case RegressionKind.LASSO_GS:
            fun = lasso_gs
        case RegressionKind.ELASTIC_NET:
            fun = elastic_net
        case RegressionKind.ELASTIC_NET_GS:
            fun = elastic_net_gs
        case _:
            raise ValueError(f"Unsupported regression kind: {kind}")

    return fun(
        X,
        y_vec,
        intercept=intercept,
        variables=variable_names,
        **kind_kwargs,
    )
