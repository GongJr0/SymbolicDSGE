from __future__ import annotations

from .ols_result import OLSResult
from ..utils import process_args
from ..enums import RegressionStatus
from ..solvers import chol_solve, lstsq_solve

import numpy as np
from numpy import float64
from numpy.typing import NDArray

NDF = NDArray[float64]


def ols(
    x: NDF,
    y: NDF,
    variables: list[str] | None = None,
    intercept: bool = True,
) -> OLSResult:
    X, y, variables = process_args(x, y, variables)

    var_names = list(variables)
    if intercept:
        X = np.hstack((np.ones((X.shape[0], 1), dtype=np.float64), X))
        var_names = ["Intercept", *var_names]

    try:
        coef, L, status = chol_solve(X, y)
    except Exception:
        coef, L, status = lstsq_solve(X, y)

    return OLSResult(
        variables=var_names,
        coefficients=coef,
        y=y,
        X=X,
        status=RegressionStatus(status),
        _L=L,
    )
