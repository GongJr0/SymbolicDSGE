from __future__ import annotations

from .ols_result import OLSResult
from ..utils import process_args
from ..enums import RegressionStatus
from ..solvers import chol_solve, lstsq_solve, use_scalar_path
from ..._ckernels.regression import ols_chol_solve

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
    """Fit an ordinary least squares regression.

    Attempts a Cholesky solve and falls back to least squares when the design is
    rank deficient; the result's ``status`` records which path was taken.

    Parameters
    ----------
    x : NDF
        Design matrix, shape ``(n, k)``.
    y : NDF
        Response vector, shape ``(n,)``.
    variables : list[str] | None
        Optional names for the design columns. Defaults to ``x0``, ``x1``, ...
    intercept : bool
        If True, include an unpenalized intercept column in the design.

    Returns
    -------
    OLSResult
        Fit with the common diagnostics plus OLS inference outputs.
    """
    X, y, variables = process_args(x, y, variables)

    var_names = list(variables)
    if intercept:
        X = np.hstack((np.ones((X.shape[0], 1), dtype=np.float64), X))
        var_names = ["Intercept", *var_names]

    n, p = X.shape
    if use_scalar_path(n, p):
        coef, L, status = ols_chol_solve(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
        )
    else:
        coef, L, status = chol_solve(X, y)
    if status != int(RegressionStatus.OK):
        coef, L, status = lstsq_solve(X, y)

    return OLSResult(
        variables=var_names,
        coefficients=coef,
        y=y,
        X=X,
        status=RegressionStatus(status),
        _L=L,
    )
