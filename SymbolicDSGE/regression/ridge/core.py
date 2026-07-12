from __future__ import annotations
from typing import Callable, Literal

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from numba import njit

from ..enums import RegressionStatus
from ..utils import process_args
from ..solvers import chol_solve_L2, use_scalar_path
from ..utils import log_grid, get_criterion
from .result import RidgeResult, RidgeObjective
from ..._ckernels.regression import (
    chol_solve_L2 as _chol_solve_L2_native,
    ridge_grid_search as _ridge_grid_search_native,
)

NDF = NDArray[float64]

OK = int(RegressionStatus.OK)
RANK_DEFICIENT = int(RegressionStatus.RANK_DEFICIENT)

#: Criterion-name -> native RegressionCriterion code (regression.h).
_CRITERION_CODES = {"aic": 1, "bic": 2, "loss": 3}


def _native_dims_ok(n: int, p: int) -> bool:
    # Native dispatch shares the scalar-path predicate: native is used only where
    # numba's xtx_xty is also on its manual branch, so the two stay bit-parity.
    return bool(use_scalar_path(n, p))


@njit(cache=True)
def l2_grid_search(
    X: NDF,
    y: NDF,
    start: float64,
    stop: float64,
    num: int,
    obj: Callable[[float64, int, float64], float64],
    intercept: bool,
) -> tuple[float64, NDF, float64, int]:

    alphas = log_grid(start, stop, num)
    obj_trace = np.empty(num, dtype=float64)
    coefs = np.empty((num, X.shape[1]), dtype=float64)
    err_trace = np.empty(num, dtype=np.int64)

    n, p = X.shape
    loop = use_scalar_path(n, p)

    for i in range(num):
        alpha = alphas[i]
        coef, _, eff_df, err = chol_solve_L2(X, y, alpha, intercept)
        coefs[i] = coef
        err_trace[i] = np.int64(err)
        if err != OK:
            obj_trace[i] = float64(np.inf)
        else:
            if loop:
                y_hat = np.empty(n, dtype=float64)
                rss = float64(0.0)
                for j in range(n):
                    y_hat[j] = np.dot(X[j], coef)
                    resid = y[j] - y_hat[j]
                    rss += resid * resid
            else:
                y_hat = X @ coef
                rss = ((y - y_hat) ** 2).sum()
            obj_trace[i] = obj(rss, n, eff_df)

    opt = np.argmin(obj_trace)
    return alphas[opt], coefs[opt], obj_trace[opt], err_trace[opt]


def ridge(
    x: NDF,
    y: NDF,
    alpha: float | float64,
    variables: list[str] | None = None,
    intercept: bool = True,
) -> RidgeResult:
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    x, y, variables = process_args(x, y, variables)
    if intercept:
        X = np.hstack((np.ones((x.shape[0], 1), dtype=np.float64), x))
        variables = ["Intercept", *variables]
    else:
        X = x

    n, p = X.shape
    if _native_dims_ok(n, p):
        coef, _, effective_dof, status = _chol_solve_L2_native(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            float(alpha),
            bool(intercept),
        )
    else:
        coef, _, effective_dof, status = chol_solve_L2(X, y, float64(alpha), intercept)
    return RidgeResult(
        variables=list(variables),
        coefficients=coef,
        y=y,
        X=X,
        status=RegressionStatus(status),
        alpha=float64(alpha),
        effective_dof=float64(effective_dof),
        intercept=intercept,
    )


def ridge_gs(
    x: NDF,
    y: NDF,
    start: float | float64,
    stop: float | float64,
    num: int,
    criterion: Literal["aic", "bic", "loss"] = "aic",
    variables: list[str] | None = None,
    intercept: bool = True,
) -> RidgeResult:
    if start <= 0 or stop <= 0:
        raise ValueError("start and stop must be positive.")
    if num <= 0:
        raise ValueError("num must be positive.")
    X, y, variables = process_args(x, y, variables)
    if intercept:
        X = np.hstack((np.ones((X.shape[0], 1), dtype=np.float64), X))
        variables = ["Intercept", *variables]

    obj = get_criterion(criterion)  # also validates the criterion name
    n, p = X.shape

    if _native_dims_ok(n, p):
        alphas = np.ascontiguousarray(
            log_grid(float64(start), float64(stop), num), dtype=np.float64
        )
        alpha, coef, obj_val, status = _ridge_grid_search_native(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            alphas,
            _CRITERION_CODES[criterion],
            bool(intercept),
        )
    else:
        alpha, coef, obj_val, status = l2_grid_search(
            X, y, float64(start), float64(stop), num, obj, intercept
        )

    if _native_dims_ok(n, p):
        _, _, effective_dof, _ = _chol_solve_L2_native(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            float(alpha),
            bool(intercept),
        )
    else:
        _, _, effective_dof, _ = chol_solve_L2(X, y, float64(alpha), intercept)
    return RidgeResult(
        variables=list(variables),
        coefficients=coef,
        y=y,
        X=X,
        status=RegressionStatus(status),
        alpha=float64(alpha),
        effective_dof=float64(effective_dof),
        intercept=intercept,
        objective=RidgeObjective(criterion),
        objective_value=float64(obj_val),
    )
