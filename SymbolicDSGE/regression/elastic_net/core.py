from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy import float64
from numpy.linalg import solve
from numpy.typing import NDArray
from numba import njit

from ..enums import RegressionStatus
from ..lasso.core import (
    _add_intercept,
    _center_for_intercept,
    _restore_intercept,
    _restore_intercept_path,
    lasso as _lasso,
    lasso_gs as _lasso_gs,
    smooth_threshold,
)
from ..ridge.core import ridge as _ridge, ridge_gs as _ridge_gs
from ..solvers import xtx_xty
from ..utils import get_criterion, log_grid, process_args
from .result import ElasticNetResult

NDF = NDArray[float64]

OK = int(RegressionStatus.OK)
NON_CONVERGENT = int(RegressionStatus.NON_CONVERGENT)


@njit(cache=True)
def split_penalty(alpha: float64, l1_ratio: float64) -> tuple[float64, float64]:
    return alpha * l1_ratio, alpha * (float64(1.0) - l1_ratio)


@njit(cache=True)
def elastic_net_gram_cd(
    G: NDF,
    g: NDF,
    alpha_l1: float64,
    alpha_l2: float64,
    beta0: NDF,
    max_iter: int = 1000,
    tol: float64 = float64(1e-10),
) -> tuple[NDF, int]:
    k = G.shape[0]
    coef = beta0.copy()
    Gcoef = np.zeros(k, dtype=float64)

    for i in range(k):
        acc = float64(0.0)
        for j in range(k):
            acc += G[i, j] * coef[j]
        Gcoef[i] = acc + alpha_l2 * coef[i]

    for _ in range(max_iter):
        max_delta = float64(0.0)
        for j in range(k):
            diag = G[j, j] + alpha_l2
            if diag <= 0.0:
                continue

            z = g[j] - Gcoef[j] + diag * coef[j]
            new_coef_j = smooth_threshold(z, alpha_l1) / diag
            delta = new_coef_j - coef[j]
            abs_delta = np.abs(delta)

            if abs_delta > tol:
                for i in range(k):
                    Gcoef[i] += G[i, j] * delta
                Gcoef[j] += alpha_l2 * delta
                coef[j] = new_coef_j

                if abs_delta > max_delta:
                    max_delta = abs_delta

        if max_delta < tol:
            return coef, OK
    return coef, NON_CONVERGENT


@njit(cache=True)
def elastic_net_gram_cd_path(
    G: NDF,
    g: NDF,
    alpha_grid: NDF,
    l1_ratio: float64,
    max_iter: int = 1000,
    tol: float64 = float64(1e-10),
) -> tuple[NDF, NDArray[np.int64]]:
    n_alpha = alpha_grid.shape[0]
    k = G.shape[0]
    coefs = np.empty((n_alpha, k), dtype=float64)
    statuses = np.empty(n_alpha, dtype=np.int64)
    beta = np.zeros(k, dtype=float64)

    if n_alpha > 1 and alpha_grid[0] < alpha_grid[n_alpha - 1]:
        for pos in range(n_alpha):
            idx = n_alpha - pos - 1
            alpha_l1, alpha_l2 = split_penalty(
                alpha_grid[idx],  # pyright: ignore
                l1_ratio,
            )
            beta, status = elastic_net_gram_cd(
                G,
                g,
                alpha_l1,
                alpha_l2,
                beta,
                max_iter,
                tol,
            )
            coefs[idx] = beta
            statuses[idx] = status
    else:
        for idx in range(n_alpha):
            alpha_l1, alpha_l2 = split_penalty(alpha_grid[idx], l1_ratio)
            beta, status = elastic_net_gram_cd(
                G,
                g,
                alpha_l1,
                alpha_l2,
                beta,
                max_iter,
                tol,
            )
            coefs[idx] = beta
            statuses[idx] = status

    return coefs, statuses


@njit(cache=True)
def elastic_net_active_dof(
    G: NDF,
    beta: NDF,
    alpha_l2: float64,
    intercept: bool,
    atol: float64 = float64(1e-10),
) -> float64:
    k = beta.shape[0]
    n_active = 0
    for j in range(k):
        if np.abs(beta[j]) > atol:
            n_active += 1

    if n_active == 0:
        return float64(int(intercept))

    active = np.empty(n_active, dtype=np.int64)
    cursor = 0
    for j in range(k):
        if np.abs(beta[j]) > atol:
            active[cursor] = j
            cursor += 1

    G_active = np.empty((n_active, n_active), dtype=float64)
    penalized = np.empty((n_active, n_active), dtype=float64)
    for i in range(n_active):
        row = active[i]
        for j in range(n_active):
            col = active[j]
            val = G[row, col]
            G_active[i, j] = val
            penalized[i, j] = val
        penalized[i, i] += alpha_l2

    try:
        smoother = solve(penalized, G_active)
        dof = float64(0.0)
        for i in range(n_active):
            dof += smoother[i, i]
    except Exception:
        dof = float64(n_active)

    return dof + float64(int(intercept))


def elastic_net(
    X: NDF,
    y: NDF,
    alpha: float | float64,
    l1_ratio: float | float64,
    variables: list[str] | None = None,
    intercept: bool = True,
    max_iter: int = 1000,
    tol: float | float64 = float64(1e-10),
) -> ElasticNetResult:
    _validate_elastic_net_params(alpha, l1_ratio, max_iter, tol)

    if l1_ratio == 0.0:
        return _wrap_result(_ridge(X, y, alpha, variables, intercept), l1_ratio)
    if l1_ratio == 1.0:
        return _wrap_result(
            _lasso(X, y, alpha, variables, intercept, max_iter, tol),
            l1_ratio,
        )

    x, y, variables = process_args(X, y, variables)
    if intercept:
        x_fit, y_fit, x_mean, y_mean = _center_for_intercept(x, y)
    else:
        x_fit = x
        y_fit = y

    G, g = _normalized_gram(x_fit, y_fit)
    alpha_l1, alpha_l2 = split_penalty(float64(alpha), float64(l1_ratio))
    beta0 = np.zeros(G.shape[0], dtype=np.float64)
    beta, status = elastic_net_gram_cd(
        G,
        g,
        alpha_l1,
        alpha_l2,
        beta0,
        max_iter,
        float64(tol),
    )
    effective_dof = elastic_net_active_dof(G, beta, alpha_l2, intercept, float64(tol))

    if intercept:
        coef = _restore_intercept(beta, x_mean, y_mean)  # pyright: ignore
        design, variables = _add_intercept(x, variables)
    else:
        coef = beta
        design = x

    return ElasticNetResult(
        variables=list(variables),
        coefficients=coef,
        y=y,
        X=design,
        status=RegressionStatus(status),
        alpha=float64(alpha),
        l1_ratio=float64(l1_ratio),
        effective_dof=effective_dof,
        intercept=intercept,
    )


def elastic_net_gs(
    X: NDF,
    y: NDF,
    start: float | float64,
    stop: float | float64,
    num: int,
    l1_ratio: float | float64,
    criterion: Literal["aic", "bic", "loss"] = "loss",
    variables: list[str] | None = None,
    intercept: bool = True,
    max_iter: int = 1000,
    tol: float | float64 = float64(1e-10),
) -> ElasticNetResult:
    if start <= 0 or stop <= 0:
        raise ValueError("start and stop must be positive.")
    if num <= 0:
        raise ValueError("num must be positive.")
    _validate_elastic_net_params(float64(start), l1_ratio, max_iter, tol)
    objective = get_criterion(criterion)

    if l1_ratio == 0.0:
        return _wrap_result(
            _ridge_gs(X, y, start, stop, num, criterion, variables, intercept),
            l1_ratio,
        )
    if l1_ratio == 1.0 and criterion == "loss":
        return _wrap_result(
            _lasso_gs(X, y, start, stop, num, variables, intercept, max_iter, tol),
            l1_ratio,
        )

    x, y, variables = process_args(X, y, variables)
    if intercept:
        x_fit, y_fit, x_mean, y_mean = _center_for_intercept(x, y)
    else:
        x_fit = x
        y_fit = y

    G, g = _normalized_gram(x_fit, y_fit)
    alpha_grid = log_grid(float64(start), float64(stop), num)
    beta_grid, status_trace = elastic_net_gram_cd_path(
        G,
        g,
        alpha_grid,
        float64(l1_ratio),
        max_iter,
        float64(tol),
    )

    if intercept:
        coef_grid = _restore_intercept_path(
            beta_grid,
            x_mean,  # pyright: ignore
            y_mean,  # pyright: ignore
        )
        design, variables = _add_intercept(x, variables)
    else:
        coef_grid = beta_grid
        design = x

    residuals = y[:, None] - design @ coef_grid.T
    rss_trace = np.asarray((residuals**2).sum(axis=0), dtype=np.float64)
    objective_trace = np.empty(num, dtype=np.float64)
    effective_dof_trace = np.empty(num, dtype=np.float64)

    for i in range(num):
        if status_trace[i] != OK:
            objective_trace[i] = np.inf
            effective_dof_trace[i] = np.nan
            continue
        _, alpha_l2 = split_penalty(alpha_grid[i], float64(l1_ratio))
        effective_dof = elastic_net_active_dof(
            G,
            beta_grid[i],
            alpha_l2,
            intercept,
            float64(tol),
        )
        effective_dof_trace[i] = effective_dof
        objective_trace[i] = objective(rss_trace[i], y.shape[0], effective_dof)

    opt = int(np.argmin(objective_trace))
    coef = np.ascontiguousarray(coef_grid[opt], dtype=np.float64)

    return ElasticNetResult(
        variables=list(variables),
        coefficients=coef,
        y=y,
        X=design,
        status=RegressionStatus(int(status_trace[opt])),
        alpha=float64(alpha_grid[opt]),
        l1_ratio=float64(l1_ratio),
        effective_dof=float64(effective_dof_trace[opt]),
        intercept=intercept,
        alpha_grid=alpha_grid,
        coefficient_path=coef_grid,
        objective_trace=objective_trace,
        rss_trace=rss_trace,
        effective_dof_trace=effective_dof_trace,
        status_trace=status_trace,
    )


def _normalized_gram(X: NDF, y: NDF) -> tuple[NDF, NDF]:
    G, g = xtx_xty(X, y)
    scale = float64(1.0 / y.shape[0])
    return (
        np.asarray(G * scale, dtype=np.float64),
        np.asarray(g * scale, dtype=np.float64),
    )


def _validate_elastic_net_params(
    alpha: float | float64,
    l1_ratio: float | float64,
    max_iter: int,
    tol: float | float64,
) -> None:
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if l1_ratio < 0 or l1_ratio > 1:
        raise ValueError("l1_ratio must be between 0 and 1.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")


def _wrap_result(result: Any, l1_ratio: float | float64) -> ElasticNetResult:
    return ElasticNetResult(
        variables=result.variables,
        coefficients=result.coefficients,
        y=result.y,
        X=result.X,
        status=result.status,
        alpha=result.alpha,
        l1_ratio=float64(l1_ratio),
        effective_dof=result.effective_dof,
        intercept=result.intercept,
        alpha_grid=getattr(result, "alpha_grid", None),
        coefficient_path=getattr(result, "coefficient_path", None),
        objective_trace=getattr(result, "objective_trace", None),
    )
