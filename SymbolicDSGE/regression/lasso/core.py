from __future__ import annotations

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from numba import njit

from .result import LassoResult
from ..solvers import xtx_xty
from ..enums import RegressionStatus
from ..utils import log_grid, process_args

OK = int(RegressionStatus.OK)
RANK_DEFICIENT = int(RegressionStatus.RANK_DEFICIENT)
NON_CONVERGENT = int(RegressionStatus.NON_CONVERGENT)

NDF = NDArray[float64]


@njit(cache=True, fastmath=True)
def smooth_threshold(z: float64, gamma: float64) -> float64:
    if z > gamma:
        return z - gamma
    elif z < -gamma:
        return z + gamma
    else:
        return float64(0.0)


@njit(cache=True, fastmath=True)
def lasso_gram_cd(
    G: NDArray[float64],
    g: NDArray[float64],
    alpha: float64,
    max_iter: int = 1000,
    tol: float64 = float64(1e-10),
) -> tuple[NDF, int]:
    k = G.shape[0]
    coef = np.zeros(k, dtype=float64)
    Gcoef = np.zeros(k, dtype=float64)

    for _ in range(max_iter):
        max_delta = float64(0.0)
        for j in range(k):
            if G[j, j] <= 0.0:
                continue
            z = g[j] - Gcoef[j] + G[j, j] * coef[j]
            new_coef_j = smooth_threshold(z, alpha) / G[j, j]
            delta = new_coef_j - coef[j]

            if not np.isclose(
                delta, 0.0, atol=tol
            ):  # fastmath=True is not exact in floating point comparisons
                for i in range(k):
                    Gcoef[i] += G[i, j] * delta
                coef[j] = new_coef_j

                if np.abs(delta) > max_delta:
                    max_delta = np.abs(delta)

        if max_delta < tol:
            return coef, OK
    return coef, NON_CONVERGENT


@njit(cache=True, fastmath=True)
def solve_small(A: NDF, b: NDF) -> NDF:
    """In-place Gaussian elimination for tiny square systems (k≤25)."""
    n = A.shape[0]
    A = A.copy()
    b = b.copy()
    for col in range(n):
        # partial pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(A[row, col]) > abs(A[max_row, col]):
                max_row = row
        A[col], A[max_row] = A[max_row].copy(), A[col].copy()
        b[col], b[max_row] = b[max_row], b[col]
        piv = A[col, col]
        for row in range(col + 1, n):
            f = A[row, col] / piv
            b[row] -= f * b[col]
            for c in range(col, n):
                A[row, c] -= f * A[col, c]
    # back-substitute
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i, j] * x[j]
        x[i] /= A[i, i]
    return x


@njit(cache=True, fastmath=True)
def lars_lasso_gram(G: NDF, c: NDF) -> tuple[NDF, NDF, int]:
    """
    Full LARS-Lasso path on the Gram matrix.

    Returns
    -------
    lambdas : (n_knots,)   λ values at each path event (descending)
    betas   : (n_knots, k) β at each knot (rows align with lambdas)

    Interpolate to any λ_target:
        i = searchsorted(lambdas[::-1], λ_target)  # lambdas is descending
        β = betas[i-1] + (λ_target - lambdas[i-1]) / ...
    Use lasso_path_eval() below for this.
    """
    k = G.shape[0]
    max_active = k

    beta = np.zeros(k)
    active = np.zeros(k, np.bool_)
    signs = np.zeros(k)  # signs of active correlations
    n_active = 0

    # storage: at most k+1 knots
    lam_path = np.zeros(k + 1)
    beta_path = np.zeros((k + 1, k))

    # residual correlations  r = c - G @ beta  (maintained incrementally)
    r = c.copy()

    # initial lambda
    lam = 0.0
    for j in range(k):
        if abs(r[j]) > lam:
            lam = abs(r[j])

    lam_path[0] = lam
    beta_path[0] = beta
    knot = 1

    for _ in range(max_active):
        # find variable(s) with |r_j| == lam
        new_var = -1
        for j in range(k):
            if not active[j] and abs(abs(r[j]) - lam) < 1e-12 * lam:
                new_var = j
                break
        if new_var < 0:
            break

        active[new_var] = True
        signs[new_var] = 1.0 if r[new_var] > 0.0 else -1.0
        n_active += 1

        # solve (G_AA) d_A = s_A  (least-squares equiangular direction)
        # collect active indices
        act_idx = np.empty(n_active, np.int64)
        cnt = 0
        for j in range(k):
            if active[j]:
                act_idx[cnt] = j
                cnt += 1

        # build G_AA and s_A  (at most ~25×25, small)
        G_AA = np.empty((n_active, n_active))
        s_A = np.empty(n_active)
        for a in range(n_active):
            s_A[a] = signs[act_idx[a]]
            for b in range(n_active):
                G_AA[a, b] = G[act_idx[a], act_idx[b]]

        # solve G_AA @ w = s_A  (use Gaussian elimination)
        w = solve_small(G_AA, s_A)
        # normalise so equiangular unit: A = 1/sqrt(s'w)
        sw = 0.0
        for a in range(n_active):
            sw += s_A[a] * w[a]
        if sw <= 0.0:
            break
        A_scalar = 1.0 / sw**0.5
        # d_A = A * w  (direction in full β space)
        d = np.zeros(k)
        for a in range(n_active):
            d[act_idx[a]] = A_scalar * w[a]

        # correlations of direction with all predictors
        Gd = np.zeros(k)
        for j in range(k):
            for a in range(n_active):
                Gd[j] += G[j, act_idx[a]] * d[act_idx[a]]

        # step length: earliest of join or lasso-drop events
        step = lam  # at most shrink to 0
        drop_var = -1

        for j in range(k):
            if not active[j]:
                # join event: when |r_j - step*Gd_j| == lam - step*A
                # two cases per inactive variable
                denom1 = A_scalar - Gd[j]
                denom2 = A_scalar + Gd[j]
                if denom1 > 1e-14:
                    t = (lam - r[j]) / denom1
                    if 0.0 < t < step:
                        step = t
                if denom2 > 1e-14:
                    t = (lam + r[j]) / denom2
                    if 0.0 < t < step:
                        step = t

        # lasso drop: active coeff hits zero
        for a in range(n_active):
            j = act_idx[a]
            if d[j] != 0.0:
                t = -beta[j] / d[j]
                if 0.0 < t < step:
                    step = t
                    drop_var = j

        # update
        lam -= step
        for j in range(k):
            beta[j] += step * d[j]
            r[j] -= step * Gd[j]

        # handle drop
        if drop_var >= 0:
            active[drop_var] = False
            beta[drop_var] = 0.0
            n_active -= 1

        lam_path[knot] = lam
        beta_path[knot] = beta.copy()
        knot += 1

        if lam <= 0.0 or n_active >= max_active:
            return lam_path[:knot], beta_path[:knot], OK

    return lam_path[:knot], beta_path[:knot], NON_CONVERGENT


@njit(cache=True, fastmath=True)
def lasso_path_eval(lam_path: NDF, beta_path: NDF, lam_grid: NDF) -> NDF:
    """
    Evaluate LARS-Lasso path at any λ grid.

    Parameters
    ----------
    lam_path : (n_knots,)   λ values at each path event (descending)
    beta_path : (n_knots, k) β at each knot (rows align with lam_path)
    lam_grid : (n_grid,) λ values to evaluate at (descending)

    Returns
    -------
    beta_grid : (n_grid, k) β at each λ in lam_grid
    """
    n_grid = lam_grid.shape[0]
    n_knots = lam_path.shape[0]
    k = beta_path.shape[1]
    out = np.zeros((n_grid, k))

    for g in range(n_grid):
        lam = lam_grid[g]

        if lam >= lam_path[0]:  # above the first knot, all zero
            continue

        if lam <= lam_path[n_knots - 1]:  # below the last knot, use last beta
            out[g] = beta_path[n_knots - 1]
            continue

        # binary search
        lo, hi = 0, n_knots - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if lam_path[mid] >= lam:
                lo = mid
            else:
                hi = mid

        # linear interpolation
        t = (lam - lam_path[lo]) / (lam_path[hi] - lam_path[lo])
        for j in range(k):
            out[g, j] = beta_path[lo, j] + t * (beta_path[hi, j] - beta_path[lo, j])

    return out


def _effective_dof(coef: NDF, intercept: bool) -> float64:
    penalized = coef[1:] if intercept else coef
    return float64(np.count_nonzero(penalized) + int(intercept))


def _add_intercept(x: NDF, variables: list[str]) -> tuple[NDF, list[str]]:
    return (
        np.hstack((np.ones((x.shape[0], 1), dtype=np.float64), x)),
        ["Intercept", *variables],
    )


def _center_for_intercept(x: NDF, y: NDF) -> tuple[NDF, NDF, NDF, float64]:
    x_mean = np.asarray(x.mean(axis=0), dtype=np.float64)
    y_mean = float64(y.mean())
    return (
        np.ascontiguousarray(x - x_mean, dtype=np.float64),
        np.ascontiguousarray(y - y_mean, dtype=np.float64),
        x_mean,
        y_mean,
    )


def _restore_intercept(beta: NDF, x_mean: NDF, y_mean: float64) -> NDF:
    intercept = float64(y_mean - np.dot(x_mean, beta))
    return np.concatenate((np.asarray([intercept], dtype=np.float64), beta))


def _restore_intercept_path(beta_path: NDF, x_mean: NDF, y_mean: float64) -> NDF:
    intercept = np.asarray(y_mean - beta_path @ x_mean, dtype=np.float64)
    return np.column_stack((intercept, beta_path)).astype(np.float64, copy=False)


def lasso(
    X: NDF,
    y: NDF,
    alpha: float | float64,
    variables: list[str] | None = None,
    intercept: bool = True,
    max_iter: int = 1000,
    tol: float | float64 = float64(1e-10),
) -> LassoResult:
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    x, y, variables = process_args(X, y, variables)
    if intercept:
        x_fit, y_fit, x_mean, y_mean = _center_for_intercept(x, y)
    else:
        x_fit = x
        y_fit = y

    G, c = xtx_xty(x_fit, y_fit)
    beta, status = lasso_gram_cd(G, c, float64(alpha), max_iter, float64(tol))

    if intercept:
        coef = _restore_intercept(beta, x_mean, y_mean)
        design, variables = _add_intercept(x, variables)
    else:
        coef = beta
        design = x

    return LassoResult(
        variables=list(variables),
        coefficients=coef,
        y=y,
        X=design,
        status=RegressionStatus(status),
        alpha=float64(alpha),
        effective_dof=_effective_dof(coef, intercept),
        intercept=intercept,
    )


def lasso_gs(
    X: NDF,
    y: NDF,
    start: float | float64,
    stop: float | float64,
    num: int,
    variables: list[str] | None = None,
    intercept: bool = True,
    max_iter: int = 1000,
    tol: float | float64 = float64(1e-10),
) -> LassoResult:
    if start <= 0 or stop <= 0:
        raise ValueError("start and stop must be positive.")
    if num <= 0:
        raise ValueError("num must be positive.")

    x, y, variables = process_args(X, y, variables)
    if intercept:
        x_fit, y_fit, x_mean, y_mean = _center_for_intercept(x, y)
    else:
        x_fit = x
        y_fit = y

    G, c = xtx_xty(x_fit, y_fit)

    lam_grid = log_grid(start, stop, num)
    beta_grid = np.empty((num, x_fit.shape[1]), dtype=np.float64)
    objective_trace = np.empty(num, dtype=np.float64)
    status_trace = np.empty(num, dtype=np.int64)

    for i in range(num):
        beta, status = lasso_gram_cd(
            G,
            c,
            float64(lam_grid[i]),
            max_iter,
            float64(tol),
        )
        beta_grid[i] = beta
        status_trace[i] = status

    if intercept:
        coef_grid = _restore_intercept_path(beta_grid, x_mean, y_mean)
        design, variables = _add_intercept(x, variables)
    else:
        coef_grid = beta_grid
        design = x

    residuals = y[:, None] - design @ coef_grid.T
    objective_trace[:] = np.asarray((residuals**2).sum(axis=0), dtype=np.float64)
    objective_trace[status_trace != OK] = np.inf
    opt = int(np.argmin(objective_trace))
    coef = np.ascontiguousarray(coef_grid[opt], dtype=np.float64)
    status = int(status_trace[opt])

    return LassoResult(
        variables=list(variables),
        coefficients=coef,
        y=y,
        X=design,
        status=RegressionStatus(status),
        alpha=float64(lam_grid[opt]),
        effective_dof=_effective_dof(coef, intercept),
        intercept=intercept,
        alpha_grid=lam_grid,
        coefficient_path=coef_grid,
        objective_trace=objective_trace,
    )
