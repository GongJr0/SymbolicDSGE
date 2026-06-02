from __future__ import annotations

from numba import njit
import numpy as np
from numpy import float64, asarray
from numpy.linalg import cholesky, lstsq, solve
from numpy.typing import NDArray
from .enums import RegressionStatus

NDF = NDArray[float64]

OK = int(RegressionStatus.OK)
RANK_DEFICIENT = int(RegressionStatus.RANK_DEFICIENT)


@njit(cache=True)
def xtx_xty(X: NDF, y: NDF) -> tuple[NDF, NDF]:
    n, p = X.shape

    if (n >= 1e5) or (p >= 100):
        # BLAS assignment is faster than manual loops for large matrices
        G = X.T @ X
        g = X.T @ y
        return G, g
    else:
        # For smaller matrices, manual loops can be more efficient
        G = np.zeros((p, p), dtype=float64)
        g = np.zeros(p, dtype=float64)

        for i in range(n):
            yi = y[i]
            for j in range(p):
                xij = X[i, j]
                g[j] += xij * yi
                for k in range(j + 1):
                    G[j, k] += xij * X[i, k]

        # Fill the triangle by symmetry
        for j in range(p):
            for k in range(j):
                G[k, j] = G[j, k]
        return G, g


@njit(cache=True)
def chol_solve(X: NDF, y: NDF) -> tuple[NDF, NDF, int]:
    G, g = xtx_xty(X, y)

    L = cholesky(G)
    z = solve(L, g)
    coef: NDF = asarray(solve(L.T, z), dtype=float64)
    return coef, L, OK  # pyright: ignore


@njit(cache=True)
def chol_solve_L2(
    X: NDF, y: NDF, alpha: float64, intercept: bool
) -> tuple[NDF, NDF, float64, int]:
    n = y.shape[0]

    G, g = xtx_xty(X, y)
    G /= n
    g /= n

    G_unpenalized = G.copy()
    p = G.shape[0]

    # Add L2 regularization term to the diagonal
    if not intercept:
        for i in range(p):
            G[i, i] += alpha
    else:
        for i in range(1, p):
            # Only add regularization to the non-intercept coefficients
            G[i, i] += alpha
    try:
        smoother = solve(G, G_unpenalized)
        dof = float64(0.0)
        for i in range(p):
            dof += smoother[i, i]
        L = cholesky(G)
        z = solve(L, g)
        coef: NDF = asarray(solve(L.T, z), dtype=float64)
        return coef, L, dof, OK  # pyright: ignore
    except Exception:
        return (
            np.full(p, np.nan, dtype=float64),
            np.empty((0, 0), dtype=float64),
            float64(np.nan),
            RANK_DEFICIENT,
        )


@njit(cache=True)
def lstsq_solve(X: NDF, y: NDF) -> tuple[NDF, NDF, int]:
    coef, _, rank, _ = lstsq(X, y)
    L = np.empty((0, 0), dtype=float64)

    if rank < X.shape[1]:
        return asarray(coef, dtype=float64), L, RANK_DEFICIENT
    return asarray(coef, dtype=float64), L, OK
