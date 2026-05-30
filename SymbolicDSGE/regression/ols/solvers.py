from __future__ import annotations

from numba import njit
import numpy as np
from numpy import float64, asarray
from numpy.linalg import cholesky, lstsq, solve
from numpy.typing import NDArray

NDF = NDArray[float64]

OK = 0
RANK_DEFICIENT = -1


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
def ltsq_solve(X: NDF, y: NDF) -> tuple[NDF, NDF, int]:
    coef, _, rank, _ = lstsq(X, y)
    L = np.empty((0, 0), dtype=float64)
    if rank < X.shape[1]:
        return asarray(coef, dtype=float64), L, RANK_DEFICIENT
    return asarray(coef, dtype=float64), L, OK
