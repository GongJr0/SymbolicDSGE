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

# Single crossover for "hand-rolled scalar kernels (numba manual / native C) beat
# BLAS/LAPACK". Used by every loop-vs-BLAS decision in the regression stack
# (xtx_xty's Gram branch, the ridge RSS branch, and the native-dispatch gate) so
# they can never drift apart -- and so native is only chosen where numba is also
# on its manual branch, keeping the two bit-parity.
#
# Both bounds must hold (logical AND): SCALAR_PATH_MAX_NP caps the O(n*p) /
# O(n*p^2) data-volume reduction (BLAS threads/vectorizes that), and
# SCALAR_PATH_MAX_P caps the dense linear algebra (the O(p^3) Cholesky and the
# p*p Gram, where LAPACK pulls far ahead). If either is exceeded, defer to BLAS.
SCALAR_PATH_MAX_NP = 1e6
SCALAR_PATH_MAX_P = 256


@njit(cache=True)
def use_scalar_path(n: int, p: int) -> bool:
    return n * p <= SCALAR_PATH_MAX_NP and p <= SCALAR_PATH_MAX_P


@njit(cache=True)
def xtx_xty(X: NDF, y: NDF) -> tuple[NDF, NDF]:
    n, p = X.shape

    if not use_scalar_path(n, p):
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
def _gram_is_pd(G: NDF) -> bool:
    """Deterministic positive-definiteness check for the Gram matrix.

    Runs a hand-rolled Cholesky purely to test the pivots against a relative
    floor (``scale * p * eps``). A rank-deficient Gram has an exact pivot of
    zero that rounds to a tiny value of either sign depending on the BLAS build,
    so relying on ``numpy.linalg.cholesky`` to raise (and on the caller's
    ``try/except`` to catch it) detects rank deficiency nondeterministically
    across builds. This gate makes the decision build-independent; when it
    passes, the fast LAPACK path below is used unchanged, so positive-definite
    results stay bit-for-bit identical. Mirrors the native ``sdsge_chol``.
    """
    p = G.shape[0]
    scale = float64(0.0)
    for i in range(p):
        if G[i, i] > scale:
            scale = G[i, i]
    pivot_tol = scale * p * np.finfo(float64).eps
    L = np.zeros((p, p), dtype=float64)
    for i in range(p):
        for j in range(i + 1):
            s = G[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= pivot_tol:
                    return False
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return True


@njit(cache=True)
def chol_solve(X: NDF, y: NDF) -> tuple[NDF, NDF, int]:
    G, g = xtx_xty(X, y)

    if not _gram_is_pd(G):
        p = G.shape[0]
        return (
            np.full(p, np.nan, dtype=float64),
            np.empty((0, 0), dtype=float64),
            RANK_DEFICIENT,
        )

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
