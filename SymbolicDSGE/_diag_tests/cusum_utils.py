from __future__ import annotations

from math import sqrt
from typing import TypeAlias

import numpy as np
from numba import njit
from numpy import float64
from numpy.typing import NDArray

from .status import TestStatus

NDF: TypeAlias = NDArray[float64]

# Test Status
OK = int(TestStatus.OK)
BAD_SHAPE = int(TestStatus.BAD_SHAPE)
INSUFFICIENT_SAMPLES = int(TestStatus.INSUFFICIENT_SAMPLES)


@njit(cache=True)
def recursive_residuals(y: NDF, X: NDF) -> tuple[int, NDF]:
    """Brown-Durbin-Evans recursive residuals (the ``w`` series).

    Shared kernel for the recursive-residual diagnostics. The standardized CUSUM
    statistic accumulates ``|w|`` against a moving boundary, while the CUSUM of
    squares accumulates ``w**2``; both start from this series, so the (more
    expensive) recursion lives here and each variant applies its own functional
    of ``w`` on top.

    Returns ``(status, w)`` where ``w`` has length ``T - p`` on success and is
    empty on a shape/sample failure.
    """
    T, p = X.shape
    if T == 0 or T <= p:
        return INSUFFICIENT_SAMPLES, np.empty(0, dtype=float64)
    if y.size != T:
        return BAD_SHAPE, np.empty(0, dtype=float64)

    # Seed the recursion from the first p observations: P = (X_p' X_p)^-1 and
    # beta = P X_p' y_p. The Gram matrix and X_p' y_p are accumulated by direct
    # indexing (no slice views), leaving only a single small p x p inverse.
    G = np.zeros((p, p), dtype=float64)
    Xty = np.zeros(p, dtype=float64)
    for r in range(p):
        yr = y[r]
        for a in range(p):
            xra = X[r, a]
            Xty[a] += xra * yr
            for b in range(p):
                G[a, b] += xra * X[r, b]
    P = np.linalg.inv(G)

    beta = np.zeros(p, dtype=float64)
    for a in range(p):
        s = 0.0
        for b in range(p):
            s += P[a, b] * Xty[b]
        beta[a] = s

    # Recursive residuals. P stays symmetric (a symmetric rank-1 downdate of a
    # symmetric matrix), so Px == P @ x_t == x_t @ P is computed once per step
    # and reused for the quadratic form, the downdate, and the beta update.
    # Manual loops avoid per-iteration BLAS dispatch, which dominates at the
    # small p typical of CUSUM regressions.
    rec_resid = np.empty(T - p, dtype=float64)
    Px = np.empty(p, dtype=float64)
    for i in range(T - p):
        xt = X[p + i]

        e = y[p + i]
        for a in range(p):
            e -= xt[a] * beta[a]

        quad = 0.0
        for a in range(p):
            s = 0.0
            for b in range(p):
                s += P[a, b] * xt[b]
            Px[a] = s
            quad += xt[a] * s

        ft = 1.0 + quad
        rec_resid[i] = e / sqrt(ft)

        inv_ft = 1.0 / ft
        coef = e * inv_ft
        for a in range(p):
            pa = Px[a]
            beta[a] += pa * coef
            for b in range(p):
                P[a, b] -= pa * Px[b] * inv_ft

    return OK, rec_resid
