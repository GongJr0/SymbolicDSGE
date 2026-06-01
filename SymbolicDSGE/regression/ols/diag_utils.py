from __future__ import annotations

import numpy as np
from numpy import float64, asarray
from numpy.typing import NDArray
from numba import njit
from ..result import r2, r2_adj

NDF = NDArray[float64]


@njit(cache=True)
def se_from_cholesky(L: NDF, y: NDF, y_hat: NDF) -> NDF:
    n = y.shape[0]
    k = L.shape[0]
    eps = y - y_hat
    sigma2 = (eps**2).sum() / (n - k)

    L_inv = np.linalg.solve(L, np.eye(k, dtype=float64))
    inv_diag = (L_inv * L_inv).sum(axis=0)
    return np.sqrt(inv_diag * sigma2)  # type: ignore[no-any-return] # MyPy complains, signature is correct


def se_from_pinv(x: NDF, y: NDF, y_hat: NDF) -> NDF:
    n = y.shape[0]
    rank = np.linalg.matrix_rank(x)
    df_resid = n - rank
    if df_resid <= 0:
        return np.full(x.shape[1], np.nan, dtype=np.float64)

    eps = y - y_hat
    sigma2 = (eps**2).sum() / df_resid
    cov_matrix = np.linalg.pinv(x.T @ x) * sigma2
    return asarray(np.sqrt(np.diag(cov_matrix)), dtype=float64)


def se(L: NDF, y: NDF, y_hat: NDF, x: NDF) -> NDF:
    if L.size == 0:
        return se_from_pinv(x, y, y_hat)

    return asarray(se_from_cholesky(L, y, y_hat), dtype=float64)
