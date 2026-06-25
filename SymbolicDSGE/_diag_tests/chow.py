from __future__ import annotations

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from numba import njit

from .status import TestStatus
from .result import TestResult
from .distributions import PvalMethod, ReferenceDistribution
from ..regression.solvers import chol_solve, lstsq_solve
from ._native import native as _native, DIAG_FALLBACK

NDF = NDArray[float64]

OK = int(TestStatus.OK)
LINALG = int(TestStatus.LINALG)
BAD_SHAPE = int(TestStatus.BAD_SHAPE)
INSUFFICIENT_SAMPLES = int(TestStatus.INSUFFICIENT_SAMPLES)
BAD_PARAMETER = int(TestStatus.BAD_PARAMETER)


def _chow_stat(y: NDF, X: NDF, t_break: int) -> tuple[int, float64]:
    """Chow break-point F statistic; native fast path, numba fallback.

    The native kernel returns ``DIAG_FALLBACK`` if any of the three regressions
    is rank-deficient, in which case the numba kernel (with its lstsq fallback)
    recomputes the statistic.
    """
    if _native is not None and y.size == X.shape[0]:
        status, stat = _native.chow_stat(
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(X, dtype=np.float64),
            t_break,
        )
        if status != DIAG_FALLBACK:
            return status, float64(stat)
    nb_status, nb_stat = _chow_stat_numba(y, X, t_break)
    return int(nb_status), float64(nb_stat)


@njit(cache=True)
def _chow_stat_numba(y: NDF, X: NDF, t_break: int) -> tuple[int, float64]:
    if y.size != X.shape[0]:
        return BAD_SHAPE, float64(np.nan)

    T, p = X.shape

    if T <= 2 * p:
        return INSUFFICIENT_SAMPLES, float64(np.nan)

    if t_break <= 0 or t_break >= T:
        return BAD_PARAMETER, float64(np.nan)

    X1 = np.ascontiguousarray(X[:t_break])
    y1 = np.ascontiguousarray(y[:t_break])

    X2 = np.ascontiguousarray(X[t_break:])
    y2 = np.ascontiguousarray(y[t_break:])

    statuses = np.empty(3, dtype=np.int32)
    tss = np.empty(3, dtype=float64)

    for i, (Xi, yi) in enumerate([(X, y), (X1, y1), (X2, y2)]):
        try:
            beta, _, status = chol_solve(Xi, yi)
        except Exception:
            beta, _, status = lstsq_solve(Xi, yi)

        statuses[i] = status
        tss_i = 0.0
        for j in range(Xi.shape[0]):
            resid = yi[j] - np.dot(Xi[j], beta)
            tss_i += resid * resid
        tss[i] = tss_i

    tss_c, tss_1, tss_2 = tss

    num = (tss_c - (tss_1 + tss_2)) / p
    denom = (tss_1 + tss_2) / (T - 2 * p)
    stat = num / denom

    if np.any(statuses != OK):
        return LINALG, stat
    return OK, stat


def chow(
    y: NDF, X: NDF, t_break: int, alpha: float = 0.05, _auto_pval: bool = True
) -> TestResult:
    T, p = X.shape
    df1 = p
    df2 = T - 2 * p

    status, stat = _chow_stat(y, X, t_break)
    return TestResult(
        test_name="chow",
        statistic=stat,
        df=(df1, df2),
        status=TestStatus(status),
        alpha=float64(alpha),
        dist=ReferenceDistribution.F,
        pval_method=PvalMethod.SF,
        _auto_pval=_auto_pval,
    )
