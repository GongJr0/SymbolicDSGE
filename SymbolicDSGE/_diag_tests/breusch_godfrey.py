from __future__ import annotations

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from numba import njit

from ..regression.solvers import chol_solve, lstsq_solve

from .status import TestStatus
from .result import TestResult
from .distributions import PvalMethod, ReferenceDistribution
from ._native import native as _native, DIAG_FALLBACK

NDF = NDArray[float64]

OK = int(TestStatus.OK)
UDEF_VARIANCE = int(TestStatus.UDEF_VARIANCE)
BAD_SHAPE = int(TestStatus.BAD_SHAPE)
INSUFFICIENT_SAMPLES = int(TestStatus.INSUFFICIENT_SAMPLES)


@njit(cache=True)
def build_design_matrix(X: NDF, eps: NDF, lags: int) -> NDF:
    N, K = X.shape
    out = np.empty((N, K + lags + 1), dtype=float64)

    out[:, 0] = 1.0  # Intercept
    out[:, 1 : K + 1] = X

    for lag in range(1, lags + 1):
        col = K + lag
        out[:lag, col] = 0.0
        out[lag:, col] = eps[:-lag]

    return out


@njit(cache=True)
def _bg_stat_numba(eps: NDF, X: NDF, lags: int) -> tuple[int, float64]:
    n = eps.size
    if n <= lags:
        return INSUFFICIENT_SAMPLES, float64(np.nan)
    if X.shape[0] != n:
        return BAD_SHAPE, float64(np.nan)

    design = build_design_matrix(X, eps, lags)

    try:
        bhat, _, _ = chol_solve(design, eps)
    except Exception:
        bhat, _, _ = lstsq_solve(design, eps)

    resid = eps - design @ bhat
    tss = np.sum(eps**2)
    stat = n * (1.0 - np.sum(resid**2) / tss) if tss > 0.0 else 0.0

    return OK, float64(stat)


def bg_stat(eps: NDF, X: NDF, lags: int) -> tuple[int, float64]:
    """Breusch-Godfrey LM statistic; native fast path, numba fallback.

    The native kernel handles the full-rank case; on a rank-deficient design it
    returns ``DIAG_FALLBACK`` and we recompute through the numba kernel (which
    has the lstsq fallback).
    """
    if _native is not None and eps.size > lags and X.shape[0] == eps.size:
        status, stat = _native.bg_stat(
            np.ascontiguousarray(eps, dtype=np.float64),
            np.ascontiguousarray(X, dtype=np.float64),
            lags,
        )
        if status != DIAG_FALLBACK:
            return status, float64(stat)
    nb_status, nb_stat = _bg_stat_numba(eps, X, lags)
    return int(nb_status), float64(nb_stat)


def breusch_godfrey(
    eps: NDF,
    X: NDF,
    lags: int,
    alpha: float = 0.05,
    _auto_pval: bool = True,
) -> TestResult:
    status, stat = bg_stat(eps, X, lags)
    return TestResult(
        test_name="breusch_godfrey",
        statistic=stat,
        status=TestStatus(status),
        df=lags,
        dist=ReferenceDistribution.CHI2,
        pval_method=PvalMethod.SF,
        alpha=float64(alpha),
        _auto_pval=_auto_pval,
    )
