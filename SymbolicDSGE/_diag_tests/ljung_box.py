from __future__ import annotations

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from numba import njit

from .distributions import PvalMethod, ReferenceDistribution
from .result import TestResult
from .status import TestStatus

NDF = NDArray[float64]
LOOP_LIMIT_N = 1e6

OK = int(TestStatus.OK)
UDEF_VARIANCE = int(TestStatus.UDEF_VARIANCE)
BAD_SHAPE = int(TestStatus.BAD_SHAPE)
BAD_LAG = int(TestStatus.BAD_LAG)


@njit(cache=True)
def acorr(x: NDF, L: int) -> tuple[int, NDF]:
    """Autocorrelation of x up to lag L."""
    n = x.size
    out = np.empty(L + 1, dtype=float64)

    mu = 0.0
    denom = 0.0
    z = np.empty_like(x)

    if n <= LOOP_LIMIT_N:
        for i in range(n):
            mu += x[i]
        mu /= n

        for i in range(n):
            z[i] = x[i] - mu
            denom += z[i] * z[i]

    else:
        mu = x.mean()
        z = x - mu
        denom = np.dot(z, z)

    if denom <= 0.0:
        for ell in range(L + 1):
            out[ell] = np.nan
        return UDEF_VARIANCE, out

    out[0] = 1.0  # L0

    for ell in range(1, L + 1):
        num = 0.0
        for t in range(ell, n):
            num += z[t] * z[t - ell]
        out[ell] = num / denom
    return OK, out


@njit(cache=True)
def lb_stat(x: NDF, L: int) -> tuple[int, float64]:
    """Ljung-Box test statistic for x up to lag L."""
    if x.ndim != 1:
        return BAD_SHAPE, float64(np.nan)

    n = x.size
    if n <= 1:
        return UDEF_VARIANCE, float64(np.nan)

    if L >= n:
        L = n - 1
    if L <= 0:
        return BAD_LAG, float64(np.nan)

    err, rho = acorr(x, L)
    if err != OK:
        return err, float64(np.nan)  # type(np.nan) == float

    n = x.size
    stat = 0.0
    for ell in range(1, L + 1):
        stat += (rho[ell] * rho[ell]) / (n - ell)
    stat *= n * (n + 2)
    return OK, float64(stat)


def ljung_box(x: NDF, L: int, alpha: float = 0.05) -> TestResult:
    """Ljung-Box test for x up to lag L."""
    err, stat = lb_stat(x, L)
    df = L
    if x.ndim == 1 and x.size > 1 and L >= x.size:
        df = x.size - 1
    name = f"Ljung-Box (L={df})"
    return TestResult(
        test_name=name,
        statistic=stat,
        dist=ReferenceDistribution.CHI2,
        pval_method=PvalMethod.SF,
        df=df,
        alpha=float64(alpha),
        status=TestStatus(err),
        _auto_pval=True,
    )
