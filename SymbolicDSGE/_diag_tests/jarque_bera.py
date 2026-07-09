from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from numba import njit

from .distributions import PvalMethod, ReferenceDistribution
from .result import TestResult
from .status import TestStatus

from ._native import native as _native

OK = int(TestStatus.OK)
UDEF_VARIANCE = int(TestStatus.UDEF_VARIANCE)
BAD_SHAPE = int(TestStatus.BAD_SHAPE)
INSUFFICIENT_SAMPLES = int(TestStatus.INSUFFICIENT_SAMPLES)

NDF: TypeAlias = NDArray[float64]


@njit(cache=True)
def jb_stat(x: NDF) -> tuple[int, float64]:
    """Calculate the Jarque-Bera test statistic."""
    if x.ndim != 1:
        return BAD_SHAPE, float64(np.nan)

    n = x.size
    if n == 0:
        return INSUFFICIENT_SAMPLES, float64(np.nan)

    mean = 0.0
    for i in range(n):
        mean += x[i]
    mean /= n

    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    for i in range(n):
        centered = x[i] - mean
        centered2 = centered * centered
        m2 += centered2
        m3 += centered2 * centered
        m4 += centered2 * centered2
    m2 /= n
    m3 /= n
    m4 /= n

    # A non-finite m2 means the input carried NaN/inf (e.g. log of a non-positive
    # value upstream). `m2 <= 0.0` alone can't catch it: every comparison with NaN
    # is False, so a NaN would otherwise slip through to an OK/NaN statistic.
    if not np.isfinite(m2) or m2 <= 0.0:
        return UDEF_VARIANCE, float64(np.nan)

    skew = m3 / m2**1.5
    kurt = m4 / m2**2.0
    kurt_minus_3 = kurt - 3.0

    inner = (skew * skew) / 6.0 + (kurt_minus_3 * kurt_minus_3) / 24.0
    if n >= 10:
        return OK, n * inner
    else:
        return INSUFFICIENT_SAMPLES, n * inner


def jarque_bera(
    x: NDF,
    alpha: float = 0.05,
    _auto_pval: bool = True,
) -> TestResult:
    """Perform the Jarque-Bera test for normality."""
    if x.ndim != 1:
        status, stat = BAD_SHAPE, float64(np.nan)
    else:
        # Coerce once to canonical f64/C-contiguous so both backends agree (the
        # native shim requires it; the numba kernel computes in the input dtype).
        xc = np.ascontiguousarray(x, dtype=float64)
        if _native is not None:
            status, stat = _native.jb_stat(xc)
        else:
            status, stat = jb_stat(xc)
    return TestResult(
        test_name="jarque_bera",
        dist=ReferenceDistribution.JB_LOOKUP,
        df=x.size,
        pval_method=PvalMethod.SF,
        alpha=float64(alpha),
        statistic=stat,
        status=TestStatus(status),
        _auto_pval=_auto_pval,
    )
