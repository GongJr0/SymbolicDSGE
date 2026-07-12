from __future__ import annotations

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from .distributions import PvalMethod, ReferenceDistribution
from .result import TestResult
from .status import TestStatus

from .._ckernels.diag import lb_stat

NDF = NDArray[float64]

BAD_SHAPE = int(TestStatus.BAD_SHAPE)


def ljung_box(
    x: NDF, L: int, alpha: float = 0.05, _auto_pval: bool = True
) -> TestResult:
    """Ljung-Box test for x up to lag L."""
    if x.ndim != 1:
        err, stat = (BAD_SHAPE, float64(np.nan))
    else:
        # Coerce once to canonical f64/C-contiguous (the native shim requires it).
        xc = np.ascontiguousarray(x, dtype=float64)
        err, stat = lb_stat(xc, L)
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
        _auto_pval=_auto_pval,
    )
