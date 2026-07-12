from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from .distributions import PvalMethod, ReferenceDistribution
from .result import TestResult
from .status import TestStatus

from .._ckernels.diag import jb_stat

BAD_SHAPE = int(TestStatus.BAD_SHAPE)

NDF: TypeAlias = NDArray[float64]


def jarque_bera(
    x: NDF,
    alpha: float = 0.05,
    _auto_pval: bool = True,
) -> TestResult:
    """Perform the Jarque-Bera test for normality."""
    if x.ndim != 1:
        status, stat = BAD_SHAPE, float64(np.nan)
    else:
        # Coerce once to canonical f64/C-contiguous (the native shim requires it).
        xc = np.ascontiguousarray(x, dtype=float64)
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
