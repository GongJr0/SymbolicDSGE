from __future__ import annotations

from typing import Any, Callable

from .ols_result import OLSResult
from ..utils import process_args
from ..enums import RegressionStatus
from ..solvers import chol_solve, lstsq_solve, use_scalar_path
from ..._native_dispatch import FORCE_NUMBA, REQUIRE_NATIVE

import numpy as np
from numpy import float64
from numpy.typing import NDArray

NDF = NDArray[float64]

# Prefer the native OLS Cholesky solve; fall back to numba when the extension is
# not built (ALWAYS_USE_NUMBA / NEVER_USE_NUMBA override -- see _native_dispatch).
_ols_chol_solve_native: Callable[..., Any] | None
if FORCE_NUMBA:
    _ols_chol_solve_native = None
else:
    try:
        from ..._ckernels.regression import ols_chol_solve as _ols_chol_solve_native
    except ImportError:  # pragma: no cover - exercised only without the extension
        if REQUIRE_NATIVE:
            raise
        _ols_chol_solve_native = None


def ols(
    x: NDF,
    y: NDF,
    variables: list[str] | None = None,
    intercept: bool = True,
) -> OLSResult:
    X, y, variables = process_args(x, y, variables)

    var_names = list(variables)
    if intercept:
        X = np.hstack((np.ones((X.shape[0], 1), dtype=np.float64), X))
        var_names = ["Intercept", *var_names]

    n, p = X.shape
    if _ols_chol_solve_native is not None and use_scalar_path(n, p):
        coef, L, status = _ols_chol_solve_native(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
        )
    else:
        coef, L, status = chol_solve(X, y)
    if status != int(RegressionStatus.OK):
        coef, L, status = lstsq_solve(X, y)

    return OLSResult(
        variables=var_names,
        coefficients=coef,
        y=y,
        X=X,
        status=RegressionStatus(status),
        _L=L,
    )
