from __future__ import annotations

from .ols_result import OLSResult, Status
from .solvers import chol_solve, ltsq_solve

from numpy import float64
from numpy.typing import NDArray

NDF = NDArray[float64]


def ols(x: NDF, y: NDF, variables: list[str] | None = None) -> OLSResult:
    if variables is None:
        variables = [f"x{i}" for i in range(x.shape[1])]

    try:
        coef, L, status = chol_solve(x, y)
    except Exception:
        coef, L, status = ltsq_solve(x, y)

    return OLSResult(
        variables=variables,
        coefficients=coef,
        y=y,
        x=x,
        status=Status(status),
        _L=L,
    )
