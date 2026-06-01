from __future__ import annotations

from dataclasses import dataclass, field, asdict
from functools import cached_property

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from .enums import RegressionStatus

NDF = NDArray[float64]


def r2(y: NDF, y_hat: NDF) -> float64:
    ssr = ((y - y_hat) ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    return float64(1 - ssr / sst) if sst > 0 else float64(0.0)


def r2_adj(r2_value: float64, n: int, k: int) -> float64:
    if n <= k + 1:
        return float64(0.0)
    return float64(1 - (1 - r2_value) * (n - 1) / (n - k - 1))


@dataclass(frozen=True)
class RegressionResult:
    variables: list[str]
    coefficients: NDF

    # Raw Data
    y: NDF
    X: NDF
    n: int = field(init=False)
    k: int = field(init=False)

    # Meta
    status: RegressionStatus

    def __post_init__(self) -> None:
        y = np.ascontiguousarray(self.y, dtype=np.float64)
        x = np.ascontiguousarray(self.X, dtype=np.float64)
        coefficients = np.ascontiguousarray(self.coefficients, dtype=np.float64)
        variables = list(self.variables)

        if y.ndim != 1:
            raise ValueError("Regression response must be a 1D array.")
        if x.ndim != 2:
            raise ValueError("Regression design matrix must be a 2D array.")
        if coefficients.ndim != 1:
            raise ValueError("Regression coefficients must be a 1D array.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Regression response and design matrix row counts differ.")
        if coefficients.shape != (x.shape[1],):
            raise ValueError("Regression coefficient count must match design columns.")
        if len(variables) != x.shape[1]:
            raise ValueError("Regression variables must match design columns.")

        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "X", x)
        object.__setattr__(self, "n", y.shape[0])
        object.__setattr__(self, "k", x.shape[1])
        object.__setattr__(self, "status", RegressionStatus(self.status))

    @cached_property
    def y_hat(self) -> NDF:
        return np.asarray(self.X @ self.coefficients, dtype=np.float64)

    @property
    def x(self) -> NDF:
        return self.X

    @cached_property
    def residuals(self) -> NDF:
        return np.asarray(self.y - self.y_hat, dtype=np.float64)

    @cached_property
    def ssr(self) -> float64:
        return float64((self.residuals**2).sum())

    @cached_property
    def sst(self) -> float64:
        centered = self.y - self.y.mean()
        return float64((centered**2).sum())

    @cached_property
    def mse(self) -> float64:
        return float64(self.ssr / self.n)

    @cached_property
    def rmse(self) -> float64:
        return float64(np.sqrt(self.mse))

    @cached_property
    def r2(self) -> float64:
        return r2(self.y, self.y_hat)

    @cached_property
    def r2_adj(self) -> float64:
        return r2_adj(self.r2, self.n, self.k)

    def to_dict(self) -> dict:
        return asdict(self)
