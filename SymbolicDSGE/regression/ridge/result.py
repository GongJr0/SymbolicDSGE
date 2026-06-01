from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from enum import StrEnum

import numpy as np
from numpy import float64

from ..result import RegressionResult


class RidgeObjective(StrEnum):
    AIC = "aic"
    BIC = "bic"
    LOSS = "loss"


@dataclass(frozen=True)
class RidgeResult(RegressionResult):
    alpha: float64
    effective_dof: float64
    intercept: bool = False
    objective: RidgeObjective | None = None
    objective_value: float64 | None = None

    @cached_property
    def l2_penalty(self) -> float64:
        coef = self.coefficients[1:] if self.intercept else self.coefficients
        return float64(self.alpha * np.dot(coef, coef))
