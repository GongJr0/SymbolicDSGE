from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from enum import StrEnum

import numpy as np
from numpy import float64

from ..result import RegressionResult


class RidgeObjective(StrEnum):
    """Criterion a ridge grid search minimizes when selecting ``alpha``."""

    AIC = "aic"
    BIC = "bic"
    LOSS = "loss"


@dataclass(frozen=True)
class RidgeResult(RegressionResult):
    """L2-penalized regression result.

    Extends :class:`RegressionResult` with ridge diagnostics. When the design
    carries an intercept it is left unregularized and excluded from ``l2_penalty``.

    Attributes
    ----------
    alpha : float64
        Selected L2 penalty weight.
    effective_dof : float64
        Effective degrees of freedom used by the grid-search criteria.
    intercept : bool
        Whether the returned design includes an intercept column.
    objective : RidgeObjective | None
        Grid-search criterion used to select ``alpha``, when one was run.
    objective_value : float64 | None
        Realized grid-search criterion value, when one was run.
    """

    alpha: float64
    effective_dof: float64
    intercept: bool = False
    objective: RidgeObjective | None = None
    objective_value: float64 | None = None

    @cached_property
    def l2_penalty(self) -> float64:
        """Realized ridge penalty, excluding the intercept term."""
        coef = self.coefficients[1:] if self.intercept else self.coefficients
        return float64(0.5 * self.alpha * np.dot(coef, coef))
