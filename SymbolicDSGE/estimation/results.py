from dataclasses import dataclass

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

NDF = NDArray[np.float64]


@dataclass(frozen=True)
class OptimizationResult:
    kind: str
    x: NDF
    theta: dict[str, float64]
    success: bool
    message: str
    fun: float64
    loglik: float64
    logprior: float64
    logpost: float64
    nfev: int
    nit: int | None
    raw: OptimizeResult


@dataclass(frozen=True)
class MCMCResult:
    param_names: list[str]
    samples: NDF
    logpost_trace: NDF
    accept_rate: float64
    n_draws: int
    burn_in: int
    thin: int
