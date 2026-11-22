"""Coverage for estimator module-level helpers and small edge branches."""

from __future__ import annotations

import numpy as np
import pytest
from numpy import float64

from SymbolicDSGE.estimation.results import MCMCResult, MLEResult

from tests._result_helpers import make_mcmc_result


def _opt_result(**cfg) -> MLEResult:
    return MLEResult(
        x=np.zeros(1, dtype=float64),
        theta={"a": float64(1.0)},
        success=True,
        message="",
        fun=float64(0.0),
        nfev=1,
        nit=1,
        optimizer_config=dict(cfg),
        loglik=float64(0.0),
    )


def _mcmc_result() -> MCMCResult:
    return make_mcmc_result(
        param_names=["a"],
        samples=np.zeros((1, 2, 1), dtype=float64),
        sampler_config={"adapt": True},
    )
