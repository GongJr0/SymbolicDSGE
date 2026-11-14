"""Result containers for Bayesian and frequentist estimation of DSGE models."""

from .optimization import OptimizationResult, MLEResult, MAPResult
from .mcmc import MCMCResult, MultiChainResult

__all__ = [
    "OptimizationResult",
    "MLEResult",
    "MAPResult",
    "MCMCResult",
    "MultiChainResult",
]
