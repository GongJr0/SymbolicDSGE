"""Estimation interface for DSGE models, including likelihood and bayesian methods.

Supports MLE, MAP, and Adative MH MCMC estimation via :class:`Estimator`.
Priors for bayesian estimation are constructed via :func:`make_prior`.
"""

from .estimator import Estimator
from ..bayesian import make_prior

__all__ = [
    "Estimator",
    "make_prior",
]
