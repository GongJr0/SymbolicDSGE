"""SymbolicDSGE: A broad DSGE modeling library.

This package provides YAML-based model specifications, symbolic parsing and manipulation,
linearization, first- and second-order solution methods, OBCs via OccBin, simulations/IRFs,
Bayesian and frequentist estimation, and a Monte Carlo experiment framework.

Additionally, the library allows bundling complete experiments into single files for easy sharing and reproducibility.
A web-based GUI also allows elementary specification, solution, simulation, estimation, and Monte Carlo experiment execution
without writing any code (beyond the YAML model specification).
"""

from .core import (
    DSGESolver,
    ModelConfig,
    ModelParser,
    Shock,
    linearize_model,
    SolvedModel,
)
from .kalman import KalmanFilter
from .estimation import Estimator
from .bundle import BundleBuilder, build_from as load_bundle
from . import utils

__all__ = [
    "ModelConfig",
    "ModelParser",
    "DSGESolver",
    "SolvedModel",
    "KalmanFilter",
    "Estimator",
    "Shock",
    "utils",
    "linearize_model",
    # .sdsge bundle API
    "BundleBuilder",
    "load_bundle",
]
