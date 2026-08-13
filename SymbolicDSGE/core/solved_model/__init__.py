"""The solved-model hierarchy: one class per solution kind.

``SolvedModel`` is the policy-independent base. ``sim`` and ``irf`` dispatch
through it, so a caller holds the subclass its solve produced and never asks
which one it is.
"""

from .base import SolvedModel
from .first_order import FirstOrderSolvedModel
from .piecewise import PiecewiseSolvedModel
from .second_order import SecondOrderSolvedModel

__all__ = [
    "SolvedModel",
    "FirstOrderSolvedModel",
    "SecondOrderSolvedModel",
    "PiecewiseSolvedModel",
]
