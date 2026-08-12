"""Piecewise-linear (OccBin) solved model."""

from __future__ import annotations

from .base import SolvedModel
from ..solver_backend import PiecewiseSolution
from ..compiled_model import CompiledModel


class PiecewiseSolvedModel(SolvedModel[PiecewiseSolution]):
    """A model whose policy is path dependent, one rule per date and regime."""

    def __init__(self, compiled: CompiledModel, policy: PiecewiseSolution) -> None:
        """Initialize a piecewise solved model.

        Args:
            model: The solved model.
            solution: The piecewise solution.
        """
        super().__init__(compiled, policy)
