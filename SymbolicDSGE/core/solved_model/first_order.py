"""First-order (Klein) solved model."""

from __future__ import annotations
from typing import Mapping, Callable, Union
from numpy import float64, ndarray
import numpy as np

from .base import SolvedModel, NDF
from .shocks import simulation_shock_matrix
from ..solver_backend import FirstOrderSolution
from ..compiled_model import CompiledModel
from ..sim_result import StatePath
from ..shock_generators import Shock

from ..._ckernels.core import simulate_linear_states_into


class FirstOrderSolvedModel(SolvedModel[FirstOrderSolution]):
    """A model solved to first order: one time-invariant rule."""

    def __init__(self, compiled: CompiledModel, policy: FirstOrderSolution) -> None:
        """Initialize a first-order solved model.

        Args:
            model: The solved model.
            solution: The first-order solution.
        """
        super().__init__(compiled, policy)

    def _simulate_state_matrix(
        self,
        T: int,
        shocks: (
            Mapping[str, Shock | Union[Callable[[float | NDF], NDF], NDF]] | None
        ) = None,
        shock_scale: float = 1,
        x0: dict[str, float | float64] | list[float | float64] | ndarray | None = None,
    ) -> StatePath:
        x0_arr = self._initial_state(x0)
        shock_mat = simulation_shock_matrix(
            self.compiled,
            T=T,
            shocks=shocks,
            shock_scale=shock_scale,
        )
        X = np.empty((T, self.compiled.n_var), dtype=float64)
        simulate_linear_states_into(
            self.policy.A,
            self.policy.B,
            x0_arr,
            shock_mat,
            X,
        )
        return StatePath(X)
