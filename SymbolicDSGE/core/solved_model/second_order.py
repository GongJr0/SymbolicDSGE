"""Second-order (SGU) solved model."""

from __future__ import annotations
from typing import Mapping, Callable, Union
from numpy import float64
import numpy as np

from .base import SolvedModel, NDF
from .shocks import simulation_shock_matrix
from ..solver_backend import SecondOrderSolution
from ..compiled_model import CompiledModel
from ..sim_result import StatePath
from ..shock_generators import Shock
from ..._ckernels.core import simulate_second_order_pruned


class SecondOrderSolvedModel(SolvedModel[SecondOrderSolution]):
    """A model solved to second order: the first-order rule plus corrections."""

    def __init__(self, compiled: CompiledModel, policy: SecondOrderSolution) -> None:
        """Initialize a second-order solved model.

        Args:
            model: The solved model.
            solution: The second-order solution.
        """
        super().__init__(compiled, policy)

    def _simulate_state_matrix(
        self,
        T: int,
        shocks: (
            Mapping[str, Shock | Union[Callable[[float | NDF], NDF], NDF]] | None
        ) = None,
        shock_scale: float = 1,
        x0: NDF | None = None,
    ) -> StatePath:
        n_state = self.compiled.n_state

        n = self.compiled.n_var
        ny = self.compiled.n_ctrl
        policy = self.policy
        ss = policy.steady_state
        ss_state = ss[:n_state]

        if x0 is None:
            x0_state = ss_state
        else:
            x0_state = self._simulation_initial_state(policy.f, x0)[:n_state]
        x0_dev = x0_state - ss_state
        shock_mat = simulation_shock_matrix(
            self.compiled,
            T=T,
            shocks=shocks,
            shock_scale=shock_scale,
        )

        x_path, y_path = simulate_second_order_pruned(
            policy.p,
            policy.f,
            policy.B[:n_state, :],
            policy.hxx,
            policy.gxx,
            policy.hss,
            policy.gss,
            x0_dev,
            shock_mat,
        )

        X = np.empty((T, n), dtype=float64)
        X[:, :n_state] = x_path + ss_state
        if ny > 0:
            X[:, n_state:] = y_path + ss[n_state:]
        return StatePath(X)
