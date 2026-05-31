from __future__ import annotations

from typing import Callable

from numpy import float64, ndarray

from ..core.solved_model import SolvedModel
from .mc_constructs import DataGenReturn, NDF, ShockMapping
from .operations import simulate_dgp


class MCReferenceConstruct:
    """
    Construct container for Monte Carlo reference data generation via simulation of a solved model.
    """

    def __init__(self, model: SolvedModel, T: int, N: int) -> None:
        self._model = model
        self._T = T
        self._N = N

    def _data_from_sim(
        self,
        *,
        shocks: ShockMapping | None = None,
        shock_scale: float | float64 = 1.0,
        x0: ndarray | None = None,
        observables: bool = True,
    ) -> DataGenReturn:
        data = simulate_dgp(
            reference=self.model,
            dgp=self.model,
            rep_idx=0,
            T=self.T,
            shocks=shocks,
            shock_scale=shock_scale,
            x0=x0,
            observables=observables,
        )
        return DataGenReturn(
            state_mat=data.states,
            obs_mat=data.observables,
            n_exog=data.n_exog,
        )

    @property
    def model(self) -> SolvedModel:
        return self._model

    @property
    def T(self) -> int:
        return self._T

    @property
    def N(self) -> int:
        return self._N

    class DataGeneratingCallable:
        """Simple callable wrapper for external data-generating functions."""

        def __init__(
            self, func: Callable[[int], tuple[NDF, NDF | None]], T: int, N: int
        ) -> None:
            self._func = func
            self._T = T
            self._N = N

        def __call__(self) -> DataGenReturn:
            state_mat, obs_mat = self.func(self.T)
            return DataGenReturn(
                state_mat=state_mat,
                obs_mat=obs_mat,
                n_exog=-1,
            )

        @property
        def func(self) -> Callable[[int], tuple[NDF, NDF | None]]:
            return self._func

        @property
        def T(self) -> int:
            return self._T

        @property
        def N(self) -> int:
            return self._N
