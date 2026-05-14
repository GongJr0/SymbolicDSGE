from ..core.solved_model import SolvedModel
from ..kalman.filter import FilterResult

import numpy as np
from numpy import float64, ndarray
from numpy.typing import NDArray

from typing import TypedDict, Unpack, Mapping, Union, Callable
from dataclasses import dataclass, asdict

NDF = NDArray[float64]


class SimSpec(TypedDict):
    shocks: Mapping[str, Union[Callable[[float | NDF], NDF], NDF]] | None
    shock_scale: float | float64
    x0: ndarray | None
    observables: bool


class SimKwargs(TypedDict):
    T: int
    shocks: Mapping[str, Union[Callable[[float | NDF], NDF], NDF]] | None
    shock_scale: float | float64
    x0: ndarray | None
    observables: bool


_DEFAULT_SIM_SPEC: SimSpec = {
    "shocks": None,
    "shock_scale": 1.0,
    "x0": None,
    "observables": True,
}


@dataclass
class DataGenReturn:
    """Data return type compatible with both simulated data and Kalman filter output."""

    state_mat: NDF
    obs_mat: NDF | None
    n_exog: int  # [:n_exog] to get exogenous shocks by convention.


class MCReferenceConstruct:
    """
    Construct container for Monte Carlo reference data generation via simulation of a solved model.
    """

    def __init__(self, model: SolvedModel, T: int, N: int) -> None:
        self._model = model
        self._T = T
        self._N = N

    def _data_from_sim(self, **spec: Unpack[SimSpec]) -> DataGenReturn:
        _spec = _DEFAULT_SIM_SPEC | spec
        _kwargs: SimKwargs = {
            "T": self.T,
            **_spec,
        }
        sim_data = self.model.sim(**_kwargs)

        # Collect states
        states = sim_data["_X"]

        # Get observables
        obs_names = self.model.compiled.observable_names  # list[str] in canonical order
        obs_mat = np.zeros((self.T, len(obs_names))) if _spec["observables"] else None
        if obs_mat is not None:
            for i, obs in enumerate(obs_names):
                obs_mat[:, i] = sim_data[obs]

        n_exog = self.model.compiled.n_exog
        return DataGenReturn(
            state_mat=states,
            obs_mat=obs_mat,
            n_exog=n_exog,
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
        """Simple Callable wrapper for a function accepting"""

        def __init__(
            self, func: Callable[[int], tuple[NDF, NDF | None]], T: int, N: int
        ) -> None:
            self._func = func
            self._T = T
            self._N = N

        def __call__(self) -> DataGenReturn:
            state_mat, obs_mat = self.func(self.T)
            n_exog = (
                -1
            )  # n_exog is not applicable when a general process instead of a model is being called.

            return DataGenReturn(
                state_mat=state_mat,
                obs_mat=obs_mat,
                n_exog=n_exog,
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
