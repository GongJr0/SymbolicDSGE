from sympy import Symbol, Expr

from numpy import float64, zeros
from numpy.typing import NDArray
from numba import njit

from dataclasses import dataclass, asdict
from typing import Callable, Any

from .config import ModelConfig
from ..kalman.config import KalmanConfig

NDF = NDArray[float64]
ND = NDArray


@dataclass(frozen=True)
class CompiledModel:
    config: ModelConfig
    kalman: KalmanConfig | None

    cur_syms: list[Symbol]

    var_names: list[str]
    idx: dict[str, int]

    objective_eqs: list[Expr]
    objective_funcs: list[Callable]
    equations: Callable[[Any, Any, Any], ND]

    calib_params: list[str]

    observable_names: list[str]
    observable_eqs: list[Expr]
    observable_funcs: list[Callable]
    observable_jacobian: Callable[..., ND]

    n_state: int
    n_exog: int

    def construct_measurement_vector_func(self) -> Callable[..., ND]:
        # TODO: Make njit-compatible (eliminate *args -> produce str source code with arg-patching -> exec -> njit)

        funcs = [*self.observable_funcs]  # Redefine to get rid of the "self" reference

        def vectorized_measurements(*args: Any) -> ND:
            out = zeros((len(funcs),), dtype=float64)
            for i, func in enumerate(funcs):
                out[i] = func(*args)
            return out

        return vectorized_measurements

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
