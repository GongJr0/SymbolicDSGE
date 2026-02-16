from sympy import Symbol, Expr

from numpy import float64
from numpy.typing import NDArray


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

    observable_names: list[str]
    observable_eqs: list[Expr]
    observable_funcs: list[Callable]

    n_state: int
    n_exog: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
