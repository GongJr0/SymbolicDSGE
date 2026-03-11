import warnings
import numba
from sympy import Symbol, Expr

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from numba import njit

from dataclasses import dataclass, asdict
from functools import cached_property
from typing import Callable, Any, cast
from textwrap import dedent

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

    calib_params: list[Symbol]

    observable_names: list[str]
    observable_eqs: list[Expr]
    observable_funcs: list[Callable]
    observable_jacobian: Callable[..., ND]

    n_state: int
    n_exog: int

    @cached_property
    def _measurement_vector_func(self) -> Callable[..., ND]:
        params = list(map(str, [*self.var_names, *self.calib_params]))
        params_typed = ", ".join(f"{p}: float64" for p in params)
        arg_names = ", ".join(params)

        lines = [
            f"out[{i}] = func_{i}({arg_names})"
            for i in range(len(self.observable_funcs))
        ]
        body = "\n    ".join(lines)

        func_str = f"""
def vectorized_measurements({params_typed}):
    out = np.zeros(({len(self.observable_funcs)},), dtype=np.float64)
    {body}
    return out
"""

        ns = {"np": np, "float64": float64}
        for i, fn in enumerate(self.observable_funcs):
            ns[f"func_{i}"] = fn

        exec(dedent(func_str), ns)
        f = njit(ns["vectorized_measurements"])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=numba.errors.NumbaExperimentalFeatureWarning
            )
            f.compile(tuple(numba.float64 for _ in params))
        return cast(Callable, f)

    def construct_measurement_vector_func(self) -> Callable[..., ND]:
        # Building the vectorized measurement function triggers Numba compilation.
        # Cache the dispatcher so extended-mode likelihood evaluation does not
        # rebuild and recompile it on every call.
        return self._measurement_vector_func

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
