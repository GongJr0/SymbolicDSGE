import warnings
import numba
from sympy import Symbol, Expr

import numpy as np
from numpy import complex128, float64
from numpy.typing import NDArray
from numba import njit

from dataclasses import dataclass, asdict
from functools import cached_property
from typing import Callable, Any, Mapping, cast
from textwrap import dedent

from .config import ModelConfig
from ..kalman.config import KalmanConfig

NDF = NDArray[float64]
NDC = NDArray[complex128]
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

    calib_params: list[Symbol]

    observable_names: list[str]
    observable_eqs: list[Expr]
    observable_funcs: list[Callable]
    observable_jacobian: Callable[..., ND]

    n_state: int
    n_exog: int

    @cached_property
    def _objective_vector_func(self) -> Callable[..., ND]:
        call_args = ", ".join(
            [
                *(f"fwd[{i}]" for i in range(len(self.var_names))),
                *(f"cur[{i}]" for i in range(len(self.var_names))),
                *(f"params[{i}]" for i in range(len(self.calib_params))),
            ]
        )
        body = "\n    ".join(
            f"out[{i}] = func_{i}({call_args})"
            for i in range(len(self.objective_funcs))
        )

        func_str = f"""
def vectorized_objective(fwd, cur, params):
    out = np.empty(({len(self.objective_funcs)},), dtype=np.complex128)
    {body}
    return out
"""

        ns: dict[str, Any] = {"np": np}
        for i, fn in enumerate(self.objective_funcs):
            ns[f"func_{i}"] = fn

        exec(dedent(func_str), ns)
        f = njit(ns["vectorized_objective"])
        complex_vector = numba.types.Array(numba.complex128, 1, "C")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=numba.errors.NumbaExperimentalFeatureWarning
            )
            f.compile((complex_vector, complex_vector, complex_vector))
        return cast(Callable, f)

    def construct_objective_vector_func(self) -> Callable[..., ND]:
        # Building the vectorized objective function triggers Numba compilation.
        # Cache the dispatcher so solve/approximation can reuse the same kernel.
        return self._objective_vector_func

    def equations(
        self,
        fwd: Any,
        cur: Any,
        par: Mapping[str, float] | Any,
    ) -> ND:
        fwd_arr = np.ascontiguousarray(np.asarray(fwd, dtype=complex128).reshape(-1))
        cur_arr = np.ascontiguousarray(np.asarray(cur, dtype=complex128).reshape(-1))

        if isinstance(par, Mapping):
            par_vec = np.array(
                [par[p.name] for p in self.calib_params],
                dtype=complex128,
            )
        else:
            par_vec = np.ascontiguousarray(
                np.asarray(par, dtype=complex128).reshape(-1)
            )
            if par_vec.shape[0] != len(self.calib_params):
                raise ValueError(
                    f"Parameter vector length {par_vec.shape[0]} != {len(self.calib_params)}"
                )

        return self.construct_objective_vector_func()(fwd_arr, cur_arr, par_vec)

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
