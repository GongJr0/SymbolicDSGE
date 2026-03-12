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

    def _coerce_param_vector(
        self,
        par: Mapping[Any, Any] | Any,
        *,
        dtype: Any,
    ) -> ND:
        if isinstance(par, Mapping):
            vals = []
            for p in self.calib_params:
                if p in par:
                    vals.append(par[p])
                elif p.name in par:
                    vals.append(par[p.name])
                else:
                    raise KeyError(f"Missing parameter '{p.name}'.")
            return np.ascontiguousarray(np.asarray(vals, dtype=dtype).reshape(-1))

        return np.ascontiguousarray(np.asarray(par, dtype=dtype).reshape(-1))

    def equations(
        self,
        fwd: Any,
        cur: Any,
        par: Mapping[str, float] | Any,
    ) -> ND:
        fwd_arr = np.ascontiguousarray(np.asarray(fwd, dtype=complex128).reshape(-1))
        cur_arr = np.ascontiguousarray(np.asarray(cur, dtype=complex128).reshape(-1))
        par_vec = self._coerce_param_vector(par, dtype=complex128)
        if par_vec.shape[0] != len(self.calib_params):
            raise ValueError(
                f"Parameter vector length {par_vec.shape[0]} != {len(self.calib_params)}"
            )

        return self.construct_objective_vector_func()(fwd_arr, cur_arr, par_vec)

    def build_affine_measurement_matrices(
        self,
        params: Mapping[Any, Any] | Any,
        observables: list[str],
    ) -> tuple[NDF, NDF]:
        param_vec = self._coerce_param_vector(params, dtype=float64)
        if param_vec.shape[0] != len(self.calib_params):
            raise ValueError(
                f"Parameter vector length {param_vec.shape[0]} != {len(self.calib_params)}"
            )

        zero_state = np.zeros((len(self.cur_syms),), dtype=float64)
        d_full = np.asarray(
            self.construct_measurement_vector_func()(*zero_state, *param_vec),
            dtype=float64,
        ).reshape(-1)
        C_full = np.asarray(
            self.observable_jacobian(*zero_state, *param_vec),
            dtype=float64,
        )

        obs_idx = {name: i for i, name in enumerate(self.observable_names)}
        rows = [obs_idx[name] for name in observables]
        return (
            np.ascontiguousarray(C_full[rows, :], dtype=float64),
            np.ascontiguousarray(d_full[rows], dtype=float64),
        )

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
