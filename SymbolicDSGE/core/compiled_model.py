from sympy import Symbol, Expr

import numpy as np
from numpy import complex128, float64
from numpy.typing import NDArray

from dataclasses import dataclass, asdict
from functools import cached_property
from typing import Callable, Any, Mapping

from .config import ModelConfig
from ..kalman.config import KalmanConfig
from SymbolicDSGE._symbolic_printers import (
    BicomplexOps,
    MeasurementLayout,
    ResidualLayout,
    build_cfunc,
    build_measurement_cfunc,
)
from .._ckernels.core import jacobian_eval, measurement_eval, residual_eval

NDF = NDArray[float64]
NDC = NDArray[complex128]
ND = NDArray


@dataclass(frozen=True)
class VariableLayout:
    declared_names: tuple[str, ...]
    canonical_names: tuple[str, ...]
    exo_state_names: tuple[str, ...]
    endo_state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    n_exog: int
    n_state: int
    idx: dict[str, int]


@dataclass(frozen=True)
class CompiledModel:
    config: ModelConfig
    kalman: KalmanConfig | None

    cur_syms: list[Symbol]

    layout: VariableLayout
    var_names: list[str]
    idx: dict[str, int]

    objective_eqs: list[Expr]

    calib_params: list[Symbol]

    observable_names: list[str]
    observable_eqs: list[Expr]
    # Flat row-major (n_obs, n_var) symbolic jacobian d(observable)/d(cur_var);
    # printed to a native cfunc on demand (construct_observable_jacobian_cfunc).
    observable_jacobian_eqs: list[Expr]

    n_state: int
    n_exog: int

    @cached_property
    def _objective_cfunc(self) -> Any:
        # Residual as a numba @cfunc (C ABI) for the native complex-step preproc
        # (klein_preprocess). Held here so its .address stays valid for the driver.
        return build_cfunc(self.objective_eqs, ResidualLayout.from_compiled(self))

    def construct_objective_cfunc(self) -> Any:
        return self._objective_cfunc

    @cached_property
    def _objective_cfunc_bicomplex(self) -> Any:
        # Residual as a bicomplex @cfunc for the second-order Hessian sweep
        # (bicomplex_hessian). Held here so its .address stays valid.
        return build_cfunc(
            self.objective_eqs, ResidualLayout.from_compiled(self), BicomplexOps()
        )

    def construct_objective_cfunc_bicomplex(self) -> Any:
        return self._objective_cfunc_bicomplex

    def _coerce_param_vector(self, par: Mapping[Any, Any] | Any) -> ND:
        # Resolve a name/Symbol-keyed mapping into calib_params order. dtype and
        # contiguity are the native boundary's job (the Cython shims cast), so this
        # only produces the ordered numeric vector.
        if isinstance(par, Mapping):
            vals = []
            for p in self.calib_params:
                if p in par:
                    vals.append(par[p])
                elif p.name in par:
                    vals.append(par[p.name])
                else:
                    raise KeyError(f"Missing parameter '{p.name}'.")
            return np.asarray(vals)

        return np.asarray(par)

    def equations(
        self,
        fwd: Any,
        cur: Any,
        par: Mapping[str, float] | Any,
    ) -> ND:
        par_vec = self._coerce_param_vector(par)
        if par_vec.shape[0] != len(self.calib_params):
            raise ValueError(
                f"Parameter vector length {par_vec.shape[0]} != {len(self.calib_params)}"
            )

        return residual_eval(
            self.construct_objective_cfunc().address,
            fwd,
            cur,
            par_vec,
            len(self.objective_eqs),
        )

    def build_affine_measurement_matrices(
        self,
        params: Mapping[Any, Any] | Any,
        observables: list[str],
    ) -> tuple[NDF, NDF]:
        param_vec = self._coerce_param_vector(params)
        if param_vec.shape[0] != len(self.calib_params):
            raise ValueError(
                f"Parameter vector length {param_vec.shape[0]} != {len(self.calib_params)}"
            )

        zero_state = np.zeros((len(self.cur_syms),), dtype=float64)
        meas_addr = self.construct_measurement_cfunc(observables).address
        jac_addr = self.construct_observable_jacobian_cfunc(observables).address
        n_obs = len(observables)

        d = measurement_eval(meas_addr, zero_state, param_vec, n_obs)
        C = jacobian_eval(jac_addr, zero_state, param_vec, n_obs, len(self.cur_syms))
        return C, d

    def _normalize_observables(
        self,
        observables: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        if observables is None:
            return tuple(self.observable_names)

        obs = tuple(observables)
        if len(set(obs)) != len(obs):
            raise ValueError("Observable list contains duplicates.")

        obs_idx = {name: i for i, name in enumerate(self.observable_names)}
        missing = [name for name in obs if name not in obs_idx]
        if missing:
            raise KeyError(f"Unknown observables not in compiled model: {missing}")

        return tuple(sorted(obs, key=lambda name: obs_idx[name]))

    @cached_property
    def _measurement_cfunc_cache(self) -> dict[tuple[str, ...], Any]:
        return {}

    def construct_measurement_cfunc(
        self,
        observables: list[str] | tuple[str, ...] | None = None,
    ) -> Any:
        obs = self._normalize_observables(observables)
        cache = self._measurement_cfunc_cache
        if obs in cache:
            return cache[obs]

        layout = MeasurementLayout.from_compiled(self, obs)
        exprs = [self.observable_eqs[i] for i in layout.observable_indices]
        cache[obs] = build_measurement_cfunc(exprs, layout)
        return cache[obs]

    @cached_property
    def _observable_jacobian_cfunc_cache(self) -> dict[tuple[str, ...], Any]:
        return {}

    def construct_observable_jacobian_cfunc(
        self,
        observables: list[str] | tuple[str, ...] | None = None,
    ) -> Any:
        obs = self._normalize_observables(observables)
        cache = self._observable_jacobian_cfunc_cache
        if obs in cache:
            return cache[obs]

        base = MeasurementLayout.from_compiled(self, obs)
        n_var = base.n_var
        obs_idx = {name: i for i, name in enumerate(self.observable_names)}
        # Flat row-major (obs, var) jacobian exprs for the selected observables.
        exprs = [
            self.observable_jacobian_eqs[obs_idx[name] * n_var + j]
            for name in obs
            for j in range(n_var)
        ]
        layout = MeasurementLayout(
            slot=base.slot,
            n_var=n_var,
            n_par=base.n_par,
            n_obs=len(exprs),
        )
        cache[obs] = build_measurement_cfunc(exprs, layout)
        return cache[obs]

    @cached_property
    def _measurement_array_func_cache(self) -> dict[tuple[str, ...], Callable[..., ND]]:
        return {}

    def construct_measurement_array_func(
        self,
        observables: list[str] | tuple[str, ...] | None = None,
    ) -> Callable[..., ND]:
        obs = self._normalize_observables(observables)
        cache = self._measurement_array_func_cache
        if obs in cache:
            return cache[obs]

        addr = self.construct_measurement_cfunc(obs).address
        n_obs = len(obs)

        def measurement_array(state: ND, params: ND) -> ND:
            return measurement_eval(addr, state, params, n_obs)

        cache[obs] = measurement_array
        return measurement_array

    @cached_property
    def _observable_jacobian_array_func_cache(
        self,
    ) -> dict[tuple[str, ...], Callable[..., ND]]:
        return {}

    def construct_observable_jacobian_array_func(
        self,
        observables: list[str] | tuple[str, ...] | None = None,
    ) -> Callable[..., ND]:
        obs = self._normalize_observables(observables)
        cache = self._observable_jacobian_array_func_cache
        if obs in cache:
            return cache[obs]

        addr = self.construct_observable_jacobian_cfunc(obs).address
        n_obs = len(obs)
        n_var = len(self.cur_syms)

        def jacobian_array(state: ND, params: ND) -> ND:
            return jacobian_eval(addr, state, params, n_obs, n_var)

        cache[obs] = jacobian_array
        return jacobian_array

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
