from sympy import Symbol, Expr

import numpy as np
from numpy import complex128, float64, int64
from numpy.typing import NDArray

from dataclasses import dataclass, asdict, field
from functools import cached_property
from typing import Callable, Any, Mapping

from sympy.logic.boolalg import Boolean

from .config import ModelConfig
from ..kalman.config import KalmanConfig
from SymbolicDSGE._symbolic_printers import (
    BicomplexOps,
    ConstraintLayout,
    MeasurementLayout,
    ResidualLayout,
    build_cfunc,
    build_constraint_cfunc,
    build_measurement_cfunc,
)
from .._ckernels.core import jacobian_eval, measurement_eval, residual_eval

NDF = NDArray[float64]
NDC = NDArray[complex128]
ND = NDArray


@dataclass(frozen=True)
class VariableLayout:
    """Where every compiled variable sits, and which ones the compiler minted.

    ``declared_names`` is the model's own declaration order followed by the
    generated variables, which is the order the parse-time ``P0`` is widened
    into. ``generated`` maps each generated name to its canonical position, so a
    consumer hiding them filters by name or by position off the same map.

    ``lag_origin`` maps each lag aux to the declared variable it tracks, at every
    depth, which is what widens a declared-order input over the generated block.
    A shock state is absent from it: it tracks an innovation, not a variable.

    ``shock_names`` names the shock columns in ``shock_map`` order, which is the
    order the compiler minted the shock states in, and ``shock_idx`` is that
    order as a lookup. A shock is named by its own symbol rather than by the
    variable it targets: after desugaring one shock may enter several equations,
    and a target name would claim a scope the innovation does not have.
    """

    declared_names: tuple[str, ...]
    canonical_names: tuple[str, ...]
    exo_state_names: tuple[str, ...]
    endo_state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    n_exog: int
    n_state: int
    idx: dict[str, int]
    generated: dict[str, int] = field(default_factory=dict)
    lag_origin: dict[str, str] = field(default_factory=dict)
    shock_names: tuple[str, ...] = ()
    shock_idx: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstraintFunc:
    """Compiled regime conditions, and everything the native side needs to call them.

    One cfunc evaluates every condition, writing ``2 * n_constraint`` signed
    distances to their boundaries, positive where the condition holds: slot
    ``2i`` is constraint ``i`` binding, slot ``2i + 1`` is it relaxing. The C
    caller reads the sign of the one slot the incoming regime asks about, with
    ``next = prev ? !relax : bind``, and takes the same number as the error that
    ranks guess-and-verify iterations.

    ``inclusive`` is the bitmask of slots that hold at a distance of exactly
    zero, the only thing a sign cannot say and the reason ``x < 0`` and ``x <=
    0`` are distinguishable at the steady state, where they are both zero.

    ``names`` is declaration order, which is the regime bit order.
    """

    cfunc: Any
    names: tuple[str, ...]
    n_var: int
    n_par: int
    inclusive: int

    @property
    def address(self) -> int:
        """Entry point of
        ``void (*)(const double *cur, const double *par, double *err)``."""
        return int(self.cfunc.address)

    @property
    def n_constraint(self) -> int:
        return len(self.names)

    @property
    def n_cond(self) -> int:
        return 2 * len(self.names)

    def bind_slot(self, name: str) -> int:
        return 2 * self.names.index(name)

    def relax_slot(self, name: str) -> int:
        return 2 * self.names.index(name) + 1

    def bit(self, name: str) -> int:
        """Bitmask position of ``name`` in a regime key."""
        return self.names.index(name)

    def mask(self, binding: frozenset[str] | set[str]) -> int:
        """Regime key as a bitmask over ``names``."""
        return sum(1 << self.bit(n) for n in binding)


@dataclass(frozen=True)
class RegimeBlock:
    rows: list[int]
    residuals: list[Expr] = field(default_factory=list)
    jac_a: list[Expr] = field(default_factory=list)
    jac_b: list[Expr] = field(default_factory=list)
    constants: list[Expr] = field(default_factory=list)


@dataclass(frozen=True)
class RegimePencilFunc:
    """Compiled regime pencil rows, and everything the native side needs to call them.

    One cfunc per regime, keyed by the same bitmask as ``regimes``, writing
    ``[jac_a; jac_b; constants]`` into a single ``2 * n_row * n_var + n_row``
    buffer: the whole a block, then the whole b block, each row-major
    ``(n_row, n_var)`` and ordered like ``rows``, then the constants. Concatenated
    rather than interleaved because the blocks patch into separate copies of the
    reference pencil, so each row is a contiguous copy on every side.

    ``jac_b`` carries klein_preproc's sign and ``constants`` is unnegated, so all
    three drop into a reference pencil copy as they are.
    """

    cfuncs: dict[int, Any]
    rows: dict[int, NDArray[int64]]
    n_var: int
    n_par: int

    @property
    def masks(self) -> tuple[int, ...]:
        return tuple(sorted(self.cfuncs))

    def address(self, mask: int) -> int:
        """Entry point of
        ``void (*)(const double *cur, const double *par, double *out)``."""
        return int(self.cfuncs[mask].address)

    def n_row(self, mask: int) -> int:
        """Reference rows this regime replaces."""
        return int(self.rows[mask].shape[0])

    def n_out(self, mask: int) -> int:
        """Length of ``out``: both blocks, ``2 * n_row * n_var + n_row``."""
        n_row = self.n_row(mask)
        return 2 * n_row * self.n_var + n_row


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

    # Regime conditions in declaration order, bind then relax per constraint;
    # printed to a native cfunc on demand (construct_constraint_func).
    constraint_names: tuple[str, ...] = ()
    constraint_exprs: list[Boolean] = field(default_factory=list)

    # One block per regime, keyed by the bitmask of its binding constraints over
    # constraint_names. Residuals stay in reference equation order and print to
    # native cfuncs on demand (construct_regime_cfuncs).
    regimes: dict[int, RegimeBlock] = field(default_factory=dict)

    @property
    def shock_names(self) -> tuple[str, ...]:
        """Shock column names, in ``shock_map`` (and so column) order."""
        return self.layout.shock_names

    @property
    def shock_idx(self) -> dict[str, int]:
        """``{shock name: column}`` into the ``(T, n_exog)`` shock matrix."""
        return self.layout.shock_idx

    @cached_property
    def _regime_cfuncs(self) -> dict[int, Any]:
        # One residual @cfunc per regime, sharing the reference layout: regimes
        # replace equations by name, so n_eq/n_var/n_par are unchanged. Held here
        # so the addresses stay valid for the driver.
        layout = ResidualLayout.from_compiled(self)
        return {
            mask: build_cfunc(block.residuals, layout)
            for mask, block in self.regimes.items()
        }

    def construct_regime_cfuncs(self) -> dict[int, Any]:
        return self._regime_cfuncs

    @cached_property
    def _regime_pencil_func(self) -> RegimePencilFunc | None:
        # Replaced rows of each regime pencil as one cfunc per regime, so the
        # assembly patches a reference pencil copy instead of sweeping the whole
        # regime. Both blocks share a cfunc: after the fwd fold they are
        # functions of the same `cur` vector and share subexpressions. Held here
        # so the addresses and the row buffers stay valid for the driver.
        if not self.regimes:
            return None

        base = MeasurementLayout.from_compiled(self)
        cfuncs: dict[int, Any] = {}
        rows: dict[int, NDArray[int64]] = {}
        for mask, block in self.regimes.items():
            want_jac = len(block.rows) * base.n_var
            want_const = len(block.rows)
            if len(block.jac_a) != want_jac or len(block.jac_b) != want_jac:
                raise ValueError(
                    f"Regime {mask} has {len(block.jac_a)}/{len(block.jac_b)} "
                    f"jacobian entries, expected {want_jac} for {len(block.rows)} "
                    f"rows over {base.n_var} variables."
                )
            if len(block.constants) != want_const:
                raise ValueError(
                    f"Regime {mask} has {len(block.constants)} constants, "
                    f"expected {want_const} for {len(block.rows)} rows."
                )
            exprs = [*block.jac_a, *block.jac_b, *block.constants]
            layout = MeasurementLayout(
                slot=base.slot,
                n_var=base.n_var,
                n_par=base.n_par,
                n_obs=len(exprs),
            )
            cfuncs[mask] = build_measurement_cfunc(exprs, layout)
            rows[mask] = np.asarray(block.rows, dtype=np.int64)

        return RegimePencilFunc(
            cfuncs=cfuncs, rows=rows, n_var=base.n_var, n_par=base.n_par
        )

    def construct_regime_pencil_func(self) -> RegimePencilFunc | None:
        return self._regime_pencil_func

    @cached_property
    def _constraint_func(self) -> ConstraintFunc | None:
        # Conditions as one numba @cfunc (C ABI) for the native OccBin driver.
        # Held here so its .address stays valid for the driver.
        if not self.constraint_names:
            return None
        layout = ConstraintLayout.from_compiled(self, self.constraint_names)
        cfunc, inclusive = build_constraint_cfunc(self.constraint_exprs, layout)
        return ConstraintFunc(
            cfunc=cfunc,
            names=self.constraint_names,
            n_var=layout.n_var,
            n_par=layout.n_par,
            inclusive=inclusive,
        )

    def construct_constraint_func(self) -> ConstraintFunc | None:
        return self._constraint_func

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
        ss: NDF,
    ) -> tuple[NDF, NDF]:
        param_vec = self._coerce_param_vector(params)
        if param_vec.shape[0] != len(self.calib_params):
            raise ValueError(
                f"Parameter vector length {param_vec.shape[0]} != {len(self.calib_params)}"
            )

        meas_addr = self.construct_measurement_cfunc(observables).address
        jac_addr = self.construct_observable_jacobian_cfunc(observables).address
        n_obs = len(observables)

        d = measurement_eval(meas_addr, ss, param_vec, n_obs)
        C = jacobian_eval(jac_addr, ss, param_vec, n_obs, len(self.cur_syms))
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
