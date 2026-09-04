from sympy import Symbol, Expr

import numpy as np
from numpy import complex128, float64, int64
from numpy.typing import NDArray

from dataclasses import dataclass, asdict, field
from functools import cached_property
from typing import Callable, Any, Mapping, Sequence

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
from .._ckernels.core import (
    INC_CUR,
    INC_LAG,
    INC_LEAD,
    jacobian_eval,
    measurement_eval,
)

NDF = NDArray[float64]
NDC = NDArray[complex128]
ND = NDArray


@dataclass(frozen=True)
class VariableLayout:
    """Where every compiled variable sits, and which ones the compiler minted.

    The counts are the model's dimensions, and a consumer sizing an array takes
    them from here rather than measuring whichever list is nearest. ``n_var`` is
    the compiled set, ``n_declared`` the model's own variables and
    ``n_generated`` the minted ones, so ``n_var == n_declared + n_generated``.
    The first two coincide until the compiler mints something, which is why they
    are separate numbers and not one: a dense ``ss_seed`` or a parse-time ``P0``
    spans ``n_declared``, a dense ``x0`` spans ``n_var``. ``n_state`` and
    ``n_ctrl`` split the canonical order and sum to ``n_var``.

    ``declared_names`` is the model's own declaration order, and nothing else.
    ``generated_names`` is the compiler's minted names in mint order, which is
    where they sit in declaration order. Neither carries a position: a name's
    canonical slot is ``idx``'s answer, and asking one map for two orderings is
    what lets them drift.

    ``aux_origin`` maps each minted lag or lead to the declared variable it
    tracks, at every depth, which is what widens a declared-order input over the
    generated block. An aux sits at its origin some periods away, so it shares
    the origin's steady state and takes its variance.

    ``state_names`` and ``control_names`` split the canonical order at
    ``n_state``. A variable is a state when it occurs at ``t-1``, which is what
    the pencil partition needs.

    ``n_exog`` counts the model's innovations, which is the width of the shock
    matrix and of ``B``, not a count of variables.

    ``shock_names`` is ordered names of the shock columns.

    ``shock_idx`` is that order as a lookup."""

    n_var: int
    n_declared: int
    n_generated: int

    n_exog: int
    n_state: int
    n_ctrl: int

    idx: dict[str, int]

    declared_names: tuple[str, ...]
    canonical_names: tuple[str, ...]
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    generated_names: tuple[str, ...] = ()

    aux_origin: dict[str, str] = field(default_factory=dict)
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
    """One regime's replaced rows, and their pencil blocks at the reference point.

    ``jac_a``/``jac_b``/``jac_c`` are the replaced rows of the three date
    Jacobians, flat row-major ``(len(rows), n_var)``, and ``jac_d`` is the shock
    block, ``(len(rows), n_exog)``. They carry ``klein_preproc``'s signs, so the
    row reads ``a y' = b y + c y_prev + d eps - const``.
    """

    rows: list[int]
    residuals: list[Expr] = field(default_factory=list)
    jac_a: list[Expr] = field(default_factory=list)
    jac_b: list[Expr] = field(default_factory=list)
    jac_c: list[Expr] = field(default_factory=list)
    jac_d: list[Expr] = field(default_factory=list)
    constants: list[Expr] = field(default_factory=list)


@dataclass(frozen=True)
class RegimePencilFunc:
    """Compiled regime pencil rows, and everything the native side needs to call them.

    One cfunc per regime, keyed by the same bitmask as ``regimes``, writing
    ``[jac_a; jac_b; jac_c; jac_d; constants]`` into a single buffer: each block
    whole and row-major, ordered like ``rows``, then the constants. Concatenated
    rather than interleaved because the blocks patch into separate copies of the
    reference pencil, so each row is a contiguous copy on every side.

    ``jac_b``/``jac_c``/``jac_d`` carry klein_preproc's signs and ``constants``
    is unnegated, so all of them drop into a reference pencil copy as they are.
    """

    cfuncs: dict[int, Any]
    rows: dict[int, NDArray[int64]]
    n_var: int
    n_exog: int
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
        """Length of ``out``: the three date blocks, the shock block, then the
        constants, ``n_row * (3 * n_var + n_exog + 1)``."""
        n_row = self.n_row(mask)
        return n_row * (3 * self.n_var + self.n_exog + 1)


@dataclass(frozen=True, repr=False)
class CompiledModel:
    config: ModelConfig
    kalman: KalmanConfig | None

    cur_syms: list[Symbol]

    layout: VariableLayout
    var_names: list[str]
    idx: dict[str, int]

    objective_eqs: list[Expr]

    calib_params: list[str]

    observable_names: list[str]
    observable_eqs: list[Expr]
    # Flat row-major (n_obs, n_var) symbolic jacobian d(observable)/d(cur_var);
    # printed to a native cfunc on demand (construct_observable_jacobian_cfunc).
    observable_jacobian_eqs: list[Expr]

    # Regime conditions in declaration order, bind then relax per constraint;
    # printed to a native cfunc on demand (construct_constraint_func).
    constraint_names: tuple[str, ...] = ()
    constraint_exprs: list[Boolean] = field(default_factory=list)

    # One block per regime, keyed by the bitmask of its binding constraints over
    # constraint_names. Residuals stay in reference equation order and print to
    # native cfuncs on demand (construct_regime_cfuncs).
    regimes: dict[int, RegimeBlock] = field(default_factory=dict)

    @property
    def n_var(self) -> int:
        return self.layout.n_var

    @property
    def n_state(self) -> int:
        return self.layout.n_state

    @property
    def n_ctrl(self) -> int:
        return self.layout.n_ctrl

    @property
    def n_exog(self) -> int:
        return self.layout.n_exog

    @property
    def n_declared(self) -> int:
        return self.layout.n_declared

    @property
    def n_generated(self) -> int:
        return self.layout.n_generated

    @property
    def n_par(self) -> int:
        return len(self.calib_params)

    @property
    def n_obs(self) -> int:
        return len(self.observable_names)

    @property
    def shock_names(self) -> tuple[str, ...]:
        """Shock column names, in ``shocks`` (and so column) order."""
        return self.layout.shock_names

    @property
    def shock_idx(self) -> dict[str, int]:
        """``{shock name: column}`` into the ``(T, n_exog)`` shock matrix."""
        return self.layout.shock_idx

    @cached_property
    def _incidence(self) -> NDArray[np.int8]:
        """``(n_var,)`` of ``SDSGE_INC_*`` bits: the dates each variable occurs at.

        The solve partitions the pencil on this, so it is read from the symbols
        the equations actually contain rather than from a Jacobian's sparsity. A
        calibration that happened to zero a coefficient would otherwise move a
        variable between groups and resize the state vector between draws.

        Unioned over the reference equations and every regime's replacements:
        the regime pencils stack by bitmask into one array, so a regime dropping
        the last occurrence of some ``v(t+1)`` must not give that regime a pencil
        of its own shape.
        """
        bit_of = {"prev": INC_LAG, "cur": INC_CUR, "fwd": INC_LEAD}

        exprs = list(self.objective_eqs)
        for block in self.regimes.values():
            exprs.extend(block.residuals)
        # Observables too: the layout's own lag scan covers them, and the solve
        # indexes `p` by `n_state` while iterating `nspred`, so the two sets
        # disagreeing is an out-of-bounds write rather than a wrong answer.
        exprs.extend(self.observable_eqs)

        present: set[str] = set()
        for expr in exprs:
            present.update(s.name for s in expr.free_symbols)  # pyright: ignore

        out = np.zeros(self.n_var, dtype=np.int8)
        for i, name in enumerate(self.var_names):
            bits = 0
            for prefix, bit in bit_of.items():
                if f"{prefix}_{name}" in present:
                    bits |= bit
            out[i] = bits
        return out

    @cached_property
    def _regime_cfuncs(self) -> dict[int, Any]:
        # One residual @cfunc per regime, sharing the reference layout: regimes
        # replace equations by name, so n_var/n_par are unchanged. Held here
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
            want_shock = len(block.rows) * self.n_exog
            want_const = len(block.rows)
            got = (len(block.jac_a), len(block.jac_b), len(block.jac_c))
            if any(n != want_jac for n in got):
                raise ValueError(
                    f"Regime {mask} has {got} a/b/c jacobian entries, expected "
                    f"{want_jac} each for {len(block.rows)} rows over "
                    f"{base.n_var} variables."
                )
            if len(block.jac_d) != want_shock:
                raise ValueError(
                    f"Regime {mask} has {len(block.jac_d)} shock jacobian "
                    f"entries, expected {want_shock} for {len(block.rows)} rows "
                    f"over {self.n_exog} shocks."
                )
            if len(block.constants) != want_const:
                raise ValueError(
                    f"Regime {mask} has {len(block.constants)} constants, "
                    f"expected {want_const} for {len(block.rows)} rows."
                )
            exprs = [
                *block.jac_a,
                *block.jac_b,
                *block.jac_c,
                *block.jac_d,
                *block.constants,
            ]
            layout = MeasurementLayout(
                slot=base.slot,
                n_var=base.n_var,
                n_par=base.n_par,
                n_obs=len(exprs),
            )
            cfuncs[mask] = build_measurement_cfunc(exprs, layout)
            rows[mask] = np.asarray(block.rows, dtype=np.int64)

        return RegimePencilFunc(
            cfuncs=cfuncs,
            rows=rows,
            n_var=base.n_var,
            n_exog=self.n_exog,
            n_par=base.n_par,
        )

    def construct_regime_pencil_func(self) -> RegimePencilFunc | None:
        return self._regime_pencil_func

    @cached_property
    def _constraint_func(self) -> ConstraintFunc:
        # Conditions as one numba @cfunc (C ABI) for the native OccBin driver.
        # Held here so its .address stays valid for the driver.
        if not self.constraint_names:
            raise ValueError(
                "Constraint function cannot be built: no constraints in compiled model."
            )
        layout = ConstraintLayout.from_compiled(self, self.constraint_names)
        cfunc, inclusive = build_constraint_cfunc(self.constraint_exprs, layout)
        return ConstraintFunc(
            cfunc=cfunc,
            names=self.constraint_names,
            n_var=layout.n_var,
            n_par=layout.n_par,
            inclusive=inclusive,
        )

    def construct_constraint_func(self) -> ConstraintFunc:
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
                else:
                    raise KeyError(f"Missing parameter '{p}'.")
            return np.asarray(vals)

        return np.asarray(par)

    def build_affine_measurement_matrices(
        self,
        params: Mapping[Any, Any] | Any,
        observables: Sequence[str],
        ss: NDF,
    ) -> tuple[NDF, NDF]:
        param_vec = self._coerce_param_vector(params)
        if param_vec.shape[0] != self.n_par:
            raise ValueError(
                f"Parameter vector length {param_vec.shape[0]} != {self.n_par}"
            )

        meas_addr = self.construct_measurement_cfunc(observables).address
        jac_addr = self.construct_observable_jacobian_cfunc(observables).address
        n_obs = len(observables)

        d = measurement_eval(meas_addr, ss, param_vec, n_obs)
        C = jacobian_eval(jac_addr, ss, param_vec, n_obs, self.n_var)
        return C, d

    def _normalize_observables(
        self,
        observables: Sequence[str] | None,
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
        observables: Sequence[str] | None = None,
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
        observables: Sequence[str] | None = None,
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
        n_var = self.n_var

        def jacobian_array(state: ND, params: ND) -> ND:
            return jacobian_eval(addr, state, params, n_obs, n_var)

        cache[obs] = jacobian_array
        return jacobian_array

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config.name})"
