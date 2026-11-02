"""Compiled, numericly-evaluable representation of a DSGE model."""

from sympy import Expr

import numpy as np
from numpy import complex128, float64, int64
from numpy.typing import NDArray

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Sequence, cast
from types import MappingProxyType

from sympy.logic.boolalg import Boolean


from .resolvers import VariableLayout
from SymbolicDSGE._symbolic_printers import (
    OpTable,
    C128Ops,
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

# Global cfunc cache for all emitters.
# Keys on (int(config_token), str(<emitter>), Optional[object(<selector>)])
# A hit on config token indicates two equal models, module level
# Caching can intentionally match on different compilation instances
# sharing the config token.
#
# --- Lifetimes ---
# The cache is global and nothing is evicted.
# Functions stay alive in the cache for the duration of the process.
# As a result, their `.address` is constant and valid at all times.
_CFUNC_CACHE: dict[tuple, Any] = {}


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
        ``void (*)(const double *cur, const double *par, double *err)``.
        """
        return int(self.cfunc.address)

    @property
    def n_constraint(self) -> int:
        """Number of regime conditions."""
        return len(self.names)

    @property
    def n_cond(self) -> int:
        """Number of constraints. (bind + relax) per condition."""
        return 2 * len(self.names)

    def bind_slot(self, name: str) -> int:
        """Index getter for the binding slot of a named regime condition.

        Parameters
        ----------
        name : str
            Name of the regime condition, which must be in ``names``.

        Returns
        -------
        int
            Index of the binding slot for the named regime condition.

        """
        return 2 * self.names.index(name)

    def relax_slot(self, name: str) -> int:
        """Index getter for the relaxing slot of a named regime condition.

        Parameters
        ----------
        name : str
            Name of the regime condition, which must be in ``names``.

        Returns
        -------
        int
            Index of the relaxing slot for the named regime condition.

        """
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

    rows: tuple[int, ...]
    residuals: tuple[Expr, ...] = ()
    jac_a: tuple[Expr, ...] = ()
    jac_b: tuple[Expr, ...] = ()
    jac_c: tuple[Expr, ...] = ()
    jac_d: tuple[Expr, ...] = ()
    constants: tuple[Expr, ...] = ()


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

    cfuncs: MappingProxyType[int, Any]
    rows: MappingProxyType[int, NDArray[int64]]
    n_var: int
    n_exog: int
    n_par: int

    @property
    def masks(self) -> tuple[int, ...]:
        """Bitmask keys of the regimes that have a regime pencil."""
        return tuple(sorted(self.cfuncs))

    def address(self, mask: int) -> int:
        """Entry point of
        ``void (*)(const double *cur, const double *par, double *out)``.
        """
        return int(self.cfuncs[mask].address)

    def n_row(self, mask: int) -> int:
        """Reference rows this regime replaces."""
        return int(self.rows[mask].shape[0])

    def n_out(self, mask: int) -> int:
        """Length of ``out``: the three date blocks, the shock block, then the
        constants, ``n_row * (3 * n_var + n_exog + 1)``.
        """
        n_row = self.n_row(mask)
        return n_row * (3 * self.n_var + self.n_exog + 1)


@dataclass(frozen=True, repr=False)
class CompiledModel:
    """Compiled, numericly-evaluable representation of a DSGE model.

    Attributes
    ----------
    name : str
        Name of the model specification that produced this compiled model.
    layout : VariableLayout
        :class:`VariableLayout` object describing the model's variable structure and
        indexing.
    var_names : tuple[str, ...]
        Names of all compiled variables, in canonical order.
    idx : MappingProxyType[str, int]
        Index mapping variable names to their canonical positions.
    objective_eqs : tuple[Expr, ...]
        Expressions of the model's residual equations, in canonical order.
    calib_params : tuple[str, ...]
        Calibration parameter names, in canonical order.
    observable_names : tuple[str, ...]
        Names of the model's observables, in canonical order.
    observable_eqs : tuple[Expr, ...]
        Observable equations, in canonical order.
    measurement_jacobian_eqs : tuple[Expr, ...]
        Jacobian expressions of the observables with respect to the model's
        variables, in row-major order.
    constraint_names : tuple[str, ...]
        Constraint names, in declaration order, for regime conditions.
    constraint_exprs : tuple[Boolean, ...]
        Expressions of the regime constraints, in declaration order.
    regimes : MappingProxyType[int, RegimeBlock]
        Regime blocks, keyed by the bitmask of their binding constraints, containing
        replaced rows and their Jacobians.
    n_var : int
        Number of compiled variables, including generated ones.
    n_state : int
        Number of state variables.
    n_ctrl : int
        Number of control variables.
    n_exog : int
        Number of exogenous shocks in the model.
    n_declared : int
        Number of declared model variables, excluding any generated ones.
    n_generated : int
        Number of generated variables, minted by the compiler to fill in missing lags or leads.
    n_par : int
        Number of calibration parameters in the model.
    n_obs : int
        Number of observables in the model.
    shock_names : tuple[str, ...]
        Names of the shock columns, in order.
    shock_idx : dict[str, int]
        Index mapping shock names to their column positions in the shock matrix.
    """

    name: str

    layout: VariableLayout
    var_names: tuple[str, ...]
    idx: MappingProxyType[str, int]

    objective_eqs: tuple[Expr, ...]

    calib_params: tuple[str, ...]

    observable_names: tuple[str, ...]
    observable_eqs: tuple[Expr, ...]
    # Flat row-major (n_obs, n_var) symbolic jacobian d(observable)/d(cur_var);
    # printed to a native cfunc on demand (construct_measurement_jacobian_cfunc).
    measurement_jacobian_eqs: tuple[Expr, ...]

    # Identifier for the config producing this compiled model,
    # used to validate compatibility of compiled models and configs
    # passed to methods together.
    config_token: int

    # Regime conditions in declaration order, bind then relax per constraint;
    # printed to a native cfunc on demand (construct_constraint_func).
    constraint_names: tuple[str, ...] = ()
    constraint_exprs: tuple[Boolean, ...] = ()

    # One block per regime, keyed by the bitmask of its binding constraints over
    # constraint_names. Residuals stay in reference equation order.
    regimes: MappingProxyType[int, RegimeBlock] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def n_var(self) -> int:
        """Number of compiled variables, including generated ones."""
        return self.layout.n_var

    @property
    def n_state(self) -> int:
        """Number of state variables, which are the ones that occur at ``t-1`` in the model's equations."""
        return self.layout.n_state

    @property
    def n_ctrl(self) -> int:
        """Number of control variables, which are the ones that occur at ``t`` or ``t+1`` in the model's equations."""
        return self.layout.n_ctrl

    @property
    def n_exog(self) -> int:
        """Number of exogenous shocks in the model, which is the width of the shock matrix and of ``B``."""
        return self.layout.n_exog

    @property
    def n_declared(self) -> int:
        """Number of declared model variables, excluding any generated ones."""
        return self.layout.n_declared

    @property
    def n_generated(self) -> int:
        """Number of generated variables, which are the ones minted by the compiler to fill in missing lags or leads."""
        return self.layout.n_generated

    @property
    def n_par(self) -> int:
        """Number of calibration parameters in the model, which is the length of the parameter vector."""
        return len(self.calib_params)

    @property
    def n_obs(self) -> int:
        """Number of observables in the model, which is the length of the observable vector."""
        return len(self.observable_names)

    @property
    def shock_names(self) -> tuple[str, ...]:
        """Shock column names, in ``shocks`` (and so column) order."""
        return self.layout.shock_names

    @property
    def shock_idx(self) -> MappingProxyType[str, int]:
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

    def construct_regime_pencil_func(self) -> RegimePencilFunc | None:
        """Build the native (C callable) functions that evaluate the model's regime pencil rows at a given state and parameter vector.

        Returns
        -------
        RegimePencilFunc | None
            :class:`RegimePencilFunc` containing the regime pencil functions, or ``None`` if the model has no regimes.

        """
        # Replaced rows of each regime pencil as one cfunc per regime, so the
        # assembly patches a reference pencil copy instead of sweeping the whole
        # regime. Both blocks share a cfunc: after the fwd fold they are
        # functions of the same `cur` vector and share subexpressions. Held here
        # so the addresses and the row buffers stay valid for the driver.
        if not self.regimes:
            return None

        key = (self.config_token, "regime", None)
        if hit := _CFUNC_CACHE.get(key):
            return cast(RegimePencilFunc, hit)

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

        reg = RegimePencilFunc(
            cfuncs=MappingProxyType(cfuncs),
            rows=MappingProxyType(rows),
            n_var=base.n_var,
            n_exog=self.n_exog,
            n_par=base.n_par,
        )
        _CFUNC_CACHE[key] = reg
        return reg

    def construct_constraint_func(self) -> ConstraintFunc:
        """Build the native (C callable) function that evaluates the model's regime conditions at a given state and parameter vector.

        Returns
        -------
        ConstraintFunc
            :class:`ConstraintFunc` containing the regime condition function.

        """
        # Conditions as one numba @cfunc (C ABI) for the native OccBin driver.
        # Held here so its .address stays valid for the driver.
        if not self.constraint_names:
            raise ValueError(
                "Constraint function cannot be built: no constraints in compiled model."
            )
        key = (self.config_token, "constraint", None)
        if hit := _CFUNC_CACHE.get(key):
            return cast(ConstraintFunc, hit)
        layout = ConstraintLayout.from_compiled(self, self.constraint_names)
        cfunc, inclusive = build_constraint_cfunc(self.constraint_exprs, layout)
        constr = ConstraintFunc(
            cfunc=cfunc,
            names=self.constraint_names,
            n_var=layout.n_var,
            n_par=layout.n_par,
            inclusive=inclusive,
        )
        _CFUNC_CACHE[key] = constr
        return constr

    def __cached_residual_cf(self, opset: OpTable, key: tuple) -> Any:
        if hit := _CFUNC_CACHE.get(key):
            return hit
        cf = build_cfunc(self.objective_eqs, ResidualLayout.from_compiled(self), opset)
        _CFUNC_CACHE[key] = cf
        return cf

    def construct_objective_cfunc(self) -> Any:
        """Built a native (C callable) function that evaluates the model's residuals at a given state and parameter vector.

        Returns
        -------
        Any
            Function as a ``numba`` ``@cfunc`` (C ABI); used by native kernels to evaluate the model's residuals.

        """
        key = (self.config_token, "residual_c128", None)
        return self.__cached_residual_cf(C128Ops(), key)

    def construct_objective_cfunc_bicomplex(self) -> Any:
        """Build a native (C callable) function that evaluates the model's residuals using bicomplex numbers for second-order Hessian computation.

        Returns
        -------
        Any
            ``numba`` ``@cfunc`` (C ABI) that evaluates the model's residuals using bicomplex numbers, used for second-order Hessian computation.

        """
        key = (self.config_token, "residual_bicomplex", None)
        return self.__cached_residual_cf(BicomplexOps(), key)

    def build_affine_measurement_matrices(
        self,
        params: NDF,
        observables: Sequence[str],
        ss: NDF,
    ) -> tuple[NDF, NDF]:
        """Build C and d matrices for the affine measurement equation ``y = C x + d + v``, evaluated at the given parameter vector and steady state.

        Parameters
        ----------
        params : NDArray[float64]
            Model parameters as a vector in calibration order.
        observables : Sequence[str]
            Obserbable names to include in the measurement equation, in the order they should appear in the output matrices.
        ss : NDF
            Steady state vector of the model's variables, evaluated at the given parameter vector.

        Returns
        -------
        tuple[NDF, NDF]
            C : Measurement matrix mapping state variables to observables.
            d : Constant vector for the affine measurement equation.

        """
        if params.shape[0] != self.n_par:
            raise ValueError(
                f"Parameter vector length {params.shape[0]} != {self.n_par}"
            )

        meas_addr = self.construct_measurement_cfunc(observables).address
        jac_addr = self.construct_measurement_jacobian_cfunc(observables).address
        n_obs = len(observables)

        d = measurement_eval(meas_addr, ss, params, n_obs)
        C = jacobian_eval(jac_addr, ss, params, n_obs, self.n_var)
        return C, d

    def _normalize_observables(
        self,
        observables: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if observables is None:
            return self.observable_names

        obs = tuple(observables)
        if obs == self.observable_names:
            return self.observable_names

        obsset = set(obs)
        if len(obsset) != len(obs):
            raise ValueError("Observable list contains duplicates.")

        obs_idx = {name: i for i, name in enumerate(self.observable_names)}
        if missing := obsset - obs_idx.keys():
            raise KeyError(
                f"Unknown observables not in compiled model: {sorted(missing)!r}"
            )

        return tuple(sorted(obs, key=lambda name: obs_idx[name]))

    def construct_measurement_cfunc(
        self,
        observables: Sequence[str] | None = None,
    ) -> Any:
        """Build a native (C callable) function that evaluates the model's measurement equations at a given state and parameter vector, for the specified observables.

        Parameters
        ----------
        observables : Sequence[str] | None
            List of observable names to include in the measurement function. If ``None``, all observables are included.

        Returns
        -------
        Any
            ``numba`` ``@cfunc`` (C ABI) that evaluates the model's measurement equations for the specified observables.

        """
        obs = self._normalize_observables(observables)
        key = (self.config_token, "measurement", obs)
        if hit := _CFUNC_CACHE.get(key):
            return hit  # Any (CFunc)

        layout = MeasurementLayout.from_compiled(self, obs)
        exprs = [self.observable_eqs[i] for i in layout.observable_indices]
        cf = build_measurement_cfunc(exprs, layout)
        _CFUNC_CACHE[key] = cf
        return cf

    def construct_measurement_jacobian_cfunc(
        self,
        observables: Sequence[str] | None = None,
    ) -> Any:
        """Build a native (C callable) function that evaluates the Jacobian of the model's measurement equations with respect to the model's variables, for the specified observables.

        Parameters
        ----------
        observables : Sequence[str] | None
            List of observable names to include in the Jacobian function. If ``None``, all observables are included.

        Returns
        -------
        Any
            ``numba`` ``@cfunc`` (C ABI) that evaluates the Jacobian of the model's measurement equations for the specified observables.

        """
        obs = self._normalize_observables(observables)
        key = (self.config_token, "measurement_jac", obs)
        if hit := _CFUNC_CACHE.get(key):
            return hit  # Any (CFunc)

        base = MeasurementLayout.from_compiled(self, obs)
        n_var = base.n_var
        obs_idx = {name: i for i, name in enumerate(self.observable_names)}
        # Flat row-major (obs, var) jacobian exprs for the selected observables.
        exprs = [
            self.measurement_jacobian_eqs[obs_idx[name] * n_var + j]
            for name in obs
            for j in range(n_var)
        ]
        layout = MeasurementLayout(
            slot=base.slot,
            n_var=n_var,
            n_par=base.n_par,
            n_obs=len(exprs),
        )
        cf = build_measurement_cfunc(exprs, layout)
        _CFUNC_CACHE[key] = cf
        return cf

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
