"""First order linearization of a compiled model about a numeric steady state."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Mapping, NamedTuple, NoReturn

import numpy as np
import sympy as sp
from numpy import float64
from numpy.typing import NDArray
from sympy import Eq, Expr, Function, Symbol
from sympy.core.function import AppliedUndef

from .._ckernels.core import steady_state_newton
from ..core.compiled_model import CompiledModel
from ..core.resolvers import resolve_ss_seed
from ..core.config import SymbolGetterDict, FunctionGetterDict
from .view import PointLinearView

NDF = NDArray[float64]


class LinearizationMethod(StrEnum):
    """Linearization methods for transforming variables in a symbolic model.

    Attributes
    ----------
    LOG : Literal["log"]
        Log-linearization
    LEVEL : Literal["level"]
        Level-linearization
    """

    LOG = "log"
    LEVEL = "level"

    @classmethod
    def _missing_(cls, value: object) -> NoReturn:
        raise ValueError(
            f"'{value}' is not a valid {cls.__name__}. Valid options are: {list(cls._value2member_map_)}"
        )


class Linearizer:
    """Per-variable transforms, applied to a compiled model at a parameterization.

    ``methods`` names the model's own variables. A compiler-minted lag or lead
    takes its origin's method: an aux is the same variable at another date, and
    the identity tying the two only survives linearization when both sides carry
    the same transform.
    """

    def __init__(
        self,
        compiled: CompiledModel,
        methods: Mapping[str, Annotated[str, Literal["log", "level"]]] | None = None,
        *,
        default: Literal["log", "level"] = "level",
    ) -> None:
        layout = compiled.layout
        requested = {str(name): method for name, method in (methods or {}).items()}

        if minted := sorted(set(requested) & set(layout.generated_names)):
            raise ValueError(
                f"`methods` names compiler-minted variable(s) {minted}, which take "
                f"their origin's method. Name the origin instead."
            )
        if unknown := sorted(set(requested) - set(layout.declared_names)):
            raise ValueError(
                f"`methods` names variables the model does not have: {unknown}. "
                f"Model variables: {list(layout.declared_names)}."
            )

        self._compiled = compiled
        self._methods = {
            name: LinearizationMethod(
                requested.get(layout.aux_origin.get(name, name), default)
            )
            for name in compiled.var_names
        }

    def at(self, params: Mapping[str, float] | None = None) -> PointLinearView:
        """Linearize at the steady state implied by ``params``.

        Parameters
        ----------
        params : Mapping[str, float] | None
            Parameter overrides on the model's calibration. ``None`` linearizes at
            the config calibration.

        Returns
        -------
        PointLinearView
            Linearized view of the model at the steady state implied by the calibration parameters.

        Notes
        -----
        This view is only a diagnostic representation and no solver kernel requires pre-linearization.
        For any model specified in levels, using a config from `PointLinearView.to_config()` lock the
        model to the linearization at the config calibration and the steady state at zero. Mind that
        estimating a model numerically linearized per parameter vector is not equivalent to estimating
        a point-linearized model whose original equations are in levels.
        """
        compiled = self._compiled
        parameters = _resolve_parameters(compiled, params)
        point = _solve_steady_state(compiled, parameters)
        _reject_nonpositive_log_point(self._methods, point)

        shocks = frozenset(compiled.config.shocks)
        equations = compiled.config.equations

        return PointLinearView(
            model={
                name: _linearize_equation(eq, self._methods, point, shocks)
                for name, eq in equations.model.items()
            },
            observables=SymbolGetterDict(
                {
                    name: _expand(expr, self._methods, point, shocks).affine
                    for name, expr in equations.observables.items()
                }
            ),
            parameters=parameters,
            steady_state=FunctionGetterDict(
                {Function(name): point[name] for name in compiled.layout.declared_names}
            ),
            _config=compiled.config,
        )


def linearize_model(
    compiled: CompiledModel,
    methods: Mapping[str, Annotated[str, Literal["log", "level"]]] | None = None,
    *,
    default: Literal["log", "level"] = "level",
) -> PointLinearView:
    """Linearize a model at its steady state at the config's calibration parameters.

    Parameters
    ----------
    compiled : CompiledModel
        Compiled model to linearize.
    methods : Mapping[str, Literal["log", "level"]] | None
        Linearization methods per variable. `None` uses the default for all variables.
        When `methods` is provided with omissions, the default is filled in for the missing variables.
    default : Literal["log", "level"]
        Default linearization method for variables not specified in `methods`.

    Returns
    -------
    PointLinearView
        Linearized view of the model at the steady state implied by the calibration parameters.

    Notes
    -----
    This view is only a diagnostic representation and no solver kernel requires pre-linearization.
    For any model specified in levels, using a config from `PointLinearView.to_config()` lock the
    model to the linearization at the config calibration and the steady state at zero. Mind that
    estimating a model numerically linearized per parameter vector is not equivalent to estimating
    a point-linearized model whose original equations are in levels.
    """
    return Linearizer(compiled, methods, default=default).at()


def _linearize_equation(
    equation: Eq,
    methods: Mapping[str, LinearizationMethod],
    point: Mapping[str, float],
    shocks: frozenset[Symbol],
) -> Eq:
    # Each side drops its own value at the expansion point, which is what leaves
    # a deviation system and makes a second pass over the result a no-op.
    lhs = _expand(equation.lhs, methods, point, shocks)  # pyright: ignore
    rhs = _expand(equation.rhs, methods, point, shocks)  # pyright: ignore
    return sp.Eq(
        lhs.affine - lhs.constant, rhs.affine - rhs.constant
    )  # pyright: ignore


class _Expansion(NamedTuple):
    affine: Expr
    constant: Expr


def _expand(
    expr: Expr,
    methods: Mapping[str, LinearizationMethod],
    point: Mapping[str, float],
    shocks: frozenset[Symbol],
) -> _Expansion:
    """Affine expansion of ``expr`` about the point, and its value there.

    The transform is substituted in place of each variable call, so a LOG or
    TAYLOR variable comes out denoting its deviation rather than its level. One
    ``xreplace`` pass is enough because the expansion point is numeric and cannot
    reintroduce a variable call.

    Shocks expand alongside the variables, about zero: a shock that enters
    nonlinearly is linearized like anything else, and the result is first order
    in the innovations as well as the states.
    """
    calls = _collect_calls(expr, methods)
    expanded = expr.xreplace(
        {
            call: _reconstruct(methods[_name(call)], point[_name(call)], call)
            for call in calls
        }
    )

    atoms: tuple[Expr, ...] = calls + tuple(  # pyright: ignore
        sorted(expanded.free_symbols & shocks, key=str)
    )
    zeros: dict[Expr, Expr] = {atom: sp.Integer(0) for atom in atoms}

    constant = expanded.xreplace(zeros)
    linear = sp.Add(
        *(
            sp.diff(expanded, atom).subs(zeros) * atom  # pyright: ignore
            for atom in atoms
        )
    )
    return _Expansion(constant + linear, constant)  # pyright: ignore


def _reconstruct(method: LinearizationMethod, point: float, call: AppliedUndef) -> Expr:
    if method is LinearizationMethod.LOG:
        return sp.Float(point) * sp.exp(call)
    return sp.Float(point) + call


def _collect_calls(
    expr: Expr, methods: Mapping[str, LinearizationMethod]
) -> tuple[AppliedUndef, ...]:
    # Sorted because atoms() hands back a set, and the emitted sum should be the
    # same expression on every run.
    calls = [call for call in expr.atoms(AppliedUndef) if _name(call) in methods]
    return tuple(sorted(calls, key=lambda call: (_name(call), sp.srepr(call))))


def _name(call: AppliedUndef) -> str:
    return str(call.func.__name__)


def _reject_nonpositive_log_point(
    methods: Mapping[str, LinearizationMethod], point: Mapping[str, float]
) -> None:
    offenders = {
        name: point[name]
        for name, method in methods.items()
        if method is LinearizationMethod.LOG
        and point[name] <= 1e-12  # tolerance for fp noise
    }
    if offenders:
        raise ValueError(
            f"log linearization needs a positive steady state, solved: {offenders}."
        )


def _resolve_parameters(
    compiled: CompiledModel, params: Mapping[str, float] | None
) -> SymbolGetterDict[float]:
    calibration = compiled.config.calibration.parameters
    overrides = {str(name): value for name, value in (params or {}).items()}
    if unknown := sorted(set(overrides) - set(compiled.calib_params)):
        raise ValueError(
            f"`params` names parameters the model does not have: {unknown}. "
            f"Model parameters: {list(compiled.calib_params)}."
        )
    return SymbolGetterDict(
        {
            name: float(overrides.get(name, calibration[name]))  # pyright: ignore
            for name in compiled.calib_params
        }
    )


def _solve_steady_state(
    compiled: CompiledModel, parameters: Mapping[Symbol, float]
) -> dict[str, float]:
    residual = compiled.construct_objective_cfunc()
    ss, _ = steady_state_newton(
        residual.address,
        resolve_ss_seed(compiled.config, compiled.layout, params=parameters),
        _parameter_vector(compiled, parameters),
        compiled.n_exog,
    )
    return {name: float(value) for name, value in zip(compiled.var_names, ss)}


def _parameter_vector(
    compiled: CompiledModel, parameters: Mapping[Symbol, float]
) -> NDF:
    return np.array(
        [parameters[Symbol(name)] for name in compiled.calib_params], dtype=float64
    )
