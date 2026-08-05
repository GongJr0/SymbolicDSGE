"""Rewrite a model into the two-date form the solver pencil requires.

Model equations are authored in natural time, with offsets of -1, 0 and +1 and
shocks dated t. The pencil carries only offsets 0 and +1, and every shock has to
reach the state space through a variable, so both are lifted here into variables
the compiler introduces.

``v(t-n)`` becomes ``v_lag{n}(t)``, defined by the chain ``v_lag1(t+1) = v(t)``,
``v_lag2(t+1) = v_lag1(t)`` and onward, so a deeper lag numbers its aux rather
than stacking suffixes. A shock ``e`` becomes ``e_st(t)``, defined by
``e_st(t+1) = e``.

Lifting every shock, not only the ones that need it, is what collapses the shock
impact block into a gather: the raw shock symbol survives in exactly one
equation, its own, so the model's shock jacobian holds a single nonzero per
shock column and carries no entry on a control row.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Iterator

import sympy as sp
from sympy import Eq, Expr, Function, Symbol
from sympy.core.function import AppliedUndef

from .linearization import LinearizationMethod

if TYPE_CHECKING:
    from .config import ModelConfig

#: ``v(t-n)`` lifts to ``f"{v}{LAG_SUFFIX}{n}"``.
LAG_SUFFIX = "_lag"

#: shock ``e`` lifts to ``f"{e}{SHOCK_SUFFIX}"``.
SHOCK_SUFFIX = "_st"


class GeneratedKind(StrEnum):
    LAG = "lag"
    SHOCK = "shock"


@dataclass(frozen=True)
class GeneratedVariable:
    """A variable the desugaring introduced, and what it was minted from.

    ``origin`` is the declared variable for a lag and the shock symbol's name for
    a shock state. ``depth`` is the lag depth, 0 for a shock state.
    """

    name: str
    origin: str
    kind: GeneratedKind
    depth: int


@dataclass(frozen=True)
class DesugarResult:
    config: ModelConfig
    generated: tuple[GeneratedVariable, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.generated)

    def positions(self, order: Any) -> dict[str, int]:
        """``{name: position}`` for the generated variables within ``order``.

        Consumers hiding generated variables filter against this by name or by
        position, so the two views cannot drift apart.
        """
        index = {name: i for i, name in enumerate(order)}
        return {g.name: index[g.name] for g in self.generated if g.name in index}


def lag_name(base: str, depth: int) -> str:
    return f"{base}{LAG_SUFFIX}{depth}"


def shock_state_name(shock: str) -> str:
    return f"{shock}{SHOCK_SUFFIX}"


def desugar_model(conf: ModelConfig) -> DesugarResult:
    """Lift lags and shocks into compiler-generated variables.

    Lag depths are scanned over the reference equations, every regime and the
    observables together, so a regime that lags something the reference does not
    still finds its aux column in the reference pencil.
    """
    t = Symbol("t", integer=True)
    declared_funcs = list(conf.variables.variables)
    declared_names = [f.__name__ for f in declared_funcs]
    declared = set(declared_names)

    depths = _scan_lag_depths(conf, declared, t)
    generated = _mint(conf, declared_names, depths)
    _reject_collisions(conf, generated, declared)

    lag_funcs = {
        (g.origin, g.depth): sp.Function(g.name)
        for g in generated
        if g.kind is GeneratedKind.LAG
    }
    shock_subs: dict[Symbol, Expr] = {
        shock: sp.Function(shock_state_name(shock.name))(t) for shock in conf.shock_map
    }

    out = deepcopy(conf)

    def rewrite(expr: Any) -> Any:
        return _rewrite(expr, lag_funcs, shock_subs, declared, t)

    out.equations.model = {
        name: _rewrite_eq(eq, rewrite) for name, eq in out.equations.model.items()
    }
    if out.equations.regime:
        out.equations.regime = {
            key: {name: _rewrite_eq(eq, rewrite) for name, eq in replacements.items()}
            for key, replacements in out.equations.regime.items()
        }
    for obs, expr in list(out.equations.observable.items()):
        out.equations.observable[obs] = rewrite(expr)
    if out.equations.constraint:
        # Conditions stay contemporaneous, so only the shock lift applies; a lag
        # in a condition is left alone for the compiler's offset check to reject.
        for constraint in out.equations.constraint.values():
            constraint.bind = constraint.bind.xreplace(shock_subs)
            constraint.relax = constraint.relax.xreplace(shock_subs)

    _register(out, generated, t)
    return DesugarResult(config=out, generated=generated)


def _scan_lag_depths(
    conf: ModelConfig, declared: set[str], t: Symbol
) -> dict[str, int]:
    depths: dict[str, int] = {}
    for expr in _lagged_expressions(conf):
        for call in expr.atoms(AppliedUndef):
            info = _call_offset(call, declared, t)
            if info is None:
                continue
            name, offset = info
            if offset < 0:
                depths[name] = max(depths.get(name, 0), -offset)
    return depths


def _lagged_expressions(conf: ModelConfig) -> Iterator[Expr]:
    for eq in conf.equations.model.values():
        yield eq.lhs  # pyright: ignore[reportReturnType]
        yield eq.rhs  # pyright: ignore[reportReturnType]
    for replacements in (conf.equations.regime or {}).values():
        for eq in replacements.values():
            yield eq.lhs  # pyright: ignore[reportReturnType]
            yield eq.rhs  # pyright: ignore[reportReturnType]
    yield from conf.equations.observable.values()


def _call_offset(
    call: AppliedUndef, declared: set[str], t: Symbol
) -> tuple[str, int] | None:
    name = str(call.func.__name__)
    if name not in declared or not call.args:
        return None
    arg0 = call.args[0]
    if t not in getattr(arg0, "free_symbols", set()):
        return None
    offset = sp.simplify(arg0 - t)
    if offset.is_Integer or offset.is_integer is True:
        return name, int(offset)
    return None


def _mint(
    conf: ModelConfig, declared_names: list[str], depths: dict[str, int]
) -> tuple[GeneratedVariable, ...]:
    # Lags in declaration order then depth, shocks in shock_map order, so the
    # generated block is stable across runs.
    generated: list[GeneratedVariable] = []
    for base in declared_names:
        for depth in range(1, depths.get(base, 0) + 1):
            generated.append(
                GeneratedVariable(
                    name=lag_name(base, depth),
                    origin=base,
                    kind=GeneratedKind.LAG,
                    depth=depth,
                )
            )
    for shock in conf.shock_map:
        generated.append(
            GeneratedVariable(
                name=shock_state_name(shock.name),
                origin=shock.name,
                kind=GeneratedKind.SHOCK,
                depth=0,
            )
        )
    return tuple(generated)


def _reject_collisions(
    conf: ModelConfig, generated: tuple[GeneratedVariable, ...], declared: set[str]
) -> None:
    taken = sorted(g.name for g in generated if g.name in declared)
    if taken:
        raise ValueError(
            f"Desugaring would generate variable(s) the model already declares: "
            f"{taken}. Rename them, or drop the second call if this config was "
            f"already desugared."
        )
    equations = set(conf.equations.model)
    clash = sorted(g.name for g in generated if g.name in equations)
    if clash:
        raise ValueError(
            f"Desugaring would generate equation(s) the model already declares: "
            f"{clash}. Rename them."
        )


def _rewrite(
    expr: Any,
    lag_funcs: dict[tuple[str, int], Function],
    shock_subs: dict[Symbol, Expr],
    declared: set[str],
    t: Symbol,
) -> Any:
    subs: dict[Any, Any] = dict(shock_subs)
    for call in expr.atoms(AppliedUndef):
        info = _call_offset(call, declared, t)
        if info is None:
            continue
        name, offset = info
        if offset < 0:
            subs[call] = lag_funcs[(name, -offset)](t)
    return expr.xreplace(subs)


def _rewrite_eq(eq: Eq, rewrite: Any) -> Eq:
    return sp.Eq(rewrite(eq.lhs), rewrite(eq.rhs))


def _register(
    conf: ModelConfig, generated: tuple[GeneratedVariable, ...], t: Symbol
) -> None:
    """Append the aux variables and their defining equations.

    Runs after the rewrite so the shock lift does not reach the aux equations and
    turn ``e_st(t+1) = e`` into ``e_st(t+1) = e_st(t)``.
    """
    shock_by_name = {shock.name: shock for shock in conf.shock_map}
    for g in generated:
        func = sp.Function(g.name)
        conf.variables.variables.append(func)

        if g.kind is GeneratedKind.SHOCK:
            conf.variables.ss_seed[func] = sp.Integer(0)
            conf.variables.linearization[func] = LinearizationMethod.NONE
            conf.equations.model[g.name] = sp.Eq(
                func(t + 1), shock_by_name[g.origin]  # pyright: ignore[reportCallIssue]
            )
            continue

        origin = sp.Function(g.origin)
        source = (
            origin if g.depth == 1 else sp.Function(lag_name(g.origin, g.depth - 1))
        )
        # The aux tracks its origin exactly, so it shares the origin's expansion
        # point and linearization.
        conf.variables.ss_seed[func] = conf.variables.ss_seed[origin]
        conf.variables.linearization[func] = conf.variables.linearization[origin]
        conf.equations.model[g.name] = sp.Eq(
            func(t + 1), source(t)  # pyright: ignore[reportCallIssue]
        )
