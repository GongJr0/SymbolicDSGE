"""Rewrite a model into the three-date form the solver pencil requires.

Model equations are authored in natural time, with offsets of -1, 0 and +1 and
shocks dated t. The residual carries all three dates and the innovations
directly, so a single lag and a shock both reach the pencil as they are written.

Only depth beyond one still needs lifting: ``v(t-2)`` and deeper are outside the
three dates, so ``v(t-n)`` for ``n >= 2`` becomes ``v_lag{n-1}(t-1)``, defined by
the chain ``v_lag1(t) = v(t-1)``, ``v_lag2(t) = v_lag1(t-1)`` and onward. A depth
of one mints nothing, so a model written with ordinary lags compiles to exactly
the variables it declares.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import sympy as sp
from sympy import Eq, Expr, Function, Symbol
from sympy.core.function import AppliedUndef

if TYPE_CHECKING:
    from .config import ModelConfig

#: ``v(t-n)`` lifts to ``f"{v}{LAG_SUFFIX}{n}"``.
LAG_SUFFIX = "_lag"


@dataclass(frozen=True)
class GeneratedVariable:
    """A lag aux, the declared variable it tracks, and how far back it starts.

    ``v_lag{depth}`` holds ``origin`` at ``t - depth``, so it serves a lag of
    ``depth + 1`` when read one date back.
    """

    name: str
    origin: str
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


def desugar_model(conf: ModelConfig) -> DesugarResult:
    """Lift lags past the first into compiler-generated variables.

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

    lag_funcs = {(g.origin, g.depth): sp.Function(g.name) for g in generated}

    out = deepcopy(conf)

    def rewrite(expr: Any) -> Any:
        return _rewrite(expr, lag_funcs, declared, t)  # pyright: ignore

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
    # Lags in declaration order then depth, so the generated block is stable
    # across runs. Depth 1 is carried by the residual's own lagged date, so the
    # chain starts one short of the depth it serves.
    generated: list[GeneratedVariable] = []
    for base in declared_names:
        for depth in range(1, depths.get(base, 0)):
            generated.append(
                GeneratedVariable(
                    name=lag_name(base, depth),
                    origin=base,
                    depth=depth,
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
    declared: set[str],
    t: Symbol,
) -> Any:
    """Depth 1 survives as written; deeper lags read their aux one date back."""
    subs: dict[Any, Any] = {}
    for call in expr.atoms(AppliedUndef):
        info = _call_offset(call, declared, t)
        if info is None:
            continue
        name, offset = info
        if offset < -1:
            subs[call] = lag_funcs[(name, -offset - 1)](t - 1)  # pyright: ignore
    return expr.xreplace(subs)


def _rewrite_eq(eq: Eq, rewrite: Any) -> Eq:
    return sp.Eq(rewrite(eq.lhs), rewrite(eq.rhs))  # pyright: ignore


def _register(
    conf: ModelConfig, generated: tuple[GeneratedVariable, ...], t: Symbol
) -> None:
    """Append the aux variables and their defining equations.

    ``v_lag{k}(t) = v_lag{k-1}(t-1)`` is written in natural time, at the two
    dates the residual already carries, so the chain needs no shift of its own.
    """
    for g in generated:
        func = sp.Function(g.name)
        conf.variables.variables.append(func)  # pyright: ignore

        origin = sp.Function(g.origin)
        source = (
            origin if g.depth == 1 else sp.Function(lag_name(g.origin, g.depth - 1))
        )
        # The aux tracks its origin exactly, so it shares the origin's expansion
        # point and linearization.
        conf.variables.ss_seed[func] = conf.variables.ss_seed[origin]  # pyright: ignore
        conf.variables.linearization[func] = (  # pyright: ignore
            conf.variables.linearization[origin]
        )
        conf.equations.model[g.name] = sp.Eq(func(t), source(t - 1))  # pyright: ignore
