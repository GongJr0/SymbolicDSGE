"""Rewrite a model into the three-date form the solver pencil requires.

Model equations are authored in natural time, with offsets of -1, 0 and +1 and
shocks dated t. The residual carries all three dates and the innovations
directly, so a single lag, a single lead and a shock all reach the pencil as
they are written.

Only depth beyond one still needs lifting, in either direction. ``v(t-n)`` for
``n >= 2`` becomes ``v_lag{n-1}(t-1)``, on the chain ``v_lag1(t) = v(t-1)``,
``v_lag2(t) = v_lag1(t-1)`` and onward; ``v(t+n)`` becomes ``v_lead{n-1}(t+1)``,
on the mirror chain ``v_lead1(t) = v(t+1)``, ``v_lead2(t) = v_lead1(t+1)``. A
depth of one mints nothing, so a model written with ordinary lags and leads
compiles to exactly the variables it declares.

A lag aux occurs at ``t-1`` and a lead aux does not, so the two land in
different halves of the compiled layout: lag auxes join the states, lead auxes
the controls.
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

#: ``v(t-n)`` lifts to ``f"{v}{LAG_SUFFIX}{n-1}"``, ``v(t+n)`` to
#: ``f"{v}{LEAD_SUFFIX}{n-1}"``.
LAG_SUFFIX = "_lag"
LEAD_SUFFIX = "_lead"


@dataclass(frozen=True)
class GeneratedVariable:
    """An aux, the declared variable it tracks, and how far it is displaced.

    ``depth`` is signed and carries the direction: negative is a lag, positive a
    lead. ``v_lag{k}`` holds ``origin`` at ``t - k`` and ``v_lead{k}`` holds it
    at ``t + k``, so either serves a displacement of ``k + 1`` when read one
    date further out.
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


def aux_name(base: str, depth: int) -> str:
    """Name of ``base``'s aux at a signed ``depth``, negative for a lag."""
    return f"{base}{LAG_SUFFIX if depth < 0 else LEAD_SUFFIX}{abs(depth)}"


def desugar_model(conf: ModelConfig) -> DesugarResult:
    """Lift lags and leads past the first into compiler-generated variables.

    Depths are scanned over the reference equations, every regime and the
    observables together, so a regime that displaces something the reference
    does not still finds its aux column in the reference pencil.
    """
    t = Symbol("t", integer=True)
    declared_funcs = list(conf.variables.variables)
    declared_names = [f.__name__ for f in declared_funcs]
    declared = set(declared_names)

    lags, leads = _scan_offset_depths(conf, declared, t)
    generated = _mint(conf, declared_names, lags, leads)
    _reject_collisions(conf, generated, declared)

    aux_funcs = {(g.origin, g.depth): sp.Function(g.name) for g in generated}

    out = deepcopy(conf)

    def rewrite(expr: Any) -> Any:
        return _rewrite(expr, aux_funcs, declared, t)  # pyright: ignore

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


def _scan_offset_depths(
    conf: ModelConfig, declared: set[str], t: Symbol
) -> tuple[dict[str, int], dict[str, int]]:
    """Deepest lag and deepest lead per variable, as signed offsets.

    A variable displaced in both directions appears in both maps.
    """
    lags: dict[str, int] = {}
    leads: dict[str, int] = {}
    for expr in _offset_expressions(conf):
        for call in expr.atoms(AppliedUndef):
            info = _call_offset(call, declared, t)
            if info is None:
                continue
            name, offset = info
            if offset < 0:
                lags[name] = min(lags.get(name, 0), offset)
            elif offset > 0:
                leads[name] = max(leads.get(name, 0), offset)
    return lags, leads


def _offset_expressions(conf: ModelConfig) -> Iterator[Expr]:
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
    raise ValueError(
        f"Equation {call} displaces {name} by a non-integer offset {offset}."
    )


def _mint(
    conf: ModelConfig,
    declared_names: list[str],
    lags: dict[str, int],
    leads: dict[str, int],
) -> tuple[GeneratedVariable, ...]:
    # Declaration order, then each base's lags before its leads, so the
    # generated block is stable across runs. Depth 1 is carried by the
    # residual's own outer dates, so each chain starts one short of the depth it
    # serves.
    generated: list[GeneratedVariable] = []
    for base in declared_names:
        lag_depth = lags.get(base, 0)
        lead_depth = leads.get(base, 0)

        for depth in range(1, abs(lag_depth)):
            generated.append(
                GeneratedVariable(
                    name=aux_name(base, -depth),
                    origin=base,
                    depth=-depth,
                )
            )
        for depth in range(1, lead_depth):
            generated.append(
                GeneratedVariable(
                    name=aux_name(base, depth),
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
            f"{taken}. Rename them."
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
    funcs: dict[tuple[str, int], Function],
    declared: set[str],
    t: Symbol,
) -> Any:
    """Depth 1 survives as written; deeper reads its aux one date further out."""
    subs: dict[Any, Any] = {}
    for call in expr.atoms(AppliedUndef):
        info = _call_offset(call, declared, t)
        if info is None:
            continue
        name, offset = info
        if offset < -1:
            subs[call] = funcs[(name, offset + 1)](t - 1)  # pyright: ignore
        elif offset > 1:
            subs[call] = funcs[(name, offset - 1)](t + 1)  # pyright: ignore
    return expr.xreplace(subs)


def _rewrite_eq(eq: Eq, rewrite: Any) -> Eq:
    return sp.Eq(rewrite(eq.lhs), rewrite(eq.rhs))  # pyright: ignore


def _register(
    conf: ModelConfig, generated: tuple[GeneratedVariable, ...], t: Symbol
) -> None:
    """Append the aux variables and their defining equations.

    ``v_lag{k}(t) = v_lag{k-1}(t-1)`` and ``v_lead{k}(t) = v_lead{k-1}(t+1)``
    are written in natural time, at dates the residual already carries, so
    neither chain needs a shift of its own.
    """
    for g in generated:
        func = sp.Function(g.name)
        conf.variables.variables.append(func)  # pyright: ignore

        origin = sp.Function(g.origin)
        source = (
            origin
            if abs(g.depth) == 1
            else sp.Function(aux_name(g.origin, g.depth - (1 if g.depth > 0 else -1)))
        )
        # The aux tracks its origin exactly, so it shares the origin's expansion
        # point and linearization.
        conf.variables.ss_seed[func] = conf.variables.ss_seed[origin]  # pyright: ignore
        conf.variables.linearization[func] = (  # pyright: ignore
            conf.variables.linearization[origin]
        )
        conf.equations.model[g.name] = sp.Eq(  # pyright: ignore
            func(t), source(t - 1) if g.depth < 0 else source(t + 1)
        )
