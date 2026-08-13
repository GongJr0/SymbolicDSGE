"""Constraint expression printers for native regime callbacks.

A condition is printed as its signed distance to its own boundary, positive
where the condition holds, rather than as a 0/1 flag. The native latch recovers
the flag from the sign, so the two can never disagree, and the same number is
the error the guess-and-verify loop ranks its iterations by.

Only the boundary itself is lost that way: ``x < 0`` and ``x <= 0`` are the same
distance and differ only at zero. That is static, so it travels once as
``inclusive`` rather than per evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, TYPE_CHECKING

import sympy as sp
from numba import cfunc, types
from sympy import Symbol

from .base import ExpressionPrinter, OpTable
from .measurement_printer import F64Ops

if TYPE_CHECKING:
    from ..core.compiled_model import CompiledModel


class ConstraintOpTable(OpTable, Protocol):
    """Op table extended with the connectives a distance folds over."""

    def min_(self, a: str, b: str) -> str: ...
    def max_(self, a: str, b: str) -> str: ...


class ConstraintOps(F64Ops, ConstraintOpTable):
    """Real valued backend emitting the distance to each boundary."""

    def min_(self, a: str, b: str) -> str:
        return f"min({a}, {b})"

    def max_(self, a: str, b: str) -> str:
        return f"max({a}, {b})"


#: Relational node types mapped to the op that renders their distance, oriented
#: so that a satisfied condition is positive. An equality has no distance.
_RELATIONAL_OPS: dict[type, Callable[[ConstraintOpTable, str, str], str]] = {
    sp.StrictLessThan: lambda ops, a, b: ops.sub(b, a),
    sp.LessThan: lambda ops, a, b: ops.sub(b, a),
    sp.StrictGreaterThan: lambda ops, a, b: ops.sub(a, b),
    sp.GreaterThan: lambda ops, a, b: ops.sub(a, b),
}

#: Whether each relational holds *at* its boundary, where the distance is zero.
_RELATIONAL_INCLUSIVE: dict[type, bool] = {
    sp.StrictLessThan: False,
    sp.LessThan: True,
    sp.StrictGreaterThan: False,
    sp.GreaterThan: True,
}


@dataclass(slots=True)
class ConstraintLayout:
    """Maps constraint symbols to native buffer slots."""

    slot: dict[Symbol, tuple[str, int]]
    n_var: int
    n_par: int
    constraint_names: tuple[str, ...] = ()

    @property
    def n_cond(self) -> int:
        """Distances written per call: bind then relax, per constraint."""
        return 2 * len(self.constraint_names)

    @property
    def n_expr(self) -> int:
        return self.n_cond

    @classmethod
    def from_compiled(
        cls, compiled: CompiledModel, constraint_names: tuple[str, ...] | list[str]
    ) -> ConstraintLayout:
        slot: dict[Symbol, tuple[str, int]] = {}
        for i, name in enumerate(compiled.var_names):
            slot[Symbol(f"cur_{name}")] = ("cur", i)
        for j, p in enumerate(compiled.calib_params):
            slot[p] = ("par", j)
        return cls(
            slot=slot,
            n_var=compiled.n_var,
            n_par=compiled.n_par,
            constraint_names=tuple(constraint_names),
        )


class ConstraintPrinter(ExpressionPrinter):
    def __init__(self, ops: ConstraintOpTable) -> None:
        super().__init__(ops)
        self.cops = ops

    @property
    def allocated_dtype(self) -> str:
        return "np.float64"

    @property
    def context_name(self) -> str:
        return "constraint"

    def render(self, expr: Any) -> str:
        render_op = _RELATIONAL_OPS.get(type(expr))
        if render_op is not None:
            lhs, rhs = expr.args
            return render_op(self.cops, self.render(lhs), self.render(rhs))
        # A connective is as satisfied as its least satisfied branch, or its
        # most satisfied one, which is the same fold the flags would have taken.
        if isinstance(expr, sp.And):
            return self._fold(expr.args, self.cops.min_)
        if isinstance(expr, sp.Or):
            return self._fold(expr.args, self.cops.max_)
        if isinstance(expr, sp.Not):
            return self.cops.neg(self.render(expr.args[0]))
        if isinstance(expr, sp.Rel):
            raise NotImplementedError(
                f"constraint printer: {type(expr).__name__} has no distance to "
                f"a boundary: {expr}"
            )
        return super().render(expr)

    def _fold(self, args: Any, op: Callable[[str, str], str]) -> str:
        result = self.render(args[0])
        for arg in args[1:]:
            result = op(result, self.render(arg))
        return result


def constraint_inclusive(exprs: list[Any]) -> int:
    """Bitmask of the conditions that hold at a distance of exactly zero.

    Bit ``k`` is slot ``k`` of the distance buffer. A connective's branches must
    agree, since either of them can be the one the fold selected.
    """
    mask = 0
    for k, expr in enumerate(exprs):
        mask |= int(_inclusive(expr)) << k
    return mask


def _inclusive(expr: Any) -> bool:
    at_boundary = _RELATIONAL_INCLUSIVE.get(type(expr))
    if at_boundary is not None:
        return at_boundary
    if isinstance(expr, (sp.And, sp.Or)):
        branches = {_inclusive(arg) for arg in expr.args}
        if len(branches) != 1:
            raise NotImplementedError(
                f"constraint printer: {type(expr).__name__} mixes strict and "
                f"inclusive comparisons, so its boundary is ambiguous: {expr}"
            )
        return branches.pop()
    if isinstance(expr, sp.Not):
        return not _inclusive(expr.args[0])
    raise NotImplementedError(
        f"constraint printer: {type(expr).__name__} has no boundary: {expr}"
    )


def build_constraint_cfunc(
    exprs: list[Any], layout: ConstraintLayout, ops: ConstraintOpTable | None = None
) -> tuple[Any, int]:
    """``(cfunc, inclusive)`` for the conditions, in bind/relax slot order.

    The cfunc writes ``layout.n_cond`` distances. ``inclusive`` carries the one
    thing their sign cannot: which conditions hold when the distance is zero.
    """
    table: ConstraintOpTable = ConstraintOps() if ops is None else ops
    body = ConstraintPrinter(table).emit(exprs, layout, allocate=False)
    preamble = [
        f"    cur = carray(cur_ptr, ({layout.n_var},))",
        f"    par = carray(par_ptr, ({layout.n_par},))",
        f"    out = carray(out_ptr, ({layout.n_cond},))",
    ]
    src = "\n".join(
        [
            *table.prelude_imports,
            "from numba import carray",
            "",
            "def _constraint_cf(cur_ptr, par_ptr, out_ptr):",
            *preamble,
            *body,
            "",
        ]
    )
    ns: dict[str, Any] = {}
    exec(src, ns)  # noqa: S102
    sig = types.void(
        types.CPointer(types.float64),
        types.CPointer(types.float64),
        types.CPointer(types.float64),
    )
    return cfunc(sig)(ns["_constraint_cf"]), constraint_inclusive(exprs)
