"""Constraint expression printers for native regime callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import sympy as sp
from numba import cfunc, types
from sympy import Symbol

from .base import ExpressionPrinter
from .ops import ConstraintOpTable, ConstraintOps

#: Relational node types mapped to the op that renders them.
_RELATIONAL_OPS: dict[type, Callable[[ConstraintOpTable, str, str], str]] = {
    sp.StrictLessThan: lambda ops, a, b: ops.lt(a, b),
    sp.LessThan: lambda ops, a, b: ops.le(a, b),
    sp.StrictGreaterThan: lambda ops, a, b: ops.gt(a, b),
    sp.GreaterThan: lambda ops, a, b: ops.ge(a, b),
    sp.Equality: lambda ops, a, b: ops.eq(a, b),
    sp.Unequality: lambda ops, a, b: ops.ne(a, b),
}


@dataclass(slots=True)
class ConstraintLayout:
    """Maps constraint symbols to native buffer slots."""

    slot: dict[Any, tuple[str, int]]
    n_var: int
    n_par: int
    constraint_names: tuple[str, ...] = ()

    @property
    def n_flag(self) -> int:
        return 2 * len(self.constraint_names)

    @property
    def n_expr(self) -> int:
        return self.n_flag

    @classmethod
    def from_compiled(
        cls, compiled: Any, constraint_names: tuple[str, ...] | list[str]
    ) -> ConstraintLayout:
        slot: dict[Any, tuple[str, int]] = {}
        for i, name in enumerate(compiled.var_names):
            slot[Symbol(f"cur_{name}")] = ("cur", i)
        for j, p in enumerate(compiled.calib_params):
            slot[p] = ("par", j)
        return cls(
            slot=slot,
            n_var=len(compiled.var_names),
            n_par=len(compiled.calib_params),
            constraint_names=tuple(constraint_names),
        )


class ConstraintPrinter(ExpressionPrinter):
    def __init__(self, ops: ConstraintOpTable) -> None:
        super().__init__(ops)
        self.cops = ops

    @property
    def context_name(self) -> str:
        return "constraint"

    def render(self, expr: Any) -> str:
        render_op = _RELATIONAL_OPS.get(type(expr))
        if render_op is not None:
            lhs, rhs = expr.args
            return render_op(self.cops, self.render(lhs), self.render(rhs))
        if isinstance(expr, sp.And):
            return self._fold(expr.args, self.cops.and_)
        if isinstance(expr, sp.Or):
            return self._fold(expr.args, self.cops.or_)
        if isinstance(expr, sp.Not):
            return self.cops.not_(self.render(expr.args[0]))
        return super().render(expr)

    def _fold(self, args: Any, op: Callable[[str, str], str]) -> str:
        result = self.render(args[0])
        for arg in args[1:]:
            result = op(result, self.render(arg))
        return result


def build_constraint_cfunc(
    exprs: list[Any], layout: ConstraintLayout, ops: ConstraintOpTable | None = None
) -> Any:
    table: ConstraintOpTable = ConstraintOps() if ops is None else ops
    body = ConstraintPrinter(table).emit(exprs, layout, allocate=False)
    preamble = [
        f"    cur = carray(cur_ptr, ({layout.n_var},))",
        f"    par = carray(par_ptr, ({layout.n_par},))",
        f"    out = carray(out_ptr, ({layout.n_flag},))",
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
        types.CPointer(table.numba_type),
        types.CPointer(table.numba_type),
        types.CPointer(table.out_numba_type),
    )
    return cfunc(sig)(ns["_constraint_cf"])
