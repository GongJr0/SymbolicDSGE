"""Residual expression printers for native perturbation callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp
from numba import cfunc, njit, types
from sympy import Symbol

from .base import ExpressionPrinter
from .ops import C128Ops, OpTable


@dataclass
class ResidualLayout:
    """Maps residual symbols to native buffer slots."""

    slot: dict[Any, tuple[str, int]]
    n_var: int
    n_par: int
    n_eq: int

    @property
    def n_expr(self) -> int:
        return self.n_eq

    @classmethod
    def from_compiled(cls, compiled: Any) -> ResidualLayout:
        slot: dict[Any, tuple[str, int]] = {}
        for i, name in enumerate(compiled.var_names):
            slot[Symbol(f"fwd_{name}")] = ("fwd", i)
            slot[Symbol(f"cur_{name}")] = ("cur", i)
            slot[Symbol(f"prev_{name}")] = ("prev", i)
        for j, p in enumerate(compiled.calib_params):
            slot[p] = ("par", j)
        return cls(
            slot=slot,
            n_var=len(compiled.var_names),
            n_par=len(compiled.calib_params),
            n_eq=len(compiled.objective_eqs),
        )


class ResidualPrinter(ExpressionPrinter):
    @property
    def context_name(self) -> str:
        return "residual"


def build_njit(
    exprs: list[sp.Expr], layout: ResidualLayout, ops: OpTable | None = None
) -> Any:
    table: OpTable = C128Ops() if ops is None else ops
    body = ResidualPrinter(table).emit(exprs, layout, allocate=True)
    src = "\n".join(
        [
            *table.prelude_imports,
            "import numpy as np",
            "",
            "def _residual(fwd, cur, prev, par):",
            *body,
            "",
        ]
    )
    ns: dict[str, Any] = {}
    exec(src, ns)  # noqa: S102
    return njit(ns["_residual"])


def build_cfunc(
    exprs: list[sp.Expr], layout: ResidualLayout, ops: OpTable | None = None
) -> Any:
    table: OpTable = C128Ops() if ops is None else ops
    body = ResidualPrinter(table).emit(exprs, layout, allocate=False)
    w = table.elems_per_var
    preamble = [
        f"    fwd = carray(fwd_ptr, ({w * layout.n_var},))",
        f"    cur = carray(cur_ptr, ({w * layout.n_var},))",
        f"    prev = carray(prev_ptr, ({w * layout.n_var},))",
        f"    par = carray(par_ptr, ({w * layout.n_par},))",
        f"    out = carray(out_ptr, ({w * layout.n_eq},))",
    ]
    src = "\n".join(
        [
            *table.prelude_imports,
            "from numba import carray",
            "",
            "def _residual_cf(fwd_ptr, cur_ptr, prev_ptr, par_ptr, out_ptr):",
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
        types.CPointer(table.numba_type),
        types.CPointer(table.numba_type),
        types.CPointer(table.out_numba_type),
    )
    return cfunc(sig)(ns["_residual_cf"])
