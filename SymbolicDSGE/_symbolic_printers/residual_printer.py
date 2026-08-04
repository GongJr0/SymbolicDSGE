"""Residual expression printers for native perturbation callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp
from numba import cfunc, njit, types
from sympy import Symbol

from .base import ExpressionPrinter, OpTable


class C128Ops(OpTable):
    """Complex valued backend used by first order perturbation preproc."""

    prelude_imports = ("import cmath",)
    elems_per_var = 1

    def const(self, v: float) -> str:
        return f"complex({float(v)!r}, 0.0)"

    def load(self, buf: str, idx: int) -> str:
        return f"{buf}[{idx}]"

    def store(self, buf: str, idx: int, expr: str) -> str:
        return f"{buf}[{idx}] = {expr}"

    def add(self, a: str, b: str) -> str:
        return f"({a} + {b})"

    def sub(self, a: str, b: str) -> str:
        return f"({a} - {b})"

    def mul(self, a: str, b: str) -> str:
        return f"({a} * {b})"

    def div(self, a: str, b: str) -> str:
        return f"({a} / {b})"

    def neg(self, a: str) -> str:
        return f"(-{a})"

    def real_scale(self, a: str, s: float) -> str:
        return f"({float(s)!r} * {a})"

    def exp(self, a: str) -> str:
        return f"cmath.exp({a})"

    def log(self, a: str) -> str:
        return f"cmath.log({a})"

    def sqrt(self, a: str) -> str:
        return f"cmath.sqrt({a})"


class BicomplexOps(OpTable):
    """Bicomplex backend used by second order Hessian preproc."""

    prelude_imports = (
        "from SymbolicDSGE.core.bicomplex import ("
        " bc_add, bc_sub, bc_neg, bc_mul, bc_div, bc_real_scale,"
        " bc_exp, bc_log, bc_sqrt )",
    )
    elems_per_var = 2

    def const(self, v: float) -> str:
        return f"(complex({float(v)!r}, 0.0), 0j)"

    def load(self, buf: str, idx: int) -> str:
        return f"({buf}[{2 * idx}], {buf}[{2 * idx + 1}])"

    def store(self, buf: str, idx: int, expr: str) -> str:
        return f"{buf}[{2 * idx}], {buf}[{2 * idx + 1}] = {expr}"

    def add(self, a: str, b: str) -> str:
        return f"bc_add({a}, {b})"

    def sub(self, a: str, b: str) -> str:
        return f"bc_sub({a}, {b})"

    def mul(self, a: str, b: str) -> str:
        return f"bc_mul({a}, {b})"

    def div(self, a: str, b: str) -> str:
        return f"bc_div({a}, {b})"

    def neg(self, a: str) -> str:
        return f"bc_neg({a})"

    def real_scale(self, a: str, s: float) -> str:
        return f"bc_real_scale({a}, {float(s)!r})"

    def exp(self, a: str) -> str:
        return f"bc_exp({a})"

    def log(self, a: str) -> str:
        return f"bc_log({a})"

    def sqrt(self, a: str) -> str:
        return f"bc_sqrt({a})"


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
    def allocated_dtype(self) -> str:
        return "np.complex128"

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
            "def _residual(fwd, cur, par):",
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
        f"    par = carray(par_ptr, ({w * layout.n_par},))",
        f"    out = carray(out_ptr, ({w * layout.n_eq},))",
    ]
    src = "\n".join(
        [
            *table.prelude_imports,
            "from numba import carray",
            "",
            "def _residual_cf(fwd_ptr, cur_ptr, par_ptr, out_ptr):",
            *preamble,
            *body,
            "",
        ]
    )
    ns: dict[str, Any] = {}
    exec(src, ns)  # noqa: S102
    sig = types.void(
        types.CPointer(types.complex128),
        types.CPointer(types.complex128),
        types.CPointer(types.complex128),
        types.CPointer(types.complex128),
    )
    return cfunc(sig)(ns["_residual_cf"])
