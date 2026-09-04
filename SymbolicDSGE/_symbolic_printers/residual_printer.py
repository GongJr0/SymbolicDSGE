"""Residual expression printers for native perturbation callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import sympy as sp
from numba import cfunc, types

from .base import ExpressionPrinter, OpTable

if TYPE_CHECKING:
    from ..core.compiled_model import CompiledModel


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

    slot: dict[str, tuple[str, int]]
    n_var: int
    n_par: int
    n_exog: int

    @property
    def n_expr(self) -> int:
        # One residual per variable: a system that is not square is degenerate.
        return self.n_var

    @classmethod
    def from_compiled(cls, compiled: CompiledModel) -> ResidualLayout:
        slot: dict[str, tuple[str, int]] = {}
        for i, name in enumerate(compiled.var_names):
            slot[f"fwd_{name}"] = ("fwd", i)
            slot[f"cur_{name}"] = ("cur", i)
            slot[f"prev_{name}"] = ("prev", i)

        for i, shock in enumerate(compiled.shock_names):
            slot[shock] = ("eps", i)

        for j, p in enumerate(compiled.calib_params):
            slot[p] = ("par", j)
        return cls(
            slot=slot,
            n_var=compiled.n_var,
            n_par=compiled.n_par,
            n_exog=compiled.n_exog,
        )


class ResidualPrinter(ExpressionPrinter):
    @property
    def allocated_dtype(self) -> str:
        return "np.complex128"

    @property
    def context_name(self) -> str:
        return "residual"


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
        f"    eps = carray(eps_ptr, ({w * layout.n_exog},))",
        f"    par = carray(par_ptr, ({w * layout.n_par},))",
        f"    out = carray(out_ptr, ({w * layout.n_expr},))",
    ]
    src = "\n".join(
        [
            *table.prelude_imports,
            "from numba import carray",
            "",
            "def _residual_cf(fwd_ptr, cur_ptr, prev_ptr, eps_ptr, par_ptr, out_ptr):",
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
        types.CPointer(types.complex128),
        types.CPointer(types.complex128),
    )
    return cfunc(sig)(ns["_residual_cf"])
