"""Op tables: one numeric target each, for the shared expression printer.

A table owns everything target specific: how a value is rendered, how wide it is
in its buffer, and the numba scalar the buffer carries. ``numba_type`` types the
input buffers and ``out_numba_type`` the output; they differ only for
constraints, where real comparisons produce integer flags.

A buffer holds ``elems_per_var`` scalars of ``numba_type`` per logical value, so
``BicomplexOps`` rides in a complex128 buffer two entries wide, matching
``bc256`` on the native side.
"""

from __future__ import annotations

from typing import Any, Protocol

from numba import complex128, float64, int8


class OpTable(Protocol):
    """Renders primitive operations for one numeric target."""

    prelude_imports: tuple[str, ...]
    elems_per_var: int
    numba_type: Any
    out_numba_type: Any

    def const(self, v: float) -> str: ...
    def load(self, buf: str, idx: int) -> str: ...
    def store(self, buf: str, idx: int, expr: str) -> str: ...
    def add(self, a: str, b: str) -> str: ...
    def sub(self, a: str, b: str) -> str: ...
    def mul(self, a: str, b: str) -> str: ...
    def div(self, a: str, b: str) -> str: ...
    def neg(self, a: str) -> str: ...
    def real_scale(self, a: str, s: float) -> str: ...
    def exp(self, a: str) -> str: ...
    def log(self, a: str) -> str: ...
    def sqrt(self, a: str) -> str: ...


class ConstraintOpTable(OpTable, Protocol):
    """Op table extended with the comparisons and connectives conditions need."""

    def lt(self, a: str, b: str) -> str: ...
    def le(self, a: str, b: str) -> str: ...
    def gt(self, a: str, b: str) -> str: ...
    def ge(self, a: str, b: str) -> str: ...
    def eq(self, a: str, b: str) -> str: ...
    def ne(self, a: str, b: str) -> str: ...
    def and_(self, a: str, b: str) -> str: ...
    def or_(self, a: str, b: str) -> str: ...
    def not_(self, a: str) -> str: ...


class F64Ops(OpTable):
    """Real valued cfunc printer ops."""

    prelude_imports = ("import math",)
    elems_per_var = 1
    numba_type = float64
    out_numba_type = float64

    def const(self, v: float) -> str:
        return repr(float(v))

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
        return f"math.exp({a})"

    def log(self, a: str) -> str:
        return f"math.log({a})"

    def sqrt(self, a: str) -> str:
        return f"math.sqrt({a})"


class C128Ops(OpTable):
    """Complex valued cfunc printer ops."""

    prelude_imports = ("import cmath",)
    elems_per_var = 1
    numba_type = complex128
    out_numba_type = complex128

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
    """Bicomplex valued cfunc printer ops.

    Two complex128 per value, laid out as ``bc256``'s ``a`` then ``b``, so the
    buffer stays complex128 and ``elems_per_var`` carries the width.
    """

    prelude_imports = (
        "from SymbolicDSGE.core.bicomplex import ("
        " bc_add, bc_sub, bc_neg, bc_mul, bc_div, bc_real_scale,"
        " bc_exp, bc_log, bc_sqrt )",
    )
    elems_per_var = 2
    numba_type = complex128
    out_numba_type = complex128

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


class ConstraintOps(F64Ops, ConstraintOpTable):
    """Real valued backend emitting 0/1 regime flags."""

    out_numba_type = int8

    def lt(self, a: str, b: str) -> str:
        return f"({a} < {b})"

    def le(self, a: str, b: str) -> str:
        return f"({a} <= {b})"

    def gt(self, a: str, b: str) -> str:
        return f"({a} > {b})"

    def ge(self, a: str, b: str) -> str:
        return f"({a} >= {b})"

    def eq(self, a: str, b: str) -> str:
        return f"({a} == {b})"

    def ne(self, a: str, b: str) -> str:
        return f"({a} != {b})"

    def and_(self, a: str, b: str) -> str:
        return f"({a} and {b})"

    def or_(self, a: str, b: str) -> str:
        return f"({a} or {b})"

    def not_(self, a: str) -> str:
        return f"(not {a})"

    def store(self, buf: str, idx: int, expr: str) -> str:
        return f"{buf}[{idx}] = 1 if {expr} else 0"
