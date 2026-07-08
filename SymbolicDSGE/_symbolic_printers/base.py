"""Shared SymPy expression printer machinery."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Protocol

import sympy as sp


class Layout(Protocol):
    """Maps symbols to input buffers and records output length."""

    slot: dict[Any, tuple[str, int]]

    @property
    def n_expr(self) -> int: ...


class OpTable(Protocol):
    """Renders primitive operations for one numeric target."""

    prelude_imports: tuple[str, ...]
    elems_per_var: int

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


_ATOM = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\[[0-9]+\])?$")


class ExpressionPrinter(ABC):
    """Walks a SymPy DAG and emits straight line SSA."""

    def __init__(self, ops: OpTable) -> None:
        self.ops = ops
        self.lines: list[str] = []
        self.layout: Layout | None = None
        self._tmp = 0

    @property
    @abstractmethod
    def allocated_dtype(self) -> str:
        """NumPy dtype used when an allocating wrapper is emitted."""

    @property
    @abstractmethod
    def context_name(self) -> str:
        """Short name used in unsupported node errors."""

    def emit(
        self, exprs: list[sp.Expr], layout: Layout, *, allocate: bool
    ) -> list[str]:
        self.layout = layout
        self.lines = []
        self._tmp = 0

        n_out = self.ops.elems_per_var * layout.n_expr
        replacements, reduced = sp.cse(exprs, symbols=sp.numbered_symbols("cse_"))
        for sym, sub in replacements:
            rendered = self.render(sub)
            self.lines.append(f"    {sym.name} = {rendered}")

        if allocate:
            self.lines.append(f"    out = np.empty(({n_out},), {self.allocated_dtype})")
        for i, expr in enumerate(reduced):
            rendered = self.render(expr)
            self.lines.append("    " + self.ops.store("out", i, rendered))
        if allocate:
            self.lines.append("    return out")
        return self.lines

    def render(self, expr: Any) -> str:
        if expr.is_Symbol:
            return self._render_symbol(expr)
        if expr.is_Number:
            return self.ops.const(float(expr))
        if expr.is_Add:
            return self._render_add(expr)
        if expr.is_Mul:
            return self._render_mul(expr)
        if expr.is_Pow:
            return self._render_pow(expr)
        if isinstance(expr, sp.exp):
            return self.ops.exp(self.render(expr.args[0]))
        if isinstance(expr, sp.log):
            return self.ops.log(self.render(expr.args[0]))
        raise NotImplementedError(
            f"{self.context_name} printer: unsupported node "
            f"{type(expr).__name__}: {expr}"
        )

    def _render_symbol(self, sym: Any) -> str:
        assert self.layout is not None
        slot = self.layout.slot.get(sym)
        if slot is None:
            return str(sym.name)
        buf, idx = slot
        return self.ops.load(buf, idx)

    def _render_add(self, expr: Any) -> str:
        result: str | None = None
        for term in expr.args:
            neg = term.could_extract_minus_sign()
            rendered = self.render(-term if neg else term)
            if result is None:
                result = self.ops.neg(rendered) if neg else rendered
            else:
                result = (
                    self.ops.sub(result, rendered)
                    if neg
                    else self.ops.add(result, rendered)
                )
        assert result is not None
        return result

    def _render_mul(self, expr: Any) -> str:
        coeff, rest = expr.as_coeff_Mul()
        num: list[Any] = []
        den: list[Any] = []
        for f in sp.Mul.make_args(rest):
            if f.is_Pow and f.exp.is_Integer and f.exp.is_negative:
                den.append(sp.Pow(f.base, -f.exp))
            else:
                num.append(f)

        result = self._fold_mul(num)
        if den:
            result = self.ops.div(result, self._fold_mul(den))

        if coeff == 1:
            return result
        if coeff == -1:
            return self.ops.neg(result)
        return self.ops.real_scale(result, float(coeff))

    def _fold_mul(self, factors: list[Any]) -> str:
        if not factors:
            return self.ops.const(1.0)
        result = self.render(factors[0])
        for f in factors[1:]:
            result = self.ops.mul(result, self.render(f))
        return result

    def _render_pow(self, expr: Any) -> str:
        base, exp = expr.base, expr.exp
        if exp.is_Integer:
            return self._ipow(self._atom_or_temp(self.render(base)), int(exp))
        if exp.is_Rational and exp.q == 2:
            b = self._atom_or_temp(self.render(base))
            root = self.ops.sqrt(b)
            q = exp.p // 2
            return root if q == 0 else self.ops.mul(self._ipow(b, q), root)
        if exp.is_Number:
            return self.ops.exp(
                self.ops.real_scale(self.ops.log(self.render(base)), float(exp))
            )
        return self.ops.exp(
            self.ops.mul(self.render(exp), self.ops.log(self.render(base)))
        )

    def _ipow(self, base: str, n: int) -> str:
        if n == 0:
            return self.ops.const(1.0)
        if n == 1:
            return base
        if n < 0:
            return self.ops.div(self.ops.const(1.0), self._ipow(base, -n))
        result = base
        for _ in range(n - 1):
            result = self.ops.mul(result, base)
        return result

    def _atom_or_temp(self, expr_str: str) -> str:
        if _ATOM.match(expr_str):
            return expr_str
        name = f"_p{self._tmp}"
        self._tmp += 1
        self.lines.append(f"    {name} = {expr_str}")
        return name
