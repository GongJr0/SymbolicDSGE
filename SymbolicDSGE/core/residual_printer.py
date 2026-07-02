"""Residual codegen for the native perturbation preproc (issue #248).

Emits an allocation-free, straight-line numba function that evaluates the model
residual in a chosen arithmetic type. This is the c128 (first-order / Klein
complex-step) backend; the bicomplex backend for second order plugs into the same
base printer via a different :class:`OpTable`.

Design: the base :class:`ResidualPrinter` owns everything type-agnostic -- the
DAG walk, common-subexpression elimination, the buffer layout, and the ``Pow``
dispatch (integer -> repeated multiply, real -> ``exp(p*log)``, half-integer ->
``ipow * sqrt``). All type-specific rendering lives in the ``OpTable``, which for
c128 is ten one-liners over numba's native ``complex128``.

Two invariants the c128 backend depends on:
  * constants render zero-imaginary (``complex(v, 0.0)``) so they inject no
    spurious complex-step derivative;
  * integer powers use repeated multiplication, never ``**`` -- numba complex
    ``z**n`` routes through ``exp(n*log z)``, whose branch cut corrupts the
    complex-step imaginary part for negative real bases (deviation variables).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

import sympy as sp
from numba import cfunc, njit, types
from sympy import Symbol


class OpTable(Protocol):
    """Renders each primitive to a target-language expression string."""

    prelude_imports: tuple[str, ...]

    def const(self, v: float) -> str: ...
    def add(self, a: str, b: str) -> str: ...
    def sub(self, a: str, b: str) -> str: ...
    def mul(self, a: str, b: str) -> str: ...
    def div(self, a: str, b: str) -> str: ...
    def neg(self, a: str) -> str: ...
    def real_scale(self, a: str, s: float) -> str: ...
    def exp(self, a: str) -> str: ...
    def log(self, a: str) -> str: ...
    def sqrt(self, a: str) -> str: ...


class C128Ops(OpTable):
    """numba ``complex128`` backend: native operators + ``cmath`` transcendentals."""

    prelude_imports = ("import cmath",)

    def const(self, v: float) -> str:
        return f"complex({float(v)!r}, 0.0)"

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


@dataclass
class ResidualLayout:
    """Maps residual symbols to native buffer slots and records dimensions."""

    slot: dict[Any, tuple[str, int]]  # sympy Symbol -> (buffer_name, index)
    n_var: int
    n_par: int
    n_eq: int

    @classmethod
    def from_compiled(cls, compiled: Any) -> ResidualLayout:
        """Build the layout for a ``CompiledModel``'s ``objective_eqs``.

        Symbols are ``fwd_<name>`` (t+1), ``cur_<name>`` (t), and the calibration
        parameters, matching ``solver._build`` and ``equations(fwd, cur, par)``.
        """
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


_ATOM = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\[[0-9]+\])?$")


class ResidualPrinter:
    """Walks the sympy residual DAG and emits straight-line SSA via an OpTable."""

    def __init__(self, ops: OpTable) -> None:
        self.ops = ops
        self.lines: list[str] = []
        self.layout: ResidualLayout | None = None
        self._tmp = 0

    # -- public emit ---------------------------------------------------------
    def emit(
        self, exprs: list[sp.Expr], layout: ResidualLayout, *, allocate: bool
    ) -> list[str]:
        """Return the indented body lines. ``allocate`` -> declare+return ``out``
        (njit form); otherwise ``out`` is a caller-provided buffer (cfunc form)."""
        self.layout = layout
        self.lines = []
        self._tmp = 0

        replacements, reduced = sp.cse(exprs, symbols=sp.numbered_symbols("cse_"))
        for sym, sub in replacements:
            rendered = self.render(sub)
            self.lines.append(f"    {sym.name} = {rendered}")

        if allocate:
            self.lines.append(f"    out = np.empty(({layout.n_eq},), np.complex128)")
        for i, expr in enumerate(reduced):
            rendered = self.render(expr)
            self.lines.append(f"    out[{i}] = {rendered}")
        if allocate:
            self.lines.append("    return out")
        return self.lines

    # -- dispatch ------------------------------------------------------------
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
            f"residual printer: unsupported node {type(expr).__name__}: {expr}"
        )

    # -- nodes ---------------------------------------------------------------
    def _render_symbol(self, sym: Any) -> str:
        assert self.layout is not None
        slot = self.layout.slot.get(sym)
        if slot is None:
            return str(sym.name)  # a cse temp (cse_0, ...)
        buf, idx = slot
        return f"{buf}[{idx}]"

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
        if exp.is_Number:  # general real exponent, positive base
            return self.ops.exp(
                self.ops.real_scale(self.ops.log(self.render(base)), float(exp))
            )
        # symbolic exponent: base**e = exp(e * log(base))
        return self.ops.exp(
            self.ops.mul(self.render(exp), self.ops.log(self.render(base)))
        )

    # -- helpers -------------------------------------------------------------
    def _ipow(self, base: str, n: int) -> str:
        """Integer power via repeated multiplication (log-free, any-sign base)."""
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
        """Bind a compound base to a temp so ipow does not duplicate its work."""
        if _ATOM.match(expr_str):
            return expr_str
        name = f"_p{self._tmp}"
        self._tmp += 1
        self.lines.append(f"    {name} = {expr_str}")
        return name


# ---------------------------------------------------------------------------
# Builders: sympy residual -> compiled numba callable.
# ---------------------------------------------------------------------------
def build_njit(
    exprs: list[sp.Expr], layout: ResidualLayout, ops: OpTable | None = None
) -> Any:
    """``_residual(fwd, cur, par) -> out`` as an njit vector function.

    Signature-compatible with ``CompiledModel.construct_objective_vector_func``,
    so it drops straight into ``klein._approximate_system_numeric``.
    """
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
    exec(src, ns)  # noqa: S102 - emitted from our own printer, not user input
    return njit(ns["_residual"])


def build_cfunc(
    exprs: list[sp.Expr], layout: ResidualLayout, ops: OpTable | None = None
) -> Any:
    """``void(fwd*, cur*, par*, out*)`` numba @cfunc; ``.address`` feeds the driver."""
    table: OpTable = C128Ops() if ops is None else ops
    body = ResidualPrinter(table).emit(exprs, layout, allocate=False)
    preamble = [
        f"    fwd = carray(fwd_ptr, ({layout.n_var},))",
        f"    cur = carray(cur_ptr, ({layout.n_var},))",
        f"    par = carray(par_ptr, ({layout.n_par},))",
        f"    out = carray(out_ptr, ({layout.n_eq},))",
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
    exec(src, ns)  # noqa: S102 - emitted from our own printer, not user input
    sig = types.void(
        types.CPointer(types.complex128),
        types.CPointer(types.complex128),
        types.CPointer(types.complex128),
        types.CPointer(types.complex128),
    )
    return cfunc(sig)(ns["_residual_cf"])
