import sympy as sp
from sympy.printing.str import StrPrinter

from typing import Any


class JFloat(sp.Function):
    nargs = 1


class JuliaTypedPrinter(StrPrinter):
    """Printer that emits Julia-ish code and renders JFloat(x) as Float{prec}(x)."""

    def __init__(self, prec: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.prec = prec

    def _print_JFloat(self, expr: sp.Expr) -> str:
        # expr.args[0] is the numeric literal
        inner = expr.args[0]
        # Print the literal in a Julia-friendly way
        if isinstance(inner, sp.Float):
            # Use full precision string SymPy provides
            lit = sp.sstr(inner)
        elif isinstance(inner, sp.Rational):
            # Prefer decimal? Either is fine.
            # Emit as "p//q" to avoid Julia parsing as Float64.
            lit = f"{inner.p}//{inner.q}"
        else:
            lit = sp.sstr(inner)

        return f"Float{self.prec}({lit})"

    def _print_Pow(self, expr: sp.Expr) -> str:  # pyright: ignore
        # Julia uses ^, SymPy prints ** by default
        base, exp = expr.as_base_exp()
        return f"({self._print(base)})^({self._print(exp)})"

    def _print_Mul(self, expr: sp.Expr) -> str:  # pyright: ignore
        # SymPy may insert rationals like 1/2*x; keep explicit *
        return "*".join(self._print(a) for a in expr.args)

    def _print_Add(self, expr: sp.Expr) -> str:  # pyright: ignore
        return " + ".join(self._print(a) for a in expr.args)


def _needs_float_wrap(expr: sp.Expr) -> bool:
    """Return True for numeric atoms that should be typed as Float{prec}."""
    # SymPy Float => definitely wrap
    if isinstance(expr, sp.Float):
        return True
    # Rational => usually wrap to avoid Float64 promotion in Julia (esp in divisions/exponents)
    if isinstance(expr, sp.Rational):
        return True

    return False


def wrap_numeric_literals(expr: sp.Expr) -> sp.Expr:
    """Wrap numeric literals in JFloat(...) where appropriate."""

    def repl(e: sp.Expr) -> sp.Expr:
        if e.is_Number and _needs_float_wrap(e):
            return JFloat(e)  # pyright: ignore
        return e

    # Replace numeric atoms bottom-up
    return expr.replace(
        lambda e: e.is_Number and _needs_float_wrap(e), repl
    )  # pyright: ignore


def sympy_to_julia_typed(expr: sp.Expr, prec: int) -> str:
    expr2 = wrap_numeric_literals(expr)
    out: str = JuliaTypedPrinter(prec=prec).doprint(expr2)
    return out


def _spec_ready_expr(expr: sp.Expr, t: sp.Symbol) -> sp.Expr:
    """
    Convert a sympy expression into a string. Strip functions of time and convert to variables for template inclusion.
    :param expr: A sympy expression to convert.
    :type expr: sp.Expr
    :return: A string representation of the expression, ready for template inclusion.
    :rtype: str
    """
    subs_dict = {
        f: sp.Symbol(  # pyright: ignore # LSP looks for __call__; SymPy apparently "doesn't implement it" on symbolic functions.
            f.func.__name__
        )
        for f in expr.atoms(sp.Function)
        if t in f.free_symbols
    }
    return expr.subs(subs_dict)  # pyright: ignore


def get_expr(expr: sp.Expr, t: sp.Symbol, prec: int) -> tuple[str, str]:
    """
    Convert a sympy expression into two string forms:
        1. A "clean" version with time dependent functions swapped out for variables.
        2. A "template-ready" version where float literals are wrapped with the float type appropriate for the model precision.

    :param expr: A sympy expression to convert.
    :type expr: sp.expr
    :param t: A sympy symbol representing time, used to identify time-dependent functions in the expression.
    :type t: sp.Symbol
    :param prec: The precision (in bits) to use for numeric literals in the template-ready version (e.g., 32 for Float32).
    :type prec: int
    :return: A tuple containing the clean expression string and the template-ready expression string.
    :rtype: tuple[str, str]
    """
    clean_expr = _spec_ready_expr(expr, t)
    template_ready_str = JuliaTypedPrinter(prec=prec).doprint(clean_expr)
    clean_str = str(clean_expr)
    return clean_str, template_ready_str
