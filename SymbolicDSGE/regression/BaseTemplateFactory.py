from pysr import TemplateExpressionSpec
import sympy as sp
from sympy.printing.str import StrPrinter

from .config import TemplateConfig
from .ConfigValidator import ConfigValidator
from .InteractionGenerator import InteractionGenerator

from typing import Sequence, Any


class JFloat(sp.Function):
    nargs = 1


def _needs_float_wrap(expr: sp.Expr) -> bool:
    """Return True for numeric atoms that should be typed as Float{prec}."""
    # SymPy Float => definitely wrap
    if isinstance(expr, sp.Float):
        return True
    # Rational => usually wrap to avoid Float64 promotion in Julia (esp in divisions/exponents)
    if isinstance(expr, sp.Rational):
        return True
    # Integer: usually safe to keep as integer. (Don't wrap by default.)
    return False


def wrap_numeric_literals(expr: sp.Expr) -> sp.Expr:
    """Wrap numeric literals in JFloat(...) where appropriate."""

    def repl(e: sp.Expr) -> sp.Expr:
        if e.is_Number and _needs_float_wrap(e):
            return JFloat(e)
        return e

    # Replace numeric atoms bottom-up
    return expr.replace(lambda e: e.is_Number and _needs_float_wrap(e), repl)


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
            # Prefer decimal? Either is fine; I'd keep rational to be exact
            # but wrapped: Float32(1//4) is fine too.
            # We'll emit as "p//q" to avoid Julia parsing as Float64.
            lit = f"{inner.p}//{inner.q}"
        else:
            lit = sp.sstr(inner)

        return f"Float{self.prec}({lit})"

    def _print_Pow(self, expr: sp.Expr) -> str:
        # Julia uses ^, SymPy prints ** by default
        base, exp = expr.as_base_exp()
        return f"({self._print(base)})^({self._print(exp)})"

    def _print_Mul(self, expr: sp.Expr) -> str:
        # SymPy may insert rationals like 1/2*x; keep explicit *
        return "*".join(self._print(a) for a in expr.args)

    def _print_Add(self, expr: sp.Expr) -> str:
        return " + ".join(self._print(a) for a in expr.args)


def sympy_to_julia_typed(expr: sp.Expr, prec: int) -> str:
    expr2 = wrap_numeric_literals(expr)
    out: str = JuliaTypedPrinter(prec=prec).doprint(expr2)
    return out


def _spec_ready_expr(expr: sp.Expr, t: sp.Symbol, prec: int) -> tuple[str, str]:
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
    clean_expr = expr.subs(subs_dict)
    return str(clean_expr), sympy_to_julia_typed(clean_expr, prec)


class MissingExpressionError(Exception):
    """Raised when include_expression is True but no expression is provided."""

    def __init__(self) -> None:
        super().__init__(
            (
                "include_expression is True but no expression was provided. "
                "Please provide an expression or set include_expression to False."
            )
        )


class BaseTemplateFactory:
    def __init__(
        self,
        config: TemplateConfig,
        variable_names: Sequence[str],
        expr: sp.Expr | None = None,
        t: sp.Symbol = sp.Symbol("t", integer=True),
    ) -> None:
        """
        Base class for generating complete model specifications via:
        - Templating: Confining model search to predetermined functional forms
        - Feature Generation: Handling interactions and transformations

        :param config: Configuration for template generation and feature generation.
        :type config: TemplateConfig
        :param X: Matrix of state variables
        :type X: NDF
        :param variable_names: List of variable names corresponding to columns in X.
        :type variable_names: list[str]
        :param expr: Sympy expression to include in the template if specified by config.include_expression.
        :type expr: sp.Expr | None
        :param t: Symbol representing time, used for template generation and expression processing.
        :type t: sp.Symbol
        :raises MissingExpressionError: If config.include_expression is True but no expression is provided.
        :returns: None
        """

        self._config = ConfigValidator._validate_config(config)
        self._variable_names = list(variable_names)
        self._t = t

        if self.config.include_expression and expr is None:
            raise MissingExpressionError()
        self._expr = expr

    def get_template(self, prec: int) -> tuple[str | None, TemplateExpressionSpec]:
        """
        Generate a PySR TemplateExpressionSpec based on the provided configuration and expression.
        :return: A TemplateExpressionSpec object containing the generated template.
        :rtype: TemplateExpressionSpec
        """
        config = self.config
        varnames = self.variable_names
        interactions = InteractionGenerator.get_interactions(
            config, varnames, self.expr, self.t
        )

        term_join_str = (
            "*" if config.interaction_form == "prod" else ", "
        )  # else == 'func' (pre-validated )

        func_names = []
        template_components = []
        clean_expr: str | None = None

        if (config.include_expression) and (self.expr is not None):
            clean_expr, expr_str = _spec_ready_expr(self.expr, self.t, prec)
            template_components.append(expr_str)

        for n, term in enumerate(interactions):
            func_name = f"f_{n+1}"
            func_names.append(func_name)
            inner = term_join_str.join(term)
            template_components.append(f"{func_name}({inner})")

        template_str = " + ".join(template_components)

        return clean_expr, TemplateExpressionSpec(
            combine=template_str,
            expressions=func_names,
            variable_names=varnames,
        )

    @property
    def config(self) -> TemplateConfig:
        return self._config

    @property
    def expr(self) -> sp.Expr | None:
        return self._expr

    @property
    def variable_names(self) -> list[str]:
        return self._variable_names

    @property
    def t(self) -> sp.Symbol:
        return self._t
