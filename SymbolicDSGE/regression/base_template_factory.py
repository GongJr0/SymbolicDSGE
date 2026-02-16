from pysr import TemplateExpressionSpec
import sympy as sp
from sympy.printing.str import StrPrinter

from .config import TemplateConfig
from .config_validator import ConfigValidator
from .interaction_generator import InteractionGenerator
from .expr_to_string import get_expr

from typing import Sequence, Any


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
            clean_expr, expr_str = get_expr(self.expr, self.t, prec)
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
