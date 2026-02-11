from .model_defaults import (
    PySRParams,
    CustomOp,
    validate_breaking_settings,
    make_operator_general,
)
from .BuiltInOpContainer import BuiltInOpContainer as bop

from .BaseTemplateFactory import BaseTemplateFactory
from .config import TemplateConfig
from .ConfigValidator import ConfigValidator
import sympy as sp

from pysr import TemplateExpressionSpec

from typing import Callable, Literal, Sequence


def _normalize_variables(
    variable_names: Sequence[str | sp.Symbol | sp.Function],
) -> list[str]:
    out = []
    for var in variable_names:
        if isinstance(var, str):
            out.append(var)
        elif isinstance(var, sp.Symbol):
            out.append(var.name)
        elif isinstance(var, (sp.Function, sp.core.function.UndefinedFunction)):
            out.append(var.func.__name__)
        else:
            raise ValueError(
                f"Invalid variable name '{var}'. Variable names must be strings, sympy Symbols, or sympy Functions."
            )
    return out


class BaseModelParametrizer:
    def __init__(
        self,
        variable_names: Sequence[str | sp.Symbol | sp.Function],
        params: PySRParams,
        config: TemplateConfig,
    ) -> None:
        self._variable_names = _normalize_variables(variable_names)
        self._params = validate_breaking_settings(params)
        self._config = ConfigValidator._validate_config(config)
        self._built_in_ops = {
            "pow": bop.pow(self.config, self.params),
            "sqrt": bop.sqrt(self.params),
            "asinh": bop.asinh(),
        }

    @staticmethod
    def make_operator(
        lamb: Callable[..., float],
        jl_str: str,
        complexity_bound: tuple[int, int] | int | None = None,
    ) -> CustomOp:
        return make_operator_general(lamb, jl_str, complexity_bound)

    def add_operator(self, op: CustomOp) -> None:
        """
        Add a custom operator to the PySRParams.
        :param op: The CustomOp to add.
        :type op: CustomOp
        """
        self._params.add_operator(op)

    def make_and_add_operator(
        self,
        lamb: Callable[..., float],
        jl_str: str,
        complexity_bound: tuple[int, int] | int | None = None,
    ) -> None:
        """
        Create a CustomOp and add it to the PySRParams.
        :param lamb: A Python function that implements the operator.
        :type lamb: Callable[..., float]
        :param jl_str: A string defining the operator in Julia syntax for PySR.
        :type jl_str: str
        """
        op = self.make_operator(lamb, jl_str, complexity_bound)
        self.add_operator(op)

    def make_template(
        self, expr: sp.Expr | None, t: sp.Symbol | None = None
    ) -> TemplateExpressionSpec:
        """
        Create a PySR TemplateExpressionSpec using the BaseTemplateFactory.
        :param variable_names: A sequence of variable names or sympy symbols/functions to include in the template.
        :type variable_names: Sequence[str | sp.Symbol | sp.Function]
        :param expr: An optional sympy expression to include in the template. Required if config.include_expression is True.
        :type expr: sp.Expr | None
        :param t: A sympy symbol representing time, used for template generation and expression processing. Default is sp.Symbol("t", integer=True).
        :type t: sp.Symbol
        :returns: A TemplateExpressionSpec object containing the generated template.
        :rtype: TemplateExpressionSpec
        """
        if t is None:
            t = sp.Symbol("t", integer=True)

        config = self.config
        variable_names = self.variable_names

        factory = BaseTemplateFactory(config, variable_names, expr, t)
        return factory.get_template()

    def add_template(self, template: TemplateExpressionSpec) -> None:
        """
        Add a PySR TemplateExpressionSpec to the PySRParams.
        :param template: The TemplateExpressionSpec to add.
        :type template: TemplateExpressionSpec
        """
        self._params.add_template(template)

    def make_and_add_template(
        self,
        expr: sp.Expr | None,
        t: sp.Symbol | None = None,
    ) -> None:
        """
        Create a PySR TemplateExpressionSpec using the BaseTemplateFactory and add it to the PySRParams.
        :param variable_names: A sequence of variable names or sympy symbols/functions to include in the template.
        :type variable_names: Sequence[str | sp.Symbol | sp.Function]
        :param expr: An optional sympy expression to include in the template. Required if config.include_expression is True.
        :type expr: sp.Expr | None
        :param t: A sympy symbol representing time, used for template generation and expression processing. Default is sp.Symbol("t", integer=True).
        :type t: sp.Symbol
        """
        template = self.make_template(expr, t)
        self.add_template(template)

    def add_built_in_ops(self, ops: list[str] | None = None) -> None:
        """
        Add built-in operators to the PySRParams based on the provided list of operator names.
        :param ops: A list of operator names to add. Valid names are "pow", "sqrt", and "asinh".
        :type ops: list[Literal["pow", "sqrt", "asinh"]]
        """
        if ops is None:
            ops = list(self.built_in_ops.keys())
        for op_name in ops:
            if op_name not in self._built_in_ops:
                raise ValueError(
                    f"Invalid operator name '{op_name}'. Valid options are: {list(self._built_in_ops.keys())}"
                )
            self.add_operator(self._built_in_ops[op_name])

    @property
    def variable_names(self) -> list[str]:
        return self._variable_names

    @property
    def params(self) -> PySRParams:
        return self._params

    @property
    def config(self) -> TemplateConfig:
        return self._config

    @property
    def built_in_ops(self) -> dict[str, CustomOp]:
        return self._built_in_ops
