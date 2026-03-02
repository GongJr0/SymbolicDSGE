# type: ignore
import sympy as sp
import pytest

from SymbolicDSGE.regression.config import TemplateConfig
from SymbolicDSGE.regression.template_factory import (
    MissingExpressionError,
    TemplateFactory,
)


def test_missing_expression_error_when_include_expression_true():
    cfg = TemplateConfig(include_expression=True)
    with pytest.raises(MissingExpressionError):
        TemplateFactory(cfg, variable_names=["x"], expr=None)


def test_get_template_free_func_uses_single_function_over_variables():
    cfg = TemplateConfig(
        include_expression=False,
        interaction_only=True,
        interaction_form="func",
    )
    fac = TemplateFactory(cfg, variable_names=["x", "y"])
    clean, spec = fac.get_template(hessian_restriction="free", prec=32)

    assert clean is None
    assert spec.expressions == ["f_1"]
    assert "f_1(x, y)" in spec.combine


def test_get_template_free_prod_uses_product_interactions():
    cfg = TemplateConfig(
        include_expression=False,
        interaction_only=True,
        poly_interaction_order=2,
        interaction_form="prod",
        powers_in_interactions=False,
    )
    fac = TemplateFactory(cfg, variable_names=["x", "y"])
    _, spec = fac.get_template(hessian_restriction="free", prec=32)

    assert spec.expressions == ["f_1"]
    assert "f_1(x*y)" in spec.combine


def test_get_template_diag_creates_function_per_interaction():
    cfg = TemplateConfig(
        include_expression=False,
        interaction_only=True,
        poly_interaction_order=2,
        powers_in_interactions=True,
        interaction_form="func",
    )
    fac = TemplateFactory(cfg, variable_names=["x", "y"])
    _, spec = fac.get_template(hessian_restriction="diag", prec=32)

    # interaction_only=True with degree-2 and powers_in_interactions=True:
    # (x*x), (x*y), (y*y)
    assert len(spec.expressions) == 3
    assert "f_1(" in spec.combine and "f_3(" in spec.combine


def test_get_template_with_include_expression_returns_clean_expression():
    t = sp.Symbol("t", integer=True)
    x = sp.Function("x")
    expr = x(t) + sp.Rational(1, 2)

    cfg = TemplateConfig(
        include_expression=True,
        interaction_only=True,
        interaction_form="func",
    )
    fac = TemplateFactory(cfg, variable_names=["x"], expr=expr, t=t)
    clean, spec = fac.get_template(hessian_restriction="free", prec=64)

    assert clean is not None and "x" in clean
    assert "Float64" in spec.combine
