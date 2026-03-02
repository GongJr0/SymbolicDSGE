# type: ignore
import sympy as sp

from SymbolicDSGE.regression.config import ConstantFiltering, TemplateConfig
from SymbolicDSGE.regression.constant_handler import ConstantHandler


def test_keep_strategy_returns_original_expression():
    expr = sp.Symbol("x") + sp.Integer(2)
    handler = ConstantHandler(TemplateConfig(constant_filtering="keep"))
    out = handler.get_handled_exprs([expr])[0]
    assert out == expr


def test_disqualify_strategy_marks_expression_with_constants_as_nan():
    expr = sp.Symbol("x") + sp.Integer(2)
    handler = ConstantHandler(TemplateConfig(constant_filtering="disqualify"))
    out = handler.get_handled_exprs([expr])[0]
    assert out is sp.S.NaN


def test_disqualify_strategy_keeps_expression_without_constants():
    expr = sp.Symbol("x") + sp.Symbol("y")
    handler = ConstantHandler(TemplateConfig(constant_filtering="disqualify"))
    out = handler.get_handled_exprs([expr])[0]
    assert out == expr


def test_parametrize_strategy_replaces_constants_with_dummy_parameters():
    x, y = sp.Symbol("x"), sp.Symbol("y")
    expr = x + y + sp.Integer(2) + sp.sqrt(2)
    handler = ConstantHandler(TemplateConfig(constant_filtering="parametrize"))
    out = handler.get_handled_exprs([expr])[0]

    dummies = out.atoms(sp.Dummy)
    assert len(dummies) == 2
    assert any(d.name == "C_1" for d in dummies)
    assert any(d.name == "C_2" for d in dummies)
    assert x in out.free_symbols and y in out.free_symbols


def test_strip_strategy_removes_free_constant_terms():
    x, y = sp.Symbol("x"), sp.Symbol("y")
    expr = x + y + sp.Integer(2)
    handler = ConstantHandler(TemplateConfig(constant_filtering="strip"))
    out = handler.get_handled_exprs([expr])[0]
    assert sp.simplify(out - (x + y)) == 0


def test_strategy_property_returns_enum_value():
    handler = ConstantHandler(TemplateConfig(constant_filtering="keep"))
    assert handler.strategy == ConstantFiltering.KEEP
