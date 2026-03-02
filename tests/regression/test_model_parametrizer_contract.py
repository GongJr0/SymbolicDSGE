# type: ignore
import sympy as sp
import pytest

from SymbolicDSGE.regression.config import TemplateConfig
from SymbolicDSGE.regression.model_defaults import PySRParams
from SymbolicDSGE.regression.model_parametrizer import (
    ModelParametrizer,
    _normalize_variables,
)


def _make_parametrizer(
    *,
    variable_names: list[str] | None = None,
    config: TemplateConfig | None = None,
) -> ModelParametrizer:
    varnames = variable_names or ["x", "y"]
    cfg = config or TemplateConfig()
    params = PySRParams(precision=32)
    return ModelParametrizer(varnames, params, cfg)


def test_normalize_variables_accepts_supported_sympy_types():
    t = sp.Symbol("t", integer=True)
    f = sp.Function("f")
    g = sp.Function("g")

    out = _normalize_variables(["x", sp.Symbol("y"), f(t), g])
    assert out == ["x", "y", "f", "g"]


def test_normalize_variables_rejects_invalid_type():
    with pytest.raises(ValueError, match="Invalid variable name"):
        _normalize_variables([object()])  # type: ignore[list-item]


def test_add_operator_updates_params_and_primitive_ops():
    p = _make_parametrizer()
    op = p.make_operator(
        lamb=lambda x: x,
        jl_str="foo(x) = x",
        primitive_operation=lambda x: x,
        complexity_bound=2,
    )
    p.add_operator(op)

    assert "foo" in p.primitive_ops
    assert p.params.constraints is not None
    assert p.params.constraints["foo"] == 2
    assert any("foo(" in s for s in p.params.unary_operators or [])


def test_make_template_and_add_template_sets_expression_spec():
    p = _make_parametrizer(config=TemplateConfig())
    tmpl = p.make_template(expr=None)
    assert "f_" in tmpl.combine

    p.add_template(tmpl)
    assert p.params.expression_spec is tmpl


def test_make_and_add_template_with_include_expression_sets_clean_expr():
    t = sp.Symbol("t", integer=True)
    x = sp.Function("x")

    cfg = TemplateConfig(include_expression=True)
    p = _make_parametrizer(variable_names=["x"], config=cfg)
    p.make_and_add_template(expr=x(t) + 1, t=t)

    assert p.clean_expr is not None
    assert "x" in p.clean_expr
    assert p.params.expression_spec is not None


def test_add_built_in_ops_accepts_subset_and_rejects_unknown_name():
    p = _make_parametrizer()
    p.add_built_in_ops(["sqrt", "asinh"])

    assert any("ssqrt" in op for op in (p.params.unary_operators or []))
    assert any("asinh" in op for op in (p.params.unary_operators or []))

    with pytest.raises(ValueError, match="Invalid operator name"):
        p.add_built_in_ops(["does_not_exist"])  # type: ignore[list-item]
