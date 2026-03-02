# type: ignore
import sympy as sp

from SymbolicDSGE.regression.expr_to_string import (
    JFloat,
    _needs_float_wrap,
    _spec_ready_expr,
    get_expr,
    sympy_to_julia_typed,
    wrap_numeric_literals,
)


def test_needs_float_wrap_for_float_and_rational():
    assert _needs_float_wrap(sp.Float(0.25))
    assert _needs_float_wrap(sp.Rational(1, 3))
    assert _needs_float_wrap(sp.Integer(2))


def test_wrap_numeric_literals_inserts_jfloat_nodes():
    x = sp.Symbol("x")
    expr = x + sp.Float(0.25) * x + sp.Rational(1, 3)
    wrapped = wrap_numeric_literals(expr)
    assert len(wrapped.atoms(JFloat)) == 2


def test_sympy_to_julia_typed_formats_pow_and_numeric_literals():
    x = sp.Symbol("x")
    expr = x**2 + sp.Rational(1, 2) * x
    out = sympy_to_julia_typed(expr, prec=32)
    assert "^" in out
    assert "Float32" in out
    assert "1//2" in out


def test_spec_ready_expr_replaces_time_functions_with_plain_symbols():
    t = sp.Symbol("t", integer=True)
    x = sp.Function("x")
    y = sp.Function("y")
    expr = x(t) + y(t - 1) + sp.Symbol("z")

    out = _spec_ready_expr(expr, t)
    assert not out.atoms(sp.Function)
    assert sp.Symbol("x") in out.free_symbols
    assert sp.Symbol("y") in out.free_symbols
    assert sp.Symbol("z") in out.free_symbols


def test_get_expr_returns_clean_and_template_strings():
    t = sp.Symbol("t", integer=True)
    x = sp.Function("x")
    expr = x(t) + sp.Rational(1, 2)

    clean, template_ready = get_expr(expr, t=t, prec=64)
    assert "x" in clean
    assert "Float64" in template_ready
