# type: ignore
import sympy as sp
import pytest

from SymbolicDSGE.regression.config import TemplateConfig
from SymbolicDSGE.regression.interaction_generator import InteractionGenerator


def test_normalize_names_accepts_strings_symbols_and_functions():
    t = sp.Symbol("t", integer=True)
    x = sp.Symbol("x")
    y = sp.Function("y")(t)

    out = InteractionGenerator._normalize_names(["a", x, y])
    assert out == ["a", "x", "y"]


def test_normalize_names_rejects_invalid_type():
    with pytest.raises(ValueError, match="must be of type"):
        InteractionGenerator._normalize_names([1])  # type: ignore[list-item]


def test_resolve_variable_space_expr_requires_expr():
    cfg = TemplateConfig(variable_space="expr")
    with pytest.raises(ValueError, match="Expression must be provided"):
        InteractionGenerator._resolve_variable_space(cfg, ["x"], None, sp.Symbol("t"))


def test_resolve_variable_space_expr_requires_time_symbol():
    cfg = TemplateConfig(variable_space="expr")
    t = sp.Symbol("t", integer=True)
    x = sp.Function("x")
    expr = x(t)
    with pytest.raises(ValueError, match="Please provide the time symbol"):
        InteractionGenerator._resolve_variable_space(cfg, ["x"], expr, None)


def test_resolve_variable_space_expr_discovers_function_names():
    cfg = TemplateConfig(variable_space="expr")
    t = sp.Symbol("t", integer=True)
    x = sp.Function("x")
    y = sp.Function("y")
    expr = x(t) + y(t - 1)

    discovered = InteractionGenerator._resolve_variable_space(cfg, None, expr, t)
    assert set(discovered) == {"x", "y"}


def test_resolve_variable_space_expr_rejects_missing_declared_variables():
    cfg = TemplateConfig(variable_space="expr")
    t = sp.Symbol("t", integer=True)
    x = sp.Function("x")
    y = sp.Function("y")
    expr = x(t) + y(t)

    with pytest.raises(ValueError, match="must be present in the variable names list"):
        InteractionGenerator._resolve_variable_space(cfg, ["x"], expr, t)


def test_get_interactions_without_powers_uses_combinations():
    cfg = TemplateConfig(
        interaction_only=False,
        poly_interaction_order=2,
        powers_in_interactions=False,
    )

    out = InteractionGenerator.get_interactions(
        cfg,
        varnames=["x", "y"],
        expr=None,
        t=sp.Symbol("t", integer=True),
    )
    assert ("x",) in out and ("y",) in out and ("x", "y") in out
    assert ("x", "x") not in out and ("y", "y") not in out


def test_get_interactions_with_powers_uses_combinations_with_replacement():
    cfg = TemplateConfig(
        interaction_only=True,
        poly_interaction_order=2,
        powers_in_interactions=True,
    )

    out = InteractionGenerator.get_interactions(
        cfg,
        varnames=["x", "y"],
        expr=None,
        t=sp.Symbol("t", integer=True),
    )
    assert set(out) == {("x", "x"), ("x", "y"), ("y", "y")}


def test_get_interactions_rejects_unbounded_variable_space():
    cfg = TemplateConfig(variable_space=None)
    with pytest.raises(ValueError, match="Variable space is unbounded"):
        InteractionGenerator.get_interactions(
            cfg,
            varnames=None,  # type: ignore[arg-type]
            expr=None,
            t=sp.Symbol("t", integer=True),
        )
