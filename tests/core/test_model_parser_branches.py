"""Branch coverage for ModelParser helpers."""

from __future__ import annotations

import pytest
import sympy as sp

from SymbolicDSGE.core.model_parser import (
    _reject_unknown_keys,
    _coerce_variable_data,
    _load_yaml,
    _sympy_parsers,
)


def test_coerce_variable_data_branches():
    # list form
    names, data = _coerce_variable_data({"variables": ["a", "b"]})
    assert names == ["a", "b"] and data == {"a": {}, "b": {}}
    # mapping with None and dict specs
    names, data = _coerce_variable_data(
        {"variables": {"a": None, "b": {"ss_seed": "b_ss"}}}
    )
    assert data["a"] == {} and data["b"] == {"ss_seed": "b_ss"}
    # unsupported metadata key
    with pytest.raises(ValueError, match="unsupported metadata keys"):
        _coerce_variable_data({"variables": {"a": {"bogus": 1}}})
    # spec neither mapping nor null
    with pytest.raises(TypeError, match="mapping or null"):
        _coerce_variable_data({"variables": {"a": 5}})
    # variables neither list nor mapping
    with pytest.raises(TypeError, match="list or a mapping"):
        _coerce_variable_data({"variables": 5})


def test_reject_unknown_keys():
    # non-dict is a no-op
    _reject_unknown_keys(["not", "a", "dict"], frozenset(), "where")
    # subset is fine
    _reject_unknown_keys({"a": 1}, frozenset({"a", "b"}), "where")
    with pytest.raises(ValueError, match="Unknown field"):
        _reject_unknown_keys({"z": 1}, frozenset({"a"}), "block")


def test_load_yaml_rejects_non_mapping_root(tmp_path):
    p = tmp_path / "root_list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(TypeError, match="root must be a mapping"):
        _load_yaml(p)


def test_sympy_parsers_relational_and_expr_guards():
    locals_ = {"x": sp.Symbol("x"), "y": sp.Symbol("y")}
    get_expr, get_relational, _get_eq = _sympy_parsers(locals_, {})

    rel = get_relational("x > y")
    assert rel.rel_op == ">"
    # a plain expression is not a Relational
    with pytest.raises(TypeError, match="not a valid SymPy Relational"):
        get_relational("x + y")
    # a relational is not an Expr
    with pytest.raises(TypeError, match="not a valid SymPy Expr"):
        get_expr("x > y")
