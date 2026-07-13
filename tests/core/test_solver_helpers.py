"""Branch coverage for DSGESolver static helpers (no model needed)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import sympy as sp

from SymbolicDSGE.core.solver import DSGESolver

t = sp.Symbol("t")
c = sp.Function("c")
a = sp.Symbol("a")


def test_coerce_variable_name_branches():
    assert DSGESolver._coerce_variable_name("x") == "x"
    # UndefinedFunction class -> __name__ branch
    assert DSGESolver._coerce_variable_name(c) == "c"
    # Symbol -> .name branch
    assert DSGESolver._coerce_variable_name(sp.Symbol("y")) == "y"
    # applied call c(t) -> .func.__name__ branch
    assert DSGESolver._coerce_variable_name(c(t)) == "c"
    # nothing matches -> str() fallback
    obj = SimpleNamespace()
    assert DSGESolver._coerce_variable_name(obj) == str(obj)


def test_function_call_offset_branches():
    declared = {"c"}
    # not a function-like expression -> None
    assert DSGESolver._function_call_offset(5, declared, t) is None
    # a declared call with an integer offset
    assert DSGESolver._function_call_offset(c(t + 1), declared, t) == ("c", 1)
    assert DSGESolver._function_call_offset(c(t - 1), declared, t) == ("c", -1)
    # arg without t -> None
    assert DSGESolver._function_call_offset(c(sp.Integer(2)), declared, t) is None
    # non-integer offset -> None
    assert DSGESolver._function_call_offset(c(t + a), declared, t) is None
    # name not declared -> None
    assert DSGESolver._function_call_offset(c(t + 1), {"k"}, t) is None


_KW = dict(
    declared_names=("z", "k", "c"),
    exo_state_names=("z",),
    endo_state_names=("k",),
    n_exog=1,
    n_state=2,
)


def test_resolve_variable_order_valid():
    assert DSGESolver._resolve_variable_order(["z", "k", "c"], **_KW) == ("z", "k", "c")


def test_resolve_variable_order_errors():
    with pytest.raises(ValueError, match="do not exist"):
        DSGESolver._resolve_variable_order(["z", "k", "nope"], **_KW)
    with pytest.raises(ValueError, match="duplicate"):
        DSGESolver._resolve_variable_order(["z", "z", "k"], **_KW)
    with pytest.raises(ValueError, match="every model variable"):
        DSGESolver._resolve_variable_order(["z", "k"], **_KW)
    with pytest.raises(ValueError, match="shocked states"):
        DSGESolver._resolve_variable_order(["k", "z", "c"], **_KW)
    with pytest.raises(ValueError, match="n_state variables to be states"):
        DSGESolver._resolve_variable_order(["z", "c", "k"], **_KW)


def test_solve_rejects_bad_order():
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        DSGESolver.solve(SimpleNamespace(), None, order=3)
