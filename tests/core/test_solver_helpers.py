"""Branch coverage for DSGESolver static helpers (no model needed)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE.core import ModelParser
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


_STATES = ("k", "z")
_CONTROLS = ("c", "y")


def test_resolve_variable_order_permutes_within_each_block():
    # Membership is the model's, position is the caller's: both blocks come back
    # in the order they were named.
    assert DSGESolver._resolve_variable_order(
        ["z", "k", "y", "c"], _STATES, _CONTROLS
    ) == (("z", "k"), ("y", "c"))


def test_resolve_variable_order_appends_the_minted_lags_to_the_states():
    assert DSGESolver._resolve_variable_order(
        ["z", "k", "c", "y"],
        (*_STATES, "k_lag1"),
        _CONTROLS,
        frozenset({"k_lag1"}),
    ) == (("z", "k", "k_lag1"), ("c", "y"))


def test_resolve_variable_order_errors():
    with pytest.raises(ValueError, match="duplicate"):
        DSGESolver._resolve_variable_order(["z", "z", "k", "c"], _STATES, _CONTROLS)
    with pytest.raises(ValueError, match="Unknown: \\['nope'\\]"):
        DSGESolver._resolve_variable_order(["z", "k", "c", "nope"], _STATES, _CONTROLS)
    with pytest.raises(ValueError, match="Missing: \\['y'\\]"):
        DSGESolver._resolve_variable_order(["z", "k", "c"], _STATES, _CONTROLS)
    # A minted lag is the compiler's to place, so naming one is unknown.
    with pytest.raises(ValueError, match="must not appear"):
        DSGESolver._resolve_variable_order(
            ["z", "k", "c", "y", "k_lag1"],
            (*_STATES, "k_lag1"),
            _CONTROLS,
            frozenset({"k_lag1"}),
        )
    # Right names, wrong blocks: a control cannot be written into the state block.
    with pytest.raises(ValueError, match="must lead with the model's states"):
        DSGESolver._resolve_variable_order(["z", "c", "k", "y"], _STATES, _CONTROLS)


def test_solve_rejects_bad_order():
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        DSGESolver.solve(SimpleNamespace(), None, order=3)


@pytest.fixture
def rbc_compiled(rbc_second_order_test_model_path):
    model, kalman = ModelParser(rbc_second_order_test_model_path).get_all()
    return DSGESolver(model, kalman).compile()


def test_ss_seed_is_written_over_the_declared_variables(rbc_compiled):
    # Declared order is c, k, z; the compiler adds e_st, k_lag1, z_lag1.
    seed = DSGESolver._resolve_ss_seed([1.9, 28.6, 0.0], rbc_compiled)

    idx = rbc_compiled.idx
    assert seed[idx["k_lag1"]] == 28.6  # a lag aux shares its origin's point
    assert seed[idx["z_lag1"]] == 0.0
    assert seed[idx["e_st"]] == 0.0  # a shock state has no steady state
    assert seed[idx["c"]] == 1.9


def test_ss_seed_dict_fills_the_generated_block_from_the_origin(rbc_compiled):
    seed = DSGESolver._resolve_ss_seed({"c": 1.9, "k": 28.6}, rbc_compiled)

    np.testing.assert_allclose(
        seed, DSGESolver._resolve_ss_seed([1.9, 28.6, 0.0], rbc_compiled)
    )
    # An explicit generated entry outranks its origin.
    override = DSGESolver._resolve_ss_seed({"k": 28.6, "k_lag1": 99.0}, rbc_compiled)
    assert override[rbc_compiled.idx["k_lag1"]] == 99.0
    assert override[rbc_compiled.idx["k"]] == 28.6


def test_ss_seed_of_compiled_length_passes_through(rbc_compiled):
    # A previous solve's steady state reads back in unchanged.
    canonical = DSGESolver._resolve_ss_seed([1.9, 28.6, 0.0], rbc_compiled)

    np.testing.assert_allclose(
        DSGESolver._resolve_ss_seed(canonical, rbc_compiled), canonical
    )


def test_ss_seed_of_neither_length_names_both(rbc_compiled):
    with pytest.raises(ValueError, match=r"expected \(3,\).*or \(6,\)"):
        DSGESolver._resolve_ss_seed([0.0, 1.0], rbc_compiled)
