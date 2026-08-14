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


@pytest.fixture
def deep_compiled(tmp_path):
    """MODELS/test.yaml with u lagged three deep, so the compiler mints a chain."""
    import yaml

    data = yaml.safe_load(open("MODELS/test.yaml", encoding="utf-8"))
    data["equations"]["model"]["u_process"] = "u(t) = rho_u*u(t-3) + e_u"
    path = tmp_path / "deep_test.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    model, kalman = ModelParser(path).get_all()
    return DSGESolver(model, kalman).compile()


def test_ss_seed_is_written_over_the_declared_variables(rbc_compiled):
    # Declared order is c, k, z, which is not the compiled order k, z, c.
    seed = DSGESolver._resolve_ss_seed([1.9, 28.6, 0.5], rbc_compiled)

    idx = rbc_compiled.idx
    assert seed[idx["c"]] == 1.9
    assert seed[idx["k"]] == 28.6
    assert seed[idx["z"]] == 0.5


def test_ss_seed_falls_to_the_config_then_to_zero(rbc_compiled):
    # Per variable: what the dict names wins, what it does not keeps the seed
    # the config declares, and a variable the config skips starts at zero.
    configured = DSGESolver._resolve_ss_seed(None, rbc_compiled)
    seed = DSGESolver._resolve_ss_seed({"k": 99.0}, rbc_compiled)

    idx = rbc_compiled.idx
    assert seed[idx["k"]] == 99.0
    assert seed[idx["c"]] == configured[idx["c"]] != 0.0
    assert seed[idx["z"]] == configured[idx["z"]] == 0.0


def test_ss_seed_of_the_wrong_length_names_the_declaration_order(rbc_compiled):
    with pytest.raises(ValueError, match=r"expected \(3,\).*\['c', 'k', 'z'\]"):
        DSGESolver._resolve_ss_seed([0.0, 1.0], rbc_compiled)


def test_ss_seed_rejects_an_unknown_variable(rbc_compiled):
    with pytest.raises(ValueError, match=r"does not have: \['ghost'\]"):
        DSGESolver._resolve_ss_seed({"k": 1.0, "ghost": 2.0}, rbc_compiled)


def test_ss_seed_seeds_a_minted_lag_through_its_origin(deep_compiled):
    # At a steady state every date coincides, so an aux is its origin. It is not
    # separately addressable, and seeding the origin seeds the whole chain.
    seed = DSGESolver._resolve_ss_seed({"u": 7.0}, deep_compiled)

    idx = deep_compiled.idx
    assert seed[idx["u"]] == seed[idx["u_lag1"]] == seed[idx["u_lag2"]] == 7.0

    with pytest.raises(ValueError, match="compiler-minted"):
        DSGESolver._resolve_ss_seed({"u_lag1": 7.0}, deep_compiled)
