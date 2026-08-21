"""End-to-end rejection tests: malformed configs must not slip through the parser.

Each case starts from a minimal valid model and breaks exactly one field, then
asserts ``ModelParser.from_string(...).get_all()`` raises with a clear message.
"""

from __future__ import annotations

import pytest
import sympy as sp

from SymbolicDSGE import ModelParser


def _parse(text: str):
    return ModelParser.from_string(text).get_all()


BASE = """
name: MINI
variables:
  x: {ss_seed: null}
shocks:
  - e
observables: [x_obs]
equations:
  model:
    x_process: "x(t+1) = rho * x(t) + e"
  constraint: {}
  observables:
    x_obs: x(t)
calibration:
  parameters:
    rho: 0.9
    sig: 0.1
  shocks:
    std:
      e: sig
    corr: {}
"""


def test_base_is_valid():
    model, _ = _parse(BASE)
    assert model is not None


def test_rejects_invalid_linearization_method():
    text = BASE.replace(
        "x: {ss_seed: null}",
        "x: {ss_seed: null, linearization: bogus}",
    )
    with pytest.raises(ValueError, match="Invalid linearization method 'bogus'"):
        _parse(text)


def test_rejects_malformed_shock_correlation_pair():
    text = BASE.replace("corr: {}", 'corr: {"e": sig}')
    with pytest.raises(ValueError, match="exactly two shocks"):
        _parse(text)


def test_rejects_trivial_equation():
    text = BASE.replace("x(t+1) = rho * x(t) + e", "1 = 1")
    with pytest.raises(TypeError, match="Not a valid equality"):
        _parse(text)


def test_rejects_malformed_observable_correlation_pair():
    # rho_obs must be calibrated so the undeclared-parameter check passes and
    # parsing reaches the R-correlation pair guard.
    text = BASE.replace("    sig: 0.1", "    sig: 0.1\n    rho_obs: 0.0")
    kalman_block = """
kalman:
  R:
    std: {}
    corr:
      x_obs: rho_obs
"""
    with pytest.raises(ValueError, match="exactly two observables"):
        _parse(text + kalman_block)


# ---------------- derived calibration entries ("locals") ----------------


def test_derived_entry_is_inlined_and_dropped():
    text = BASE.replace("    sig: 0.1", "    sig: 0.1\n    half_rho: rho / 2").replace(
        "rho * x(t)", "half_rho * x(t)"
    )
    model, _ = _parse(text)

    equation = model.equations.model["x_process"]
    free = (equation.lhs - equation.rhs).free_symbols
    assert sp.Symbol("half_rho") not in free
    assert sp.Symbol("rho") in free
    assert sp.Symbol("half_rho") not in model.parameters
    assert sp.Symbol("half_rho") not in model.calibration.parameters


def test_derived_entries_resolve_through_each_other():
    text = BASE.replace(
        "    sig: 0.1",
        "    sig: 0.1\n    half_rho: rho / 2\n    quarter_rho: half_rho / 2",
    ).replace("x: {ss_seed: null}", "x: {ss_seed: quarter_rho}")
    model, _ = _parse(text)

    seed = model.variables.ss_seed[model.variables.variables[0]]
    assert sp.simplify(seed - sp.Symbol("rho") / 4) == 0


def test_rejects_calibration_entry_with_undeclared_symbol():
    text = BASE.replace("    sig: 0.1", "    sig: 0.1\n    bad: rho * nope")
    with pytest.raises(ValueError, match="references undeclared parameter"):
        _parse(text)


def test_rejects_calibration_entry_referencing_model_variable():
    text = BASE.replace("    sig: 0.1", "    sig: 0.1\n    bad: rho * x(t)")
    with pytest.raises(ValueError, match="references model variable"):
        _parse(text)


def test_rejects_self_referential_calibration_entry():
    text = BASE.replace("    sig: 0.1", "    sig: 0.1\n    loop: loop + 1")
    with pytest.raises(ValueError, match="cycle: loop"):
        _parse(text)


def test_rejects_mutually_referential_calibration_entries():
    text = BASE.replace("    sig: 0.1", "    sig: 0.1\n    a: b + 1\n    b: a * 2")
    with pytest.raises(ValueError, match="cycle: a, b"):
        _parse(text)


def test_rejects_derived_entry_named_as_shock_std():
    text = BASE.replace("    sig: 0.1", "    sig: 0.1\n    sig_loc: sig * 2").replace(
        "      e: sig\n", "      e: sig_loc\n"
    )
    with pytest.raises(ValueError, match="cannot name a derived calibration entry"):
        _parse(text)
