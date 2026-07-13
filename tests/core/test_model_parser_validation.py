"""End-to-end rejection tests: malformed configs must not slip through the parser.

Each case starts from a minimal valid model and breaks exactly one field, then
asserts ``ModelParser.from_string(...).get_all()`` raises with a clear message.
"""

from __future__ import annotations

import pytest

from SymbolicDSGE import ModelParser


def _parse(text: str):
    return ModelParser.from_string(text).get_all()


BASE = """
name: MINI
variables:
  x: {steady_state: null}
parameters: [rho, sig]
shock_map:
  e: x
observables: [x_obs]
equations:
  model:
    - x(t+1) = rho * x(t) + e
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
        "x: {steady_state: null}",
        "x: {steady_state: null, linearization: bogus}",
    )
    with pytest.raises(ValueError, match="Invalid linearization method 'bogus'"):
        _parse(text)


def test_rejects_declared_but_uncalibrated_parameter():
    text = BASE.replace(
        "parameters: [rho, sig]", "parameters: [rho, sig, phi]"
    ).replace("x(t+1) = rho * x(t) + e", "x(t+1) = rho * x(t) + phi * x(t) + e")
    with pytest.raises(ValueError, match="[Mm]issing calibration values"):
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
    # rho_obs must be declared + calibrated so the undeclared-parameter check
    # passes and parsing reaches the R-correlation pair guard.
    text = BASE.replace(
        "parameters: [rho, sig]", "parameters: [rho, sig, rho_obs]"
    ).replace("    sig: 0.1", "    sig: 0.1\n    rho_obs: 0.0")
    kalman_block = """
kalman:
  R:
    std: {}
    corr:
      x_obs: rho_obs
"""
    with pytest.raises(ValueError, match="exactly two observables"):
        _parse(text + kalman_block)
