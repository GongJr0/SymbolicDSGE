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


# ---------------- parameter-name uniqueness across std/corr maps ----------------

# Three shocks and three observables, so every map has more than one std entry
# and more than one correlation pair to collide with.
MULTI = """
name: MULTI
variables:
  x: {ss_seed: null}
  y: {ss_seed: null}
  z: {ss_seed: null}
shocks:
  - e_x
  - e_y
  - e_z
observables: [x_obs, y_obs, z_obs]
equations:
  model:
    x_process: "x(t+1) = rho * x(t) + e_x"
    y_process: "y(t+1) = rho * y(t) + e_y"
    z_process: "z(t+1) = rho * z(t) + e_z"
  constraint: {}
  observables:
    x_obs: x(t)
    y_obs: y(t)
    z_obs: z(t)
calibration:
  parameters:
    rho: 0.9
    sig_x: 0.1
    sig_y: 0.2
    sig_z: 0.3
    rho_xy: 0.0
    rho_xz: 0.0
    rho_yz: 0.0
    meas_x: 0.1
    meas_y: 0.2
    meas_z: 0.3
    meas_rho_xy: 0.0
    meas_rho_xz: 0.0
    meas_rho_yz: 0.0
  shocks:
    std:
      e_x: sig_x
      e_y: sig_y
      e_z: sig_z
    corr:
      "e_x, e_y": rho_xy
      "e_x, e_z": rho_xz
      "e_y, e_z": rho_yz
kalman:
  R:
    std:
      x_obs: meas_x
      y_obs: meas_y
      z_obs: meas_z
    corr:
      "x_obs, y_obs": meas_rho_xy
      "x_obs, z_obs": meas_rho_xz
      "y_obs, z_obs": meas_rho_yz
"""


def test_multi_is_valid():
    model, kalman = _parse(MULTI)
    assert model is not None
    assert kalman is not None


def test_accepts_shock_std_shared_across_shocks():
    # One scale driving two shocks is a restriction, not an error: the covariance
    # spec indexes the calibration by slot, so a repeated slot resolves the same
    # value for both members and one estimated parameter moves both variances.
    text = MULTI.replace("      e_y: sig_y\n", "      e_y: sig_x\n")
    model, _ = _parse(text)
    assert model.calibration.shock_std["e_x"] == "sig_x"
    assert model.calibration.shock_std["e_y"] == "sig_x"


def test_accepts_measurement_std_shared_across_observables():
    text = MULTI.replace("      y_obs: meas_y\n", "      y_obs: meas_x\n")
    _, kalman = _parse(text)
    assert kalman.R_std_param_map["x_obs"] == "meas_x"
    assert kalman.R_std_param_map["y_obs"] == "meas_x"


def test_rejects_shock_corr_parameter_used_for_two_pairs():
    text = MULTI.replace('"e_x, e_z": rho_xz', '"e_x, e_z": rho_xy')
    with pytest.raises(
        ValueError, match="calibration.shocks.corr values must be unique"
    ):
        _parse(text)


def test_rejects_shock_parameter_used_as_both_std_and_corr():
    text = MULTI.replace('"e_x, e_y": rho_xy', '"e_x, e_y": sig_x')
    with pytest.raises(
        ValueError, match="cannot share parameter names between std and corr"
    ):
        _parse(text)


def test_rejects_measurement_corr_parameter_used_for_two_pairs():
    text = MULTI.replace('"x_obs, z_obs": meas_rho_xz', '"x_obs, z_obs": meas_rho_xy')
    with pytest.raises(ValueError, match="kalman.R.corr values must be unique"):
        _parse(text)


def test_rejects_measurement_parameter_used_as_both_std_and_corr():
    text = MULTI.replace('"x_obs, y_obs": meas_rho_xy', '"x_obs, y_obs": meas_x')
    with pytest.raises(
        ValueError, match="cannot share parameter names between std and corr"
    ):
        _parse(text)


def test_unspecified_correlation_pairs_do_not_count_as_duplicates():
    # The parser fills every undeclared pair with None, so several Nones coexist
    # in the map; only named parameters are subject to the uniqueness rule.
    text = MULTI.replace('      "e_x, e_z": rho_xz\n', "").replace(
        '      "e_y, e_z": rho_yz\n', ""
    )
    model, _ = _parse(text)
    corr = model.calibration.shock_corr
    assert corr["e_x", "e_y"] == "rho_xy"
    assert corr["e_x", "e_z"] is None
    assert corr["e_y", "e_z"] is None
