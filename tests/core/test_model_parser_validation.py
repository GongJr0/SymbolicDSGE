"""Rejection tests: malformed configs must not slip through the parser.

Each case starts from a minimal valid model and breaks exactly one field, then
asserts ``ModelParser.from_string(...).get_all()`` raises with a clear message.
The last section calls three parser helpers directly, for guards whose branches
no authored config can reach.
"""

from __future__ import annotations

import pytest
import sympy as sp

from SymbolicDSGE import ModelParser
from SymbolicDSGE.core.model_parser import (
    _load_yaml,
    _reject_unknown_keys,
    _sympy_parsers,
)


def _parse(text: str):
    return ModelParser.from_string(text).get()


BASE = """
name: MINI
variables: [x]
shocks:
  - e
equations:
  model:
    x_process: "x(t+1) = rho * x(t) + e"
  constraints: {}
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
    model = _parse(BASE)
    assert model is not None


def test_rejects_malformed_shock_correlation_pair():
    text = BASE.replace("corr: {}", 'corr: {"e": sig}')
    with pytest.raises(ValueError, match="exactly two shocks"):
        _parse(text)


def test_rejects_trivial_equation():
    text = BASE.replace("x(t+1) = rho * x(t) + e", "1 = 1")
    with pytest.raises(TypeError, match="Not a valid equality"):
        _parse(text)


def test_rejects_malformed_observable_correlation_pair():
    # Both names must be calibrated and every observable must carry a standard
    # deviation, so that parsing clears those guards and reaches the pair guard.
    text = BASE.replace(
        "    sig: 0.1", "    sig: 0.1\n    rho_obs: 0.0\n    meas_x: 0.2"
    )
    kalman_block = """
  measurement_noise:
    std:
      x_obs: meas_x
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
    model = _parse(text)

    equation = model.equations.model["x_process"]
    free = (equation.lhs - equation.rhs).free_symbols
    assert sp.Symbol("half_rho") not in free
    assert sp.Symbol("rho") in free
    assert sp.Symbol("half_rho") not in model.calibration.parameters


def test_derived_entries_resolve_through_each_other():
    text = BASE.replace(
        "    sig: 0.1",
        "    sig: 0.1\n    half_rho: rho / 2\n    quarter_rho: half_rho / 2",
    ).replace("variables: [x]", "variables: [x]\nss_seed:\n  x: quarter_rho")
    model = _parse(text)

    seed = model.ss_seed[model.variables[0]]
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
    with pytest.raises(ValueError, match=r"cycle: \['loop'\]"):
        _parse(text)


def test_rejects_mutually_referential_calibration_entries():
    text = BASE.replace("    sig: 0.1", "    sig: 0.1\n    a: b + 1\n    b: a * 2")
    with pytest.raises(ValueError, match=r"cycle: \['a', 'b'\]"):
        _parse(text)


def test_rejects_derived_entry_named_as_shock_std():
    text = BASE.replace("    sig: 0.1", "    sig: 0.1\n    sig_loc: sig * 2").replace(
        "      e: sig\n", "      e: sig_loc\n"
    )
    with pytest.raises(ValueError, match="reference derived calibration entries"):
        _parse(text)


# ---------------- parameter-name uniqueness across std/corr maps ----------------

# Three shocks and three observables, so every map has more than one std entry
# and more than one correlation pair to collide with.
MULTI = """
name: MULTI
variables: [x, y, z]
shocks:
  - e_x
  - e_y
  - e_z
equations:
  model:
    x_process: "x(t+1) = rho * x(t) + e_x"
    y_process: "y(t+1) = rho * y(t) + e_y"
    z_process: "z(t+1) = rho * z(t) + e_z"
  constraints: {}
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
  measurement_noise:
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
    model = _parse(MULTI)
    assert model is not None


def test_accepts_shock_std_shared_across_shocks():
    # One scale driving two shocks is a restriction, not an error: the covariance
    # spec indexes the calibration by slot, so a repeated slot resolves the same
    # value for both members and one estimated parameter moves both variances.
    text = MULTI.replace("      e_y: sig_y\n", "      e_y: sig_x\n")
    model = _parse(text)
    assert model.calibration.shocks.std["e_x"] == "sig_x"
    assert model.calibration.shocks.std["e_y"] == "sig_x"


def test_accepts_measurement_std_shared_across_observables():
    text = MULTI.replace("      y_obs: meas_y\n", "      y_obs: meas_x\n")
    model = _parse(text)
    meas_noise = model.calibration.measurement_noise
    assert meas_noise
    assert meas_noise.std["x_obs"] == "meas_x"
    assert meas_noise.std["y_obs"] == "meas_x"


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
    with pytest.raises(
        ValueError, match="calibration.measurement_noise.corr values must be unique"
    ):
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
    model = _parse(text)
    corr = model.calibration.shocks.corr
    assert corr["e_x", "e_y"] == "rho_xy"
    assert corr["e_x", "e_z"] is None
    assert corr["e_y", "e_z"] is None


# ---------------- variables and the top-level ss_seed block ----------------


def test_rejects_missing_variables():
    text = BASE.replace("variables: [x]", "")
    with pytest.raises(ValueError, match="Missing required 'variables'"):
        _parse(text)


def test_rejects_mapping_form_variables():
    text = BASE.replace("variables: [x]", "variables:\n  x: {ss_seed: null}")
    with pytest.raises(TypeError, match="'variables' must be a list"):
        _parse(text)


def test_rejects_non_string_variable_names():
    text = BASE.replace("variables: [x]", "variables: [1]")
    with pytest.raises(TypeError, match="list of strings"):
        _parse(text)


def test_rejects_non_mapping_ss_seed():
    text = BASE.replace("variables: [x]", "variables: [x]\nss_seed: [x]")
    with pytest.raises(TypeError, match="'ss_seed' must be a mapping"):
        _parse(text)


def test_rejects_ss_seed_for_undeclared_variable():
    text = BASE.replace("variables: [x]", "variables: [x]\nss_seed:\n  xx: rho")
    with pytest.raises(ValueError, match=r"Unknown field\(s\) under 'ss_seed'"):
        _parse(text)


def test_ss_seed_densifies_over_the_declared_set():
    # Authored sparse, stored dense: every declared variable gets a key, and
    # None is what the solver reads as "seed this one at zero".
    text = MULTI.replace(
        "variables: [x, y, z]", "variables: [x, y, z]\nss_seed:\n  y: rho"
    )
    model = _parse(text)
    seeds = model.ss_seed

    assert set(seeds) == set(model.variables)
    assert seeds["y"] == sp.Symbol("rho")
    assert seeds["x"] is None
    assert seeds["z"] is None


def test_accepts_an_empty_ss_seed_block():
    text = BASE.replace("variables: [x]", "variables: [x]\nss_seed:")
    model = _parse(text)

    assert model.ss_seed["x"] is None


# ---------------- the restricted parse namespace ----------------

# A name the config namespace does not carry resolves against ``_GLOBALS``,
# which admits only the functions a printer renders. Every other SymPy name
# reaches validation as an ordinary unknown symbol rather than as a constant.


@pytest.mark.parametrize("name", ["pi", "E", "I", "oo", "nan", "EulerGamma"])
def test_rejects_sympy_constants_as_undeclared_symbols(name):
    text = BASE.replace("rho * x(t) + e", f"rho * x(t) + {name} + e")
    with pytest.raises(
        ValueError, match=rf"references unknown symbol\(s\): \['{name}'\]"
    ):
        _parse(text)


@pytest.mark.parametrize(
    "call", ["tan(x(t))", "sin(x(t))", "Abs(x(t))", "Max(x(t), 0)"]
)
def test_rejects_calls_no_printer_renders(call):
    name = call.split("(")[0]
    text = BASE.replace("rho * x(t) + e", f"rho * {call} + e")
    with pytest.raises(
        ValueError, match=rf"references unknown symbol\(s\): \['{name}'\]"
    ):
        _parse(text)


def test_rejects_factorial_notation():
    text = BASE.replace("rho * x(t)", "rho! * x(t)")
    with pytest.raises(SyntaxError, match="invalid syntax"):
        _parse(text)


def test_rejects_unrenderable_call_in_ss_seed():
    text = BASE.replace("variables: [x]", "variables: [x]\nss_seed:\n  x: tan(rho)")
    with pytest.raises(ValueError, match=r"references unknown symbol\(s\): \['tan'\]"):
        _parse(text)


def test_rejects_model_variable_in_ss_seed():
    # The declared-variable branch, which the unknown-call guard must not claim.
    text = BASE.replace("variables: [x]", "variables: [x]\nss_seed:\n  x: x(t)")
    with pytest.raises(ValueError, match="must resolve to a number"):
        _parse(text)


def test_rejects_unrenderable_call_in_calibration_entry():
    text = BASE.replace("    sig: 0.1", "    sig: 0.1\n    bad: tan(rho)")
    with pytest.raises(ValueError, match=r"references unknown symbol\(s\): \['tan'\]"):
        _parse(text)


# ---------------- helper guards with no authored expression ----------------


def test_reject_unknown_keys():
    # non-dict is a no-op
    _reject_unknown_keys(["not", "a", "dict"], frozenset(), "where")
    # subset is fine
    _reject_unknown_keys({"a": 1}, frozenset({"a", "b"}), "where")
    with pytest.raises(ValueError, match="Unknown field"):
        _reject_unknown_keys({"z": 1}, frozenset({"a"}), "block")


def test_load_yaml_rejects_non_mapping_root():
    with pytest.raises(TypeError, match="root must be a mapping"):
        _load_yaml("- a\n- b\n", "<test>")


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


def test_sympy_parsers_accept_spelled_out_conditions():
    locals_ = {"x": sp.Symbol("x"), "y": sp.Symbol("y")}
    _get_expr, get_relational, _get_eq = _sympy_parsers(locals_, {})

    assert get_relational("And((x > 0), (y < 1))").func is sp.And
    assert get_relational("Or((x > 0), (y < 1))").func is sp.Or
    assert get_relational("Not(x > 0)").rel_op == "<="
    assert get_relational("Le(x, 0)").rel_op == "<="
    # An equality has no distance to a boundary, so no printer renders it and
    # its name is not admitted.
    with pytest.raises(TypeError, match="not a valid SymPy Relational"):
        get_relational("Eq(x, 0)")


@pytest.mark.parametrize(
    "text, func",
    # sqrt lowers to a Pow, which the printer reaches through its exponent.
    [("exp(x)", sp.exp), ("log(x)", sp.log), ("sqrt(x)", sp.Pow)],
)
def test_sympy_parsers_resolve_the_renderable_functions(text, func):
    get_expr, _get_relational, _get_eq = _sympy_parsers({"x": sp.Symbol("x")}, {})
    assert get_expr(text).func is func
