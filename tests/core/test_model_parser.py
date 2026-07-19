# type: ignore
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
import yaml

from SymbolicDSGE.core.model_parser import ModelParser
from SymbolicDSGE.core.linearization import LinearizationMethod


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_model_parser_get_and_get_all(parsed_test):
    model, kalman = parsed_test
    parser = ModelParser("MODELS/test.yaml")

    assert parser.get() is parser.get_all().model
    assert model.name == "TEST"
    assert kalman is None


def test_parsed_config_is_iterable(parsed_post82):
    model, kalman = parsed_post82

    assert model.name == "NK_LS_POST82"
    assert kalman is not None
    assert kalman.R is not None


def test_kalman_R_built_numerically_from_calibration(parsed_post82):
    # R is now assembled directly from calibration values at parse time (no
    # sympy Matrix / lambdify). POST82 calibrates every measurement std to 1 and
    # every measurement correlation to 0, so R collapses to the identity.
    _, kalman = parsed_post82

    assert isinstance(kalman.R, np.ndarray)
    assert kalman.R.dtype == np.float64
    np.testing.assert_allclose(kalman.R, np.eye(3, dtype=np.float64))

    # Surviving metadata: the name->position maps that drive R reconstruction.
    assert kalman.R_std_param_map == {
        "OutGap": "meas_outgap",
        "Infl": "meas_infl",
        "Rate": "meas_rate",
    }
    assert kalman.R_corr_param_map == {
        frozenset({"Infl", "Rate"}): "meas_rho_ir",
        frozenset({"OutGap", "Infl"}): "meas_rho_gi",
        frozenset({"OutGap", "Rate"}): "meas_rho_gr",
    }
    assert set(kalman.R_param_names) == {
        "meas_outgap",
        "meas_infl",
        "meas_rate",
        "meas_rho_ir",
        "meas_rho_gi",
        "meas_rho_gr",
    }

    # The lambdify-era fields are gone from the config.
    assert not hasattr(kalman, "R_builder")
    assert not hasattr(kalman, "R_symbolic")
    assert not hasattr(kalman, "R_param_symbols")


_R_ARITHMETIC_MODEL = """
name: RTEST
variables:
  x: {steady_state: null}
  y: {steady_state: null}
  z: {steady_state: null}
parameters: [rho, sig, sig_x, sig_y, sig_z, rho_xy]
shock_map:
  e_x: x
  e_y: y
  e_z: z
observables: [x_obs, y_obs, z_obs]
equations:
  model:
    - x(t+1) = rho * x(t) + e_x
    - y(t+1) = rho * y(t) + e_y
    - z(t+1) = rho * z(t) + e_z
  constraint: {}
  observables:
    x_obs: x(t)
    y_obs: y(t)
    z_obs: z(t)
calibration:
  parameters:
    rho: 0.9
    sig: 0.1
    sig_x: 2.0
    sig_y: 3.0
    sig_z: 4.0
    rho_xy: 0.5
  shocks:
    std:
      e_x: sig
      e_y: sig
      e_z: sig
    corr: {}
kalman:
  R:
    std:
      x_obs: sig_x
      y_obs: sig_y
      z_obs: sig_z
    corr:
      x_obs, y_obs: rho_xy
"""


def test_kalman_R_arithmetic_covers_offdiag_and_missing_corr():
    # Non-trivial stds with a single specified correlation: exercises both the
    # sig_i * sig_j * rho off-diagonal (x_obs, y_obs) and the missing-pair -> 0
    # default (any pair involving z_obs).
    _, kalman = ModelParser.from_string(_R_ARITHMETIC_MODEL).get_all()

    expected = np.array(
        [
            [4.0, 3.0, 0.0],  # sig_x^2, sig_x*sig_y*rho_xy, 0
            [3.0, 9.0, 0.0],  # symmetric, sig_y^2, 0
            [0.0, 0.0, 16.0],  # 0, 0, sig_z^2
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(kalman.R, expected)

    # Unspecified pairs are recorded as None, not dropped.
    assert kalman.R_corr_param_map == {
        frozenset({"x_obs", "y_obs"}): "rho_xy",
        frozenset({"x_obs", "z_obs"}): None,
        frozenset({"y_obs", "z_obs"}): None,
    }
    assert set(kalman.R_param_names) == {"sig_x", "sig_y", "sig_z", "rho_xy"}


def _p0_model_dict(p0: dict) -> dict:
    """A minimal two-variable (x, y) model carrying a `kalman.P0` block."""
    return {
        "name": "P0TEST",
        "variables": {"x": {"steady_state": None}, "y": {"steady_state": None}},
        "parameters": ["rho", "sig"],
        "shock_map": {"e_x": "x", "e_y": "y"},
        "observables": ["x_obs", "y_obs"],
        "equations": {
            "model": ["x(t+1) = rho * x(t) + e_x", "y(t+1) = rho * y(t) + e_y"],
            "constraint": {},
            "observables": {"x_obs": "x(t)", "y_obs": "y(t)"},
        },
        "calibration": {
            "parameters": {"rho": 0.9, "sig": 0.1},
            "shocks": {"std": {"e_x": "sig", "e_y": "sig"}, "corr": {}},
        },
        "kalman": {"P0": p0},
    }


def test_validate_P0_accepts_exact_diag_and_eye():
    ModelParser._validate_P0("diag", {"a": 1.0, "b": 2.0}, ["a", "b"])
    ModelParser._validate_P0("eye", None, ["a", "b"])


@pytest.mark.parametrize(
    "mode, diag, declared, match",
    [
        ("diag", None, ["a"], "missing in configuration"),
        ("diag", {"a": 1.0}, ["a", "b"], r"missing \['b'\], unknown \[\]"),
        (
            "diag",
            {"a": 1.0, "b": 2.0, "c": 3.0},
            ["a", "b"],
            r"missing \[\], unknown \['c'\]",
        ),
        ("diag", {"a": 1.0, "x": 2.0}, ["a", "b"], r"missing \['b'\], unknown \['x'\]"),
        ("diag", {"a": -1.0, "b": 2.0}, ["a", "b"], "must be non-negative"),
        ("triangle", {}, ["a"], "Unrecognized P0 mode"),
    ],
)
def test_validate_P0_rejects_bad_specs(mode, diag, declared, match):
    with pytest.raises(ValueError, match=match):
        ModelParser._validate_P0(mode, diag, declared)


def test_parse_builds_p0_ndarray_in_declared_order():
    text = yaml.safe_dump(
        _p0_model_dict({"mode": "diag", "diag": {"x": 1.0, "y": 2.0}})
    )
    _, kalman = ModelParser.from_string(text).get_all()
    np.testing.assert_array_equal(kalman.P0, np.diag([1.0, 2.0]).astype(np.float64))


def test_parse_defaults_p0_to_identity_for_eye_mode():
    text = yaml.safe_dump(_p0_model_dict({"mode": "eye"}))
    _, kalman = ModelParser.from_string(text).get_all()
    np.testing.assert_array_equal(kalman.P0, np.eye(2, dtype=np.float64))


def test_parse_rejects_incomplete_p0_diag():
    text = yaml.safe_dump(_p0_model_dict({"mode": "diag", "diag": {"x": 1.0}}))
    with pytest.raises(ValueError, match=r"missing \['y'\], unknown \[\]"):
        ModelParser.from_string(text)


def test_validate_constraints_errors_on_unknown_symbols(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    t = sp.Symbol("t", integer=True)
    ghost = sp.Function("ghost")
    var = conf.variables.variables[0]
    # OBC condition references an undeclared variable -> rejected.
    conf.equations.constraint = type(conf.equations.constraint)(
        {var: {ghost(t) < 0: sp.Integer(0)}}
    )

    with pytest.raises(ValueError, match="unknown symbols"):
        ModelParser.validate_constraints(conf)


def test_validate_constraints_accepts_valid_obc(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    t = sp.Symbol("t", integer=True)
    var = conf.variables.variables[0]
    # Well-formed OBC over a declared variable ({var(t) < 0: 0}) must not raise;
    # the time symbol is excluded and the binding is a valid Expr.
    conf.equations.constraint = type(conf.equations.constraint)(
        {var: {var(t) < 0: sp.Integer(0)}}
    )

    ModelParser.validate_constraints(conf)


def test_validate_calib_errors_for_unknown_parameter(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    bad_param = sp.Symbol("not_declared")
    conf.calibration.parameters[bad_param] = 1.0

    with pytest.raises(ValueError, match="unknown parameters"):
        ModelParser.validate_calib(conf)


def test_require_calibrated_params_rejects_missing_declared(tmp_path):
    data = yaml.safe_load(Path("MODELS/test.yaml").read_text(encoding="utf-8"))
    data["calibration"]["parameters"].pop("beta")
    bad = _write_yaml(tmp_path / "missing_declared.yaml", data)

    with pytest.raises(ValueError, match="Missing calibration values"):
        ModelParser(bad)


def test_require_calibrated_params_rejects_unknown_referenced_parameter(tmp_path):
    data = yaml.safe_load(Path("MODELS/test.yaml").read_text(encoding="utf-8"))
    data["calibration"]["shocks"]["std"]["e_u"] = "unknown_sigma"
    bad = _write_yaml(tmp_path / "unknown_ref.yaml", data)

    with pytest.raises(ValueError, match="not declared in `parameters`"):
        ModelParser(bad)


def test_require_calibrated_params_rejects_missing_declared_parameter_even_if_referenced(
    tmp_path,
):
    data = yaml.safe_load(Path("MODELS/test.yaml").read_text(encoding="utf-8"))
    data["calibration"]["shocks"]["std"]["e_u"] = "sig_u"
    data["calibration"]["parameters"].pop("sig_u")
    bad = _write_yaml(tmp_path / "missing_ref.yaml", data)

    with pytest.raises(ValueError, match="Missing calibration values"):
        ModelParser(bad)


def test_parser_rejects_model_equation_without_single_equals(tmp_path):
    data = yaml.safe_load(Path("MODELS/test.yaml").read_text(encoding="utf-8"))
    data["equations"]["model"][0] = "Pi(t) + x(t)"
    bad = _write_yaml(tmp_path / "bad_eq.yaml", data)

    with pytest.raises(ValueError, match="must contain exactly one '='"):
        ModelParser(bad)


def test_legacy_variable_list_defaults_linearization_and_steady_state(parsed_test):
    conf = parsed_test.model

    assert conf.symbolically_linearized is False
    assert [v.__name__ for v in conf.variables.variables] == [
        "u",
        "v",
        "r",
        "Pi",
        "x",
        "r_star",
    ]
    assert all(
        method == LinearizationMethod.NONE
        for method in conf.variables.linearization.values()
    )
    assert all(ss is None for ss in conf.variables.steady_state.values())


def test_parser_builds_variable_metadata_from_mapping(tmp_path):
    data = yaml.safe_load(Path("MODELS/test.yaml").read_text(encoding="utf-8"))
    data["variables"] = {
        "u": {"linearization": "taylor"},
        "v": {},
        "r": {"linearization": "log", "steady_state": "rbar"},
        "Pi": {"steady_state": "pi_mean"},
        "x": {"steady_state": None},
        "r_star": {"linearization": "none"},
    }
    bad = _write_yaml(tmp_path / "variable_metadata.yaml", data)

    conf = ModelParser(bad).get()

    assert [v.__name__ for v in conf.variables.variables] == [
        "u",
        "v",
        "r",
        "Pi",
        "x",
        "r_star",
    ]
    assert conf.variables.linearization["u"] == LinearizationMethod.TAYLOR
    assert conf.variables.linearization["v"] == LinearizationMethod.NONE
    assert conf.variables.linearization["r"] == LinearizationMethod.LOG
    assert conf.variables.steady_state["r"] == sp.Symbol("rbar")
    assert conf.variables.steady_state["Pi"] == sp.Symbol("pi_mean")
    assert conf.variables.steady_state["x"] is None


def test_parser_rejects_legacy_steady_state_typo_key(tmp_path):
    data = yaml.safe_load(Path("MODELS/test.yaml").read_text(encoding="utf-8"))
    data["variables"] = {
        "u": {"stead_state": "ubar"},
        "v": {},
        "r": {},
        "Pi": {},
        "x": {},
        "r_star": {},
    }
    bad = _write_yaml(tmp_path / "bad_steady_state_key.yaml", data)

    with pytest.raises(ValueError, match="unsupported metadata keys"):
        ModelParser(bad)


def test_parser_rejects_unknown_variable_metadata_keys(tmp_path):
    data = yaml.safe_load(Path("MODELS/test.yaml").read_text(encoding="utf-8"))
    data["variables"] = {
        "u": {"linearization": "taylor", "foo": 1},
        "v": {},
        "r": {},
        "Pi": {},
        "x": {},
        "r_star": {},
    }
    bad = _write_yaml(tmp_path / "bad_variable_metadata.yaml", data)

    with pytest.raises(ValueError, match="unsupported metadata keys"):
        ModelParser(bad)
