# type: ignore
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
import yaml

from SymbolicDSGE.core.model_parser import ModelParser, _check_connective_parens
from SymbolicDSGE.core.config import Constraint
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
  x: {ss_seed: null}
  y: {ss_seed: null}
  z: {ss_seed: null}
shock_map:
  e_x: x
  e_y: y
  e_z: z
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
        "variables": {"x": {"ss_seed": None}, "y": {"ss_seed": None}},
        "shock_map": {"e_x": "x", "e_y": "y"},
        "observables": ["x_obs", "y_obs"],
        "equations": {
            "model": {
                "x_process": "x(t+1) = rho * x(t) + e_x",
                "y_process": "y(t+1) = rho * y(t) + e_y",
            },
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
    # Binding condition references an undeclared variable -> rejected.
    conf.equations.constraint = {
        "obc": Constraint(bind=ghost(t) < 0, relax=ghost(t) >= 0)
    }

    with pytest.raises(ValueError, match="unknown symbols"):
        ModelParser.validate_constraints(conf)


def test_validate_constraints_accepts_valid_conditions(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    t = sp.Symbol("t", integer=True)
    var = conf.variables.variables[0]
    # Declared variable on both conditions; the time symbol is excluded.
    conf.equations.constraint = {"obc": Constraint(bind=var(t) < 0, relax=var(t) >= 0)}

    ModelParser.validate_constraints(conf)


def test_validate_constraints_accepts_boolean_conditions(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    t = sp.Symbol("t", integer=True)
    a, b = conf.variables.variables[:2]
    conf.equations.constraint = {
        "obc": Constraint(bind=sp.And(a(t) < 0, b(t) < 0), relax=a(t) >= 0)
    }

    ModelParser.validate_constraints(conf)


def test_validate_constraints_accepts_nested_connectives(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    t = sp.Symbol("t", integer=True)
    a, b = conf.variables.variables[:2]
    cond = sp.Or(sp.And(a(t) < 0, b(t) < 0), sp.Not(a(t) > 5))
    conf.equations.constraint = {"obc": Constraint(bind=cond, relax=a(t) >= 0)}

    ModelParser.validate_constraints(conf)


@pytest.mark.parametrize("depth", ["root", "nested"])
def test_validate_constraints_rejects_non_relational_leaves(parsed_test, depth):
    # Symbol subclasses Boolean, so And(x < 0, param) builds fine and the root
    # type gate accepts it; only the leaf walk rejects the bare operand.
    conf = copy.deepcopy(parsed_test.model)
    t = sp.Symbol("t", integer=True)
    a = conf.variables.variables[0]
    param = conf.parameters[0]
    bind = param if depth == "root" else sp.Or(sp.And(a(t) < 0, param), a(t) > 5)
    conf.equations.constraint = {"obc": Constraint(bind=bind, relax=a(t) >= 0)}

    with pytest.raises(TypeError, match="is not a valid SymPy Relational"):
        ModelParser.validate_constraints(conf)


def test_validate_constraints_rejects_relation_compared_to_relation(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    t = sp.Symbol("t", integer=True)
    a, b = conf.variables.variables[:2]
    conf.equations.constraint = {
        "obc": Constraint(bind=sp.Eq(a(t) < 0, b(t) < 0), relax=a(t) >= 0)
    }

    with pytest.raises(TypeError, match="compares the truth value"):
        ModelParser.validate_constraints(conf)


@pytest.mark.parametrize("side", ["bind", "relax"])
def test_validate_constraints_rejects_shocks(parsed_test, side):
    conf = copy.deepcopy(parsed_test.model)
    t = sp.Symbol("t", integer=True)
    a = conf.variables.variables[0]
    shock = next(iter(conf.shock_map))
    conditions = {"bind": a(t) < 0, "relax": a(t) >= 0}
    conditions[side] = a(t) + shock < 0

    conf.equations.constraint = {"obc": Constraint(**conditions)}

    match = f"references shock\\(s\\) \\['{shock.name}'\\]"
    with pytest.raises(ValueError, match=match):
        ModelParser.validate_constraints(conf)


@pytest.mark.parametrize(
    "condition",
    [
        "x(t) < 0",
        "(x(t) > 0) & (y(t) < 1)",
        "(x(t) > 0)|(y(t) < 1)",
        "((x(t) > 0) & (y(t) < 1)) | (z(t) < 2)",
        "~(x(t) < 0)",
    ],
)
def test_connective_parens_accepts_parenthesized_relations(condition):
    _check_connective_parens(condition)


@pytest.mark.parametrize(
    "condition",
    [
        "x(t) > 0 | y(t) < 1",
        "x(t) > 0 & y(t) < 1",
        "(x(t) > 0) & y(t) < 1",
        "x(t) > 0 & (y(t) < 1)",
    ],
)
def test_connective_parens_rejects_unparenthesized_relations(condition):
    # '&'/'|' bind tighter than the comparisons, so 'x > 0 | y < 1' parses as
    # And(x > y, y < 1): a valid tree testing something never written.
    with pytest.raises(ValueError, match="outside parentheses"):
        _check_connective_parens(condition)


def test_parser_rejects_unparenthesized_connective(parsed_test):
    data = yaml.safe_load(_R_ARITHMETIC_MODEL)
    data["equations"]["constraint"] = {
        "obc": {"bind": "x(t) < 0 & y(t) < 0", "relax": "x(t) >= 0"}
    }

    with pytest.raises(ValueError, match="outside parentheses"):
        ModelParser.from_string(yaml.safe_dump(data))


def test_validate_constraints_rejects_more_than_two(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    t = sp.Symbol("t", integer=True)
    conf.equations.constraint = {
        f"obc{i}": Constraint(bind=var(t) < 0, relax=var(t) >= 0)
        for i, var in enumerate(conf.variables.variables[:3])
    }

    with pytest.raises(NotImplementedError, match="1- and 2-constraint"):
        ModelParser.validate_constraints(conf)


def test_validate_regimes_requires_constraint_and_regime_together(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    t = sp.Symbol("t", integer=True)
    var = conf.variables.variables[0]
    conf.equations.constraint = {"obc": Constraint(bind=var(t) < 0, relax=var(t) >= 0)}
    conf.equations.regime = None

    with pytest.raises(ValueError, match="declared together"):
        ModelParser.validate_regimes(conf)


def test_validate_regimes_requires_every_binding_combination(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    t = sp.Symbol("t", integer=True)
    a, b = conf.variables.variables[:2]
    first, second = list(conf.equations.model)[:2]
    conf.equations.constraint = {
        "lo": Constraint(bind=a(t) < 0, relax=a(t) >= 0),
        "hi": Constraint(bind=b(t) < 0, relax=b(t) >= 0),
    }
    # The joint cell {lo, hi} is absent.
    conf.equations.regime = {
        frozenset({"lo"}): {first: sp.Eq(a(t), 0)},
        frozenset({"hi"}): {second: sp.Eq(b(t), 0)},
    }

    with pytest.raises(ValueError, match="missing entries for binding combinations"):
        ModelParser.validate_regimes(conf)


def test_validate_regimes_rejects_unknown_replacement_target(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    t = sp.Symbol("t", integer=True)
    var = conf.variables.variables[0]
    conf.equations.constraint = {"obc": Constraint(bind=var(t) < 0, relax=var(t) >= 0)}
    conf.equations.regime = {frozenset({"obc"}): {"nosuch": sp.Eq(var(t), 0)}}

    with pytest.raises(ValueError, match="replaces undeclared model equations"):
        ModelParser.validate_regimes(conf)


def test_validate_regimes_accepts_a_complete_two_constraint_grid(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    t = sp.Symbol("t", integer=True)
    a, b = conf.variables.variables[:2]
    first, second = list(conf.equations.model)[:2]
    conf.equations.constraint = {
        "lo": Constraint(bind=a(t) < 0, relax=a(t) >= 0),
        "hi": Constraint(bind=b(t) < 0, relax=b(t) >= 0),
    }
    conf.equations.regime = {
        frozenset({"lo"}): {first: sp.Eq(a(t), 0)},
        frozenset({"hi"}): {second: sp.Eq(b(t), 0)},
        frozenset({"lo", "hi"}): {first: sp.Eq(a(t), 0), second: sp.Eq(b(t), 0)},
    }

    ModelParser.validate_regimes(conf)


def test_validate_ss_seed_accepts_scalars_and_parameter_expressions(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    beta, rho_u = sp.Symbol("beta"), sp.Symbol("rho_u")
    var = conf.variables.variables[0]
    for expr in (sp.Float(0.8), sp.Integer(0), beta, beta / (1 - rho_u), None):
        conf.variables.ss_seed[var] = expr
        ModelParser.validate_ss_seed(conf)


def test_validate_ss_seed_errors_on_undeclared_parameter(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    var = conf.variables.variables[0]
    conf.variables.ss_seed[var] = sp.Symbol("not_a_param")

    with pytest.raises(ValueError, match="references unknown symbols"):
        ModelParser.validate_ss_seed(conf)


def test_validate_ss_seed_errors_on_model_variable_reference(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    t = sp.Symbol("t", integer=True)
    var = conf.variables.variables[0]
    conf.variables.ss_seed[var] = var(t)

    with pytest.raises(ValueError, match="must resolve to a number"):
        ModelParser.validate_ss_seed(conf)


def test_parser_rejects_undeclared_ss_seed_symbol(tmp_path):
    data = yaml.safe_load(Path("MODELS/test.yaml").read_text(encoding="utf-8"))
    data["variables"] = {
        "u": {"ss_seed": "u_bar"},
        "v": {},
        "r": {},
        "Pi": {},
        "x": {},
        "r_star": {},
    }
    bad = _write_yaml(tmp_path / "bad_ss_seed.yaml", data)

    with pytest.raises(ValueError, match="references unknown symbols"):
        ModelParser(bad)


def test_uncalibrated_equation_parameter_fails_to_sympify(tmp_path):
    data = yaml.safe_load(Path("MODELS/test.yaml").read_text(encoding="utf-8"))
    data["calibration"]["parameters"].pop("beta")
    bad = _write_yaml(tmp_path / "missing_declared.yaml", data)

    with pytest.raises(ValueError, match="SympifyError: beta"):
        ModelParser(bad)


def test_require_calibrated_params_rejects_unknown_referenced_parameter(tmp_path):
    data = yaml.safe_load(Path("MODELS/test.yaml").read_text(encoding="utf-8"))
    data["calibration"]["shocks"]["std"]["e_u"] = "unknown_sigma"
    bad = _write_yaml(tmp_path / "unknown_ref.yaml", data)

    with pytest.raises(ValueError, match="not declared in `calibration.parameters`"):
        ModelParser(bad)


def test_require_calibrated_params_rejects_uncalibrated_referenced_parameter(
    tmp_path,
):
    data = yaml.safe_load(Path("MODELS/test.yaml").read_text(encoding="utf-8"))
    data["calibration"]["shocks"]["std"]["e_u"] = "sig_u"
    data["calibration"]["parameters"].pop("sig_u")
    bad = _write_yaml(tmp_path / "missing_ref.yaml", data)

    with pytest.raises(ValueError, match="not declared in `calibration.parameters`"):
        ModelParser(bad)


def test_parser_rejects_model_equation_without_single_equals(tmp_path):
    data = yaml.safe_load(Path("MODELS/test.yaml").read_text(encoding="utf-8"))
    data["equations"]["model"][0] = "Pi(t) + x(t)"
    bad = _write_yaml(tmp_path / "bad_eq.yaml", data)

    with pytest.raises(ValueError, match="must contain exactly one '='"):
        ModelParser(bad)


def test_legacy_variable_list_defaults_linearization_and_ss_seed(parsed_test):
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
    assert all(ss is None for ss in conf.variables.ss_seed.values())


def test_parser_builds_variable_metadata_from_mapping(tmp_path):
    data = yaml.safe_load(Path("MODELS/test.yaml").read_text(encoding="utf-8"))
    data["variables"] = {
        "u": {"linearization": "taylor"},
        "v": {},
        "r": {"linearization": "log", "ss_seed": "rbar"},
        "Pi": {"ss_seed": "pi_mean"},
        "x": {"ss_seed": None},
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
    assert conf.variables.ss_seed["r"] == sp.Symbol("rbar")
    assert conf.variables.ss_seed["Pi"] == sp.Symbol("pi_mean")
    assert conf.variables.ss_seed["x"] is None


def test_parser_rejects_retired_steady_state_key(tmp_path):
    data = yaml.safe_load(Path("MODELS/test.yaml").read_text(encoding="utf-8"))
    data["variables"] = {
        "u": {"steady_state": "ubar"},
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
