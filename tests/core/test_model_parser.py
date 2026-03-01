# type: ignore
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import sympy as sp
import yaml

from SymbolicDSGE.core.model_parser import ModelParser


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
    assert kalman.y_names == ["Infl", "Rate"]


def test_validate_constraints_errors_when_marked_constrained_without_equation(
    parsed_test,
):
    conf = copy.deepcopy(parsed_test.model)
    conf.constrained[conf.variables[0]] = True
    conf.equations.constraint = type(conf.equations.constraint)({})

    with pytest.raises(ValueError, match="no constraint equations"):
        ModelParser.validate_constraints(conf)


def test_validate_constraints_errors_for_unknown_constrained_variable(parsed_test):
    conf = copy.deepcopy(parsed_test.model)
    ghost = sp.Function("ghost")
    conf.constrained[ghost] = True

    with pytest.raises(ValueError, match="do not exist"):
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
