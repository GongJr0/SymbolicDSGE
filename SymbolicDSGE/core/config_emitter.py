from __future__ import annotations
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

import yaml
from pathlib import Path

from sympy import Eq, Basic
from sympy.core import Float
from sympy.printing import StrPrinter


class _FloatNormalizingPrinter(StrPrinter):
    """A sympy printer that normalizes floats to a consistent format."""

    def _print_Float(self, expr: Float) -> str:
        return repr(float(expr))


_PRINTER = _FloatNormalizingPrinter()


def _sstr(expr: Basic) -> str:
    """Convert a sympy expression to a string using the custom printer."""
    return str(_PRINTER.doprint(expr))


if TYPE_CHECKING:
    from .config import (
        ModelConfig,
        Equations,
        Calib,
        Constraint,
        RegimeGetterDict,
        CovarianceSpec,
    )

# --- YAML Representation Spec ---

T = TypeVar("T")


class _InlineList(list[T]):
    """Indicatoes a list should be rendered inline in YAML."""

    pass


class _ConfigDumper(yaml.SafeDumper):
    """Custom YAML dumper that handles inline lists."""

    pass


def _represent_inline_list(dumper: _ConfigDumper, data: _InlineList[Any]) -> yaml.Node:
    """Represent an _InlineList as a YAML sequence with flow style."""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_ConfigDumper.add_representer(_InlineList, _represent_inline_list)

# --- Formatting ---


def _round_digits(value: float, digits: int | None) -> float:
    """Round a float to a specified number of digits."""
    if digits is None:
        return value
    return round(value, digits)


def _eq_fmt(eq: Eq) -> str:
    """Format a sympy equation as a string."""
    if not isinstance(eq, Eq):
        raise TypeError(f"Expected sympy.Eq, got {type(eq)}")
    return f"{_sstr(eq.lhs)} = {_sstr(eq.rhs)}"


# --- Schemas ---


class _EquationsSchema(TypedDict):
    model: dict[str, str]
    constraints: dict[str, dict[str, str]]
    regimes: dict[str, dict[str, str]]
    observables: dict[str, str]


class _CalibSchema(TypedDict):
    parameters: dict[str, float]
    shocks: dict[str, dict[str, str]]
    measurement_noise: dict[str, dict[str, str]]


class _ConfigSchema(TypedDict):
    name: str
    variables: _InlineList[str]
    ss_seed: dict[str, str | float]
    shocks: _InlineList[str]
    equations: _EquationsSchema
    calibration: _CalibSchema


# --- Equation Block ---


def _emit_constraints(
    constraints: dict[str, Constraint] | None,
) -> dict[str, dict[str, str]]:
    """Emit constraints as a dictionary of strings."""
    if constraints is None:
        return {}
    return {
        name: {"bind": _sstr(const.bind), "relax": _sstr(const.relax)}
        for name, const in constraints.items()
    }


def _emit_regimes(regimes: RegimeGetterDict | None) -> dict[str, dict[str, str]]:
    """Emit regimes as a dictionary of strings."""
    if regimes is None:
        return {}
    return {
        ", ".join(sorted(constraints)): {
            name: _eq_fmt(eq) for name, eq in regime.items()
        }
        for constraints, regime in regimes.items()
    }


def _emit_equations(
    equations: Equations,
) -> _EquationsSchema:
    """Emit equations as a dictionary of strings."""
    model = {str(name): _eq_fmt(eq) for name, eq in equations.model.items()}
    constraints = _emit_constraints(equations.constraints)
    regimes = _emit_regimes(equations.regimes)
    observables = {
        str(name): _sstr(expr) for name, expr in equations.observables.items()
    }

    return {
        "model": model,
        "constraints": constraints,
        "regimes": regimes,
        "observables": observables,
    }


# --- Calibration Block ---


def _emit_covariance_spec(spec: CovarianceSpec | None) -> dict[str, dict[str, str]]:
    """Emit a covariance specification as a dictionary of strings."""
    if spec is None:
        return {}

    std = {str(name): str(param) for name, param in spec.std.items()}
    corr = {
        ", ".join(map(str, sorted(pair, key=str))): str(param)
        for pair, param in spec.corr.items()
        if param is not None
    }
    return {"std": std, "corr": corr}


def _emit_calib(calib: Calib, digits: int | None = None) -> _CalibSchema:
    """Emit calibration as a dictionary of strings."""
    parameters = {
        str(name): _round_digits(float(value), digits)
        for name, value in calib.parameters.items()
    }
    shocks = _emit_covariance_spec(calib.shocks)
    measurement_noise = _emit_covariance_spec(calib.measurement_noise)
    return {
        "parameters": parameters,
        "shocks": shocks,
        "measurement_noise": measurement_noise,
    }


# --- Model Config ---


def _emit_model_config(config: ModelConfig, digits: int | None = None) -> _ConfigSchema:
    """Emit a model configuration as a dictionary of strings."""
    name = config.name
    variables = [str(var) for var in config.variables]
    ss_seed = {
        str(name): (
            _round_digits(float(expr), digits)
            if expr.is_number and expr.is_real
            else _sstr(expr)
        )
        for name, expr in config.ss_seed.items()
        if expr is not None
    }
    shocks = [str(shock) for shock in config.shocks]
    equations = _emit_equations(config.equations)
    calibration = _emit_calib(config.calibration, digits)
    return {
        "name": name,
        "variables": _InlineList(variables),
        "ss_seed": ss_seed,
        "shocks": _InlineList(shocks),
        "equations": equations,
        "calibration": calibration,
    }


# --- YAML Dumping ---


def emit_config_yaml(config: ModelConfig, digits: int | None = None) -> str:
    """Emit a model configuration as a YAML string.

    Parameters
    ----------
    config : ModelConfig
        Config to produce a YAML representation of.
    digits : int | None
        Number of digits to use for plain floats.

    Returns
    -------
    str
        YAML representation of the model configuration.

    """
    return yaml.dump(
        _emit_model_config(config, digits),
        Dumper=_ConfigDumper,
        sort_keys=False,
        default_flow_style=False,
        width=10**6,
        indent=2,
        allow_unicode=True,
    )


def write_config_yaml(
    config: ModelConfig, path: str | Path, digits: int | None = None
) -> None:
    """Write a model configuration to a YAML file.

    Parameters
    ----------
    config : ModelConfig
        Config to write to a YAML file.
    path : str | Path
        Path to the output YAML file.
    digits : int | None
        Number of digits to use for plain floats.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    yaml_str = emit_config_yaml(config, digits)

    with open(path, "w", encoding="utf-8") as f:
        f.write(yaml_str)
