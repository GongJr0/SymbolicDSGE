from __future__ import annotations

import base64
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sympy import Symbol

from SymbolicDSGE.core.config import ModelConfig
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.kalman.config import KalmanConfig

from .schemas import ArrayEnvelope, Role


def encode_array(array: NDArray[Any]) -> dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return {
        "dtype": "float64",
        "shape": list(arr.shape),
        "order": "C",
        "data_b64": base64.b64encode(arr.tobytes(order="C")).decode("ascii"),
    }


def decode_array(envelope: ArrayEnvelope) -> NDArray[np.float64]:
    if envelope.dtype != "float64" or envelope.order != "C":
        raise ValueError("Only C-contiguous float64 array envelopes are supported.")
    raw = base64.b64decode(envelope.data_b64.encode("ascii"))
    out = np.frombuffer(raw, dtype=np.float64).copy()
    expected_size = int(np.prod(envelope.shape, dtype=np.int64))
    if out.size != expected_size:
        raise ValueError(
            f"Array payload size {out.size} does not match shape {envelope.shape}."
        )
    return out.reshape(tuple(envelope.shape))


def encode_named_arrays(values: Mapping[str, NDArray[Any]]) -> list[dict[str, Any]]:
    return [
        {"name": name, "array": encode_array(value)} for name, value in values.items()
    ]


def summarize_parsed_model(
    *,
    role: Role,
    model: ModelConfig,
    kalman: KalmanConfig | None,
    source: str | None,
) -> dict[str, Any]:
    return {
        "role": role,
        "loaded": True,
        "solved": False,
        "source": source,
        "name": model.name,
        "variables": _symbol_names(model.variables.variables),
        "parameters": _symbol_names(model.parameters),
        "parameter_values": _parameter_values(model),
        "observables": _symbol_names(model.observables),
        "shock_specs": _shock_specs(model),
        "shock_corr_specs": _shock_corr_specs(model),
        "has_kalman": kalman is not None,
    }


def summarize_solved_model(
    *,
    role: Role,
    model: SolvedModel,
    source: str | None,
) -> dict[str, Any]:
    compiled = model.compiled
    layout = compiled.layout
    policy_stab = getattr(model.policy, "stab", None)
    return {
        "role": role,
        "loaded": True,
        "solved": True,
        "source": source,
        "name": compiled.config.name,
        "variables": list(compiled.var_names),
        "observables": list(compiled.observable_names),
        "parameters": _symbol_names(compiled.config.parameters),
        "parameter_values": _parameter_values(compiled.config),
        "shock_specs": _shock_specs(compiled.config),
        "shock_corr_specs": _shock_corr_specs(compiled.config),
        "n_state": int(compiled.n_state),
        "n_exog": int(compiled.n_exog),
        "A_shape": list(model.A.shape),
        "B_shape": list(model.B.shape),
        "has_kalman": compiled.kalman is not None,
        "policy": {"stab": _coerce_json_scalar(policy_stab)},
        "layout": {
            "declared_names": list(layout.declared_names),
            "canonical_names": list(layout.canonical_names),
            "exo_state_names": list(layout.exo_state_names),
            "endo_state_names": list(layout.endo_state_names),
            "control_names": list(layout.control_names),
        },
    }


def empty_model_summary(role: Role) -> dict[str, Any]:
    return {"role": role, "loaded": False, "solved": False}


def _symbol_names(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        name = getattr(value, "name", None)
        out.append(str(name if name is not None else value))
    return out


def _shock_specs(model: ModelConfig) -> list[dict[str, Any]]:
    return [
        {
            "shock": str(shock),
            "target": str(target),
            "std_param": str(std_param) if std_param is not None else None,
            "std_value": _param_value(model, std_param),
        }
        for shock, target in model.shock_map.items()
        for std_param in [model.calibration.shock_std.get(shock)]
    ]


def _coerce_json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _shock_corr_specs(model: ModelConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    shock_order = {str(shock): i for i, shock in enumerate(model.shock_map)}
    for pair, param in model.calibration.shock_corr.items():
        if param is None:
            continue
        shocks = sorted(
            (str(shock) for shock in pair), key=lambda name: shock_order.get(name, 0)
        )
        out.append(
            {
                "pair": shocks,
                "key": ",".join(shocks),
                "corr_param": str(param),
                "corr_value": _param_value(model, param),
            }
        )
    return out


def _param_value(model: ModelConfig, param: Any) -> float | None:
    if param is None:
        return None
    if param in model.calibration.parameters:
        return float(model.calibration.parameters[param])
    if isinstance(param, str):
        sym = Symbol(param)
    else:
        sym = Symbol(str(param))
    if sym in model.calibration.parameters:
        return float(model.calibration.parameters[sym])
    return None


def _parameter_values(model: ModelConfig) -> dict[str, float]:
    return {
        str(parameter): float(value)
        for parameter, value in model.calibration.parameters.items()
    }
