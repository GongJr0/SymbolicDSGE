from __future__ import annotations

import base64
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from SymbolicDSGE.core.config import ModelConfig
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.kalman.config import KalmanConfig

from .schemas import ArrayEnvelope


def encode_array(array: NDArray[Any]) -> ArrayEnvelope:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return {
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(arr.tobytes(order="C")).decode("ascii"),
    }


def decode_array(envelope: ArrayEnvelope) -> NDArray[np.float64]:
    raw = base64.b64decode(envelope["data_b64"].encode("ascii"))
    out = np.frombuffer(raw, dtype=np.float64).copy()
    expected_size = int(np.prod(envelope["shape"], dtype=np.int64))
    if out.size != expected_size:
        raise ValueError(
            f"Array payload size {out.size} does not match shape {envelope['shape']}."
        )
    return out.reshape(tuple(envelope["shape"]))


def encode_named_arrays(values: Mapping[str, NDArray[Any]]) -> list[dict[str, Any]]:
    return [
        {"name": name, "array": encode_array(value)} for name, value in values.items()
    ]


def summarize_parsed_model(
    *,
    model_name: str,
    model: ModelConfig,
    kalman: KalmanConfig | None,
    source: str | None,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "loaded": True,
        "solved": False,
        "source": source,
        "name": model.name,
        "variables": _symbol_names(model.variables.variables),
        "parameters": _symbol_names(model.parameters),
        "parameter_values": _parameter_values(model),
        "observables": _symbol_names(model.observables),
        "shocks": _symbol_names(model.shocks),
        "has_kalman": kalman is not None,
    }


def summarize_solved_model(
    *,
    model_name: str,
    model: SolvedModel,
    source: str | None,
) -> dict[str, Any]:
    compiled = model.compiled
    layout = compiled.layout
    policy = model.policy
    return {
        "model_name": model_name,
        "loaded": True,
        "solved": True,
        "source": source,
        "name": compiled.config.name,
        "variables": list(compiled.var_names),
        "observables": list(compiled.observable_names),
        "parameters": _symbol_names(compiled.config.parameters),
        "parameter_values": _parameter_values(compiled.config),
        "shocks": _symbol_names(compiled.config.shocks),
        "n_state": int(compiled.n_state),
        "n_exog": int(compiled.n_exog),
        "A_shape": list(policy.A.shape),
        "B_shape": list(policy.B.shape),
        "has_kalman": compiled.kalman is not None,
        "policy": {"stab": _coerce_json_scalar(policy.stab)},
        "layout": {
            "declared_names": list(layout.declared_names),
            "canonical_names": list(layout.canonical_names),
            "state_names": list(layout.state_names),
            "control_names": list(layout.control_names),
        },
    }


def empty_model_summary(model_name: str) -> dict[str, Any]:
    return {"model_name": model_name, "loaded": False, "solved": False}


def _symbol_names(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        name = getattr(value, "name", None)
        out.append(str(name if name is not None else value))
    return out


def _coerce_json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parameter_values(model: ModelConfig) -> dict[str, float]:
    return {
        str(parameter): float(value)
        for parameter, value in model.calibration.parameters.items()
    }
