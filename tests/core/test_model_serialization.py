from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from SymbolicDSGE.core import ModelConfig, parse_config
from SymbolicDSGE.core.config_emitter import emit_config_yaml
from SymbolicDSGE.core.model_parser import ModelParser


def _model_signature(cfg: ModelConfig) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "parameters": sorted(s.name for s in cfg.calibration.parameters.keys()),
        "observables": [s.name for s in cfg.observables],
        "shocks": [s.name for s in cfg.shocks],
        "calibration": {
            k.name: float(v) for k, v in cfg.calibration.parameters.items()
        },
    }


def _reparse(cfg: ModelConfig) -> ModelConfig:
    return ModelParser.from_string(emit_config_yaml(cfg)).get()


def test_model_config_round_trips_without_mnoise(test_model_path) -> None:
    config = parse_config(test_model_path)
    rebuilt = _reparse(config)

    assert _model_signature(rebuilt) == _model_signature(config)
    assert rebuilt.calibration.measurement_noise is None


def test_mnoise_config_round_trips_via_reparse(post82_test_model_path) -> None:
    config = parse_config(post82_test_model_path)
    rebuilt = _reparse(config)

    k0 = config.calibration.measurement_noise
    k1 = rebuilt.calibration.measurement_noise
    assert k0 is not None and k1 is not None

    # Derived R machinery is rebuilt by the parser, not serialized.
    assert k1.param_names == k0.param_names
    assert k1.std == k0.std
    assert k1.corr == k0.corr


def test_emission_carries_an_in_memory_calibration_edit(test_model_path) -> None:
    config = parse_config(test_model_path)
    target = next(iter(config.calibration.parameters))
    config.calibration.parameters[target] = np.float64(0.123)

    rebuilt_calib = {
        k.name: float(v) for k, v in _reparse(config).calibration.parameters.items()
    }
    assert rebuilt_calib[target.name] == pytest.approx(0.123)


def test_emission_is_idempotent(test_model_path) -> None:
    config = parse_config(test_model_path)
    once = emit_config_yaml(config)
    assert emit_config_yaml(_reparse(config)) == once
