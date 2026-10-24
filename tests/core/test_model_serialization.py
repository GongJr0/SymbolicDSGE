from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from SymbolicDSGE.core.model_parser import ModelParser


def _model_signature(parser: ModelParser) -> dict[str, Any]:
    cfg = parser.get()
    return {
        "name": cfg.name,
        "parameters": sorted(s.name for s in cfg.calibration.parameters.keys()),
        "observables": [s.name for s in cfg.observables],
        "shocks": [s.name for s in cfg.shocks],
        "calibration": {
            k.name: float(v) for k, v in cfg.calibration.parameters.items()
        },
    }


def test_model_config_round_trips_without_mnoise(test_model_path) -> None:
    parser = ModelParser(test_model_path)
    rebuilt = ModelParser.from_string(parser.to_yaml())

    assert _model_signature(rebuilt) == _model_signature(parser)
    assert rebuilt.get().calibration.measurement_noise is None


def test_mnoise_config_round_trips_via_reparse(post82_test_model_path) -> None:
    parser = ModelParser(post82_test_model_path)
    rebuilt = ModelParser.from_string(parser.to_yaml())

    k0 = parser.get().calibration.measurement_noise
    k1 = rebuilt.get().calibration.measurement_noise
    assert k0 is not None and k1 is not None

    # Derived R machinery is rebuilt by the parser, not serialized.
    assert k1.param_names == k0.param_names
    assert k1.std == k0.std
    assert k1.corr == k0.corr


def test_to_yaml_bakes_updated_calibration(test_model_path) -> None:
    parser = ModelParser(test_model_path)
    config = parser.get()
    target = next(iter(config.calibration.parameters))
    config.calibration.parameters[target] = np.float64(0.123)

    rebuilt = ModelParser.from_string(parser.to_yaml(config))
    rebuilt_calib = {
        k.name: float(v) for k, v in rebuilt.get().calibration.parameters.items()
    }
    assert rebuilt_calib[target.name] == pytest.approx(0.123)
