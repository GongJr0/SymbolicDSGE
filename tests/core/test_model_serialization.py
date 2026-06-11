from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from SymbolicDSGE.core.model_parser import ModelParser


def _model_signature(parser: ModelParser) -> dict[str, Any]:
    cfg = parser.get()
    return {
        "name": cfg.name,
        "parameters": sorted(s.name for s in cfg.parameters),
        "observables": [s.name for s in cfg.observables],
        "shock_map": {k.name: v.name for k, v in cfg.shock_map.items()},
        "calibration": {
            k.name: float(v) for k, v in cfg.calibration.parameters.items()
        },
    }


def test_model_config_round_trips_without_kalman() -> None:
    parser = ModelParser("MODELS/test.yaml")
    rebuilt = ModelParser.from_string(parser.to_yaml())

    assert _model_signature(rebuilt) == _model_signature(parser)
    assert rebuilt.get_all().kalman is None


def test_kalman_config_round_trips_via_reparse() -> None:
    parser = ModelParser("MODELS/POST82.yaml")
    rebuilt = ModelParser.from_string(parser.to_yaml())

    k0 = parser.get_all().kalman
    k1 = rebuilt.get_all().kalman
    assert k0 is not None and k1 is not None

    # Authored fields survive verbatim.
    assert k1.y_names == k0.y_names
    assert k1.jitter == k0.jitter
    assert k1.symmetrize == k0.symmetrize
    assert k1.P0.mode == k0.P0.mode
    assert k1.P0.scale == k0.P0.scale
    assert k1.P0.diag == k0.P0.diag

    # Derived R machinery is rebuilt by the parser, not serialized.
    assert k1.R_param_names == k0.R_param_names
    assert k1.R_std_param_map == k0.R_std_param_map
    assert k1.R_corr_param_map == k0.R_corr_param_map
    assert (k0.R is None) == (k1.R is None)
    if k0.R is not None:
        np.testing.assert_allclose(np.asarray(k1.R), np.asarray(k0.R))


def test_to_yaml_bakes_updated_calibration() -> None:
    parser = ModelParser("MODELS/test.yaml")
    config = parser.get()
    target = next(iter(config.calibration.parameters))
    config.calibration.parameters[target] = np.float64(0.123)

    rebuilt = ModelParser.from_string(parser.to_yaml(config))
    rebuilt_calib = {
        k.name: float(v) for k, v in rebuilt.get().calibration.parameters.items()
    }
    assert rebuilt_calib[target.name] == pytest.approx(0.123)


def test_to_yaml_emits_kalman_block_when_present() -> None:
    parser = ModelParser("MODELS/POST82.yaml")
    text = parser.to_yaml()
    assert "kalman:" in text

    text_no_kalman = ModelParser("MODELS/test.yaml").to_yaml()
    assert "kalman:" not in text_no_kalman
