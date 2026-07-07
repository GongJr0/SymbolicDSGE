# type: ignore
from types import SimpleNamespace

import numpy as np
import pytest

import SymbolicDSGE.core.solved_model as solved_model_module
from SymbolicDSGE.core.solved_model import SolvedModel


def _make_solved_model() -> SolvedModel:
    return SolvedModel(
        compiled=SimpleNamespace(var_names=["x"]),
        policy=SimpleNamespace(order=1),
        A=np.zeros((1, 1), dtype=np.float64),
        B=np.zeros((1, 1), dtype=np.float64),
    )


class _FakeTemplateConfig:
    pass


class _FakePySRParams:
    pass


class _FakeParametrizer:
    def __init__(self, variable_names, params=None, config=None):
        self.variable_names = list(variable_names)
        self.params = params
        self.config = config


def test_fit_kf_accepts_prebuilt_parametrizer(monkeypatch):
    captured = {}

    class _FakeSRInterface:
        def __init__(self, *, model, obs_name, parametrizer):
            captured["model"] = model
            captured["obs_name"] = obs_name
            captured["parametrizer"] = parametrizer

        def fit_to_kf(self, y):
            captured["y"] = y
            return "fit-result"

    monkeypatch.setattr(
        solved_model_module,
        "_load_sr_fit_dependencies",
        lambda: (_FakeParametrizer, _FakeSRInterface),
    )

    solved = _make_solved_model()
    parametrizer = _FakeParametrizer(["x"])
    y = np.zeros((4, 1), dtype=np.float64)

    out = solved.fit_kf(
        y=y,
        observable="obs",
        parametrizer=parametrizer,
    )

    assert out == "fit-result"
    assert captured["model"] is solved
    assert captured["obs_name"] == "obs"
    assert captured["parametrizer"] is parametrizer
    assert captured["y"] is y


def test_fit_kf_rejects_mismatched_variables_for_prebuilt_parametrizer():
    solved = _make_solved_model()
    parametrizer = _FakeParametrizer(["x"])

    with pytest.raises(ValueError, match="do not match the parametrizer"):
        solved.fit_kf(
            y=np.zeros((2, 1), dtype=np.float64),
            observable="obs",
            variables=["z"],
            parametrizer=parametrizer,
        )


def test_fit_kf_requires_template_inputs_when_parametrizer_is_missing():
    solved = _make_solved_model()

    with pytest.raises(ValueError, match="Provide either a pre-built parametrizer"):
        solved.fit_kf(
            y=np.zeros((2, 1), dtype=np.float64),
            observable="obs",
        )


def test_fit_kf_builds_parametrizer_when_not_provided(monkeypatch):
    captured = {}

    class _CapturingParametrizer:
        def __init__(self, variable_names, params, config):
            captured["variable_names"] = variable_names
            captured["params"] = params
            captured["config"] = config
            self.variable_names = list(variable_names)

    class _FakeSRInterface:
        def __init__(self, *, model, obs_name, parametrizer):
            captured["model"] = model
            captured["obs_name"] = obs_name
            captured["parametrizer"] = parametrizer

        def fit_to_kf(self, y):
            captured["y"] = y
            return "fit-result"

    monkeypatch.setattr(
        solved_model_module,
        "_load_sr_fit_dependencies",
        lambda: (_CapturingParametrizer, _FakeSRInterface),
    )

    solved = _make_solved_model()
    template_config = _FakeTemplateConfig()
    sr_params = _FakePySRParams()
    y = np.zeros((3, 1), dtype=np.float64)

    out = solved.fit_kf(
        y=y,
        observable="obs",
        template_config=template_config,
        sr_params=sr_params,
    )

    assert out == "fit-result"
    assert captured["variable_names"] == ["x"]
    assert captured["params"] is sr_params
    assert captured["config"] is template_config
    assert captured["model"] is solved
    assert captured["obs_name"] == "obs"
    assert captured["y"] is y
