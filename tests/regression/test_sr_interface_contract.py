# type: ignore
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sympy as sp

from SymbolicDSGE.kalman.filter import FilterResult
from SymbolicDSGE.regression.config import TemplateConfig
from SymbolicDSGE.regression.model_defaults import PySRParams
from SymbolicDSGE.regression.model_parametrizer import ModelParametrizer
from SymbolicDSGE.regression.fit_result import FitResult
from SymbolicDSGE.regression.sr_interface import SRInterface
from SymbolicDSGE.regression.symbolic_regression import SymbolicRegressor


class _BuiltinParametrizer(ModelParametrizer):
    def __init__(self, variable_names, params, config):
        super().__init__(variable_names, params, config)
        self.add_built_in_ops(["sqrt"])


def _make_filter_result() -> FilterResult:
    x_pred = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float64)
    zeros_state = np.zeros((3, 2, 2), dtype=np.float64)
    y_pred = np.array([[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]], dtype=np.float64)
    innov = np.array([[7.0, 70.0], [8.0, 80.0], [9.0, 90.0]], dtype=np.float64)
    return FilterResult(
        x_pred=x_pred,
        x_filt=x_pred.copy(),
        P_pred=zeros_state,
        P_filt=zeros_state.copy(),
        y_pred=y_pred,
        y_filt=y_pred.copy(),
        innov=innov,
        S=zeros_state.copy(),
        loglik=np.float64(0.0),
    )


def _make_interface(
    monkeypatch,
    *,
    include_expression: bool = False,
    obs_is_affine: dict[str, bool] | None = None,
):
    monkeypatch.setattr(SymbolicRegressor, "_load_params", lambda self: object())

    t = sp.Symbol("t", integer=True)
    x = sp.Function("x")
    beta = sp.Symbol("beta")
    calls = {}
    filter_result = _make_filter_result()

    if obs_is_affine is None:
        obs_is_affine = {"pi": True, "y": True}

    def kalman(**kwargs):
        calls["kwargs"] = kwargs
        return filter_result

    parametrizer = _BuiltinParametrizer(
        ["x"],
        PySRParams(precision=32),
        TemplateConfig(include_expression=include_expression),
    )
    model = SimpleNamespace(
        kalman=kalman,
        compiled=SimpleNamespace(
            idx={"pi_state": 0, "x": 1},
            observable_names=["pi", "y"],
            var_names=["pi_state", "x"],
        ),
        config=SimpleNamespace(
            equations=SimpleNamespace(
                observable={"y": x(t) + beta, "pi": x(t)},
                obs_is_affine=obs_is_affine,
            ),
            calibration=SimpleNamespace(parameters={beta: 2.0}),
        ),
    )

    interface = SRInterface(
        model=model,
        obs_name="y",
        parametrizer=parametrizer,
    )
    return interface, calls, filter_result, x, t


def test_sr_interface_uses_prebuilt_parametrizer(monkeypatch):
    monkeypatch.setattr(SymbolicRegressor, "_load_params", lambda self: object())

    t = sp.Symbol("t", integer=True)
    x = sp.Function("x")
    parametrizer = _BuiltinParametrizer(
        ["x"],
        PySRParams(precision=32),
        TemplateConfig(),
    )
    model = SimpleNamespace(
        compiled=SimpleNamespace(var_names=["x"]),
        config=SimpleNamespace(
            equations=SimpleNamespace(observable={"y": x(t) + 1}),
            calibration=SimpleNamespace(parameters={}),
        ),
    )

    interface = SRInterface(
        model=model,
        obs_name="y",
        parametrizer=parametrizer,
    )

    assert interface.sr.parametrizer is parametrizer
    assert interface.selected_var_names == ["x"]
    assert interface.sr.parametrizer.params.expression_spec is not None
    assert any("ssqrt" in op for op in (parametrizer.params.unary_operators or []))


def test_get_kf_selects_linear_or_extended_mode(monkeypatch):
    linear_interface, linear_calls, _, _, _ = _make_interface(
        monkeypatch,
        obs_is_affine={"pi": True, "y": True},
    )
    linear_out = linear_interface.get_kf(np.zeros((3, 2), dtype=np.float64))

    assert linear_out is not None
    assert linear_calls["kwargs"]["filter_mode"] == "linear"
    assert linear_calls["kwargs"]["observables"] == ["pi", "y"]
    assert linear_calls["kwargs"]["estimate_R_diag"] is False
    assert linear_calls["kwargs"]["return_shocks"] is False
    assert linear_calls["kwargs"]["_debug"] is False

    extended_interface, extended_calls, _, _, _ = _make_interface(
        monkeypatch,
        obs_is_affine={"pi": True, "y": False},
    )
    extended_interface.get_kf(pd.DataFrame([[0.0, 0.0]], columns=["pi", "y"]))

    assert extended_calls["kwargs"]["filter_mode"] == "extended"


def test_sr_interface_substitutes_observable_equation(monkeypatch):
    interface, _, _, x, t = _make_interface(monkeypatch)

    assert sp.simplify(interface._get_equation("y") - (x(t) + 2.0)) == 0
    assert interface.obs_idx == {"pi": 0, "y": 1}
    assert interface.selected_obs_name == "y"
    assert interface.selected_var_names == ["x"]


@pytest.mark.parametrize(
    ("include_expression", "expected_column"),
    [(False, np.array([70.0, 80.0, 90.0])), (True, np.array([40.0, 50.0, 60.0]))],
)
def test_fit_to_kf_uses_expected_target(
    monkeypatch, include_expression, expected_column
):
    captured = {}
    expressions = pd.DataFrame({"sympy_format": ["x"], "score": [1.0]})
    best = pd.Series({"loss": 0.1})

    def fake_fit(self, X, y, variable_names):
        captured["X"] = X
        captured["y"] = y
        captured["variable_names"] = variable_names
        self.model = SimpleNamespace(equations_=expressions)
        return best

    monkeypatch.setattr(SymbolicRegressor, "_load_params", lambda self: object())
    monkeypatch.setattr(SymbolicRegressor, "fit", fake_fit)

    interface, _, filter_result, _, _ = _make_interface(
        monkeypatch,
        include_expression=include_expression,
    )
    out = interface.fit_to_kf(np.zeros((3, 2), dtype=np.float64))

    assert isinstance(out, FitResult)
    assert out.expressions.equals(expressions)
    assert out.best.equals(best)
    assert np.array_equal(captured["X"], filter_result.x_pred[:, [1]])
    assert np.array_equal(captured["y"], expected_column)
    assert captured["variable_names"] == ["x"]
