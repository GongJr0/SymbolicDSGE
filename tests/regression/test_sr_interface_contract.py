# type: ignore
from types import SimpleNamespace

import sympy as sp

from SymbolicDSGE.regression.config import TemplateConfig
from SymbolicDSGE.regression.model_defaults import PySRParams
from SymbolicDSGE.regression.model_parametrizer import ModelParametrizer
from SymbolicDSGE.regression.sr_interface import SRInterface
from SymbolicDSGE.regression.symbolic_regression import SymbolicRegressor


class _BuiltinParametrizer(ModelParametrizer):
    def __init__(self, variable_names, params, config):
        super().__init__(variable_names, params, config)
        self.add_built_in_ops(["sqrt"])


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
