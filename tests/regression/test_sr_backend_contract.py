# type: ignore
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sympy as sp
from pysr import ExpressionSpec, TemplateExpressionSpec

from SymbolicDSGE.regression.config import TemplateConfig
from SymbolicDSGE.regression.model_defaults import PySRParams
from SymbolicDSGE.regression.model_parametrizer import ModelParametrizer
from SymbolicDSGE.regression.sr_backend import SymbolicRegressorBackend


@dataclass
class _FakeModel:
    equations_: pd.DataFrame
    set_params_kwargs: dict | None = None
    fit_variable_names: list[str] | None = None

    def set_params(self, **kwargs):
        self.set_params_kwargs = kwargs

    def fit(self, X, y, variable_names):
        self.fit_variable_names = list(variable_names)

    def get_best(self):
        return pd.Series({"equation": "best_expr", "loss": 0.1})


class _FakeBackend(SymbolicRegressorBackend):
    def __init__(self, parametrizer: ModelParametrizer, equations: pd.DataFrame):
        self._equations = equations
        self.loaded_models: list[_FakeModel] = []
        super().__init__(parametrizer)

    def _load_params(self):
        m = _FakeModel(equations_=self._equations.copy())
        self.loaded_models.append(m)
        return m


def _make_parametrizer(
    *,
    varnames: list[str] | None = None,
    config: TemplateConfig | None = None,
    combine: str = "f_1(x, y)",
    expressions: list[str] | None = None,
    clean_expr: str | None = None,
) -> ModelParametrizer:
    v = varnames or ["x", "y"]
    cfg = config or TemplateConfig()
    p = ModelParametrizer(v, PySRParams(precision=32), cfg)
    p.params.expression_spec = TemplateExpressionSpec(
        combine=combine,
        expressions=expressions or ["f_1"],
        variable_names=v,
    )
    p.clean_expr = clean_expr
    return p


def _eq_df(equation: str = "f_1 = #1 + #2") -> pd.DataFrame:
    return pd.DataFrame([{"equation": equation, "loss": 0.1, "complexity": 1}])


def test_fit_rejects_numpy_without_variable_names():
    p = _make_parametrizer()
    b = _FakeBackend(p, _eq_df())
    with pytest.raises(ValueError, match="provide the variable names"):
        b.fit(np.zeros((3, 2)), np.zeros(3))


def test_fit_uses_dataframe_columns_and_param_overrides():
    p = _make_parametrizer()
    b = _FakeBackend(p, _eq_df("f_1 = #1 + #2"))

    X = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    y = pd.Series([1.0, 2.0])
    best = b.fit(X, y, param_overrides={"niterations": 7})

    assert isinstance(best, pd.Series)
    assert b.model.set_params_kwargs == {"niterations": 7}
    assert b.model.fit_variable_names == ["x", "y"]
    assert "sympy_format" in b.model.equations_.columns
    assert str(b.model.equations_.iloc[0]["sympy_format"]) == "x + y"


def test_fit_warns_and_falls_back_when_variable_names_are_invalid():
    p = _make_parametrizer()
    b = _FakeBackend(p, _eq_df())

    X = pd.DataFrame({"x": [1.0], "y": [2.0]})
    y = pd.Series([0.0])
    with pytest.warns(UserWarning, match="Using inferred variable names"):
        b.fit(X, y, variable_names=[SimpleNamespace(name="bad")])  # type: ignore[list-item]
    assert b.model.fit_variable_names == ["x", "y"]


def test_validate_and_normalize_varnames_rejects_mismatch():
    p = _make_parametrizer(varnames=["x", "y"])
    b = _FakeBackend(p, _eq_df())
    with pytest.raises(ValueError, match="do not match the parametrizer"):
        b._validate_and_normalize_varnames(["x", "z"])


def test_get_sp_from_template_returns_input_for_expression_spec():
    p = _make_parametrizer()
    p.params.expression_spec = ExpressionSpec()
    b = _FakeBackend(p, _eq_df())

    row = pd.Series(
        {"equation": "x + y", "sympy_format": sp.Symbol("x") + sp.Symbol("y")}
    )
    out = b._get_sp_from_template(row)
    assert out.equals(row)


def test_get_sp_from_template_parses_placeholders_and_combine():
    p = _make_parametrizer(
        combine="f_1(x, y) + f_2(y)",
        expressions=["f_1", "f_2"],
    )
    b = _FakeBackend(p, _eq_df())
    row = pd.Series({"equation": "f_1 = #1 + #2; f_2 = #1"})

    out = b._get_sp_from_template(row)
    assert str(out["sympy_format"]) == "x + 2*y"


def test_get_sp_from_template_honors_clean_expr_prefix_replacement():
    p = _make_parametrizer(
        varnames=["x"],
        combine="old_term + f_1(x)",
        expressions=["f_1"],
        clean_expr="x",
    )
    b = _FakeBackend(p, _eq_df("f_1 = #1 + 1"))
    row = pd.Series({"equation": "f_1 = #1 + 1"})

    out = b._get_sp_from_template(row)
    assert str(out["sympy_format"]) == "2*x + 1"


def test_get_sp_from_template_raises_for_unparseable_definition_line():
    p = _make_parametrizer()
    b = _FakeBackend(p, _eq_df())
    row = pd.Series({"equation": "this is not parseable"})

    with pytest.raises(ValueError, match="Unparseable template function line"):
        b._get_sp_from_template(row)


def test_get_sp_from_template_raises_when_function_missing_from_combine():
    p = _make_parametrizer(combine="f_1(x)", expressions=["f_1"])
    b = _FakeBackend(p, _eq_df())
    row = pd.Series({"equation": "f_2 = #1 + 1"})

    with pytest.raises(KeyError, match="appears in template output but not in combine"):
        b._get_sp_from_template(row)


def test_convert_and_handle_constants_applies_strategy():
    cfg = TemplateConfig(constant_filtering="parametrize")
    p = _make_parametrizer(
        config=cfg, combine="f_1(x)", expressions=["f_1"], varnames=["x"]
    )
    b = _FakeBackend(p, _eq_df("f_1 = #1 + 2"))

    converted = b._convert_and_handle_constants(_eq_df("f_1 = #1 + 2"))
    sym_expr = converted.iloc[0]["sympy_format"]
    assert len(sym_expr.atoms(sp.Dummy)) == 1
    assert "initial_expr" in converted.columns
