# type: ignore
import pandas as pd
import sympy as sp

from SymbolicDSGE.regression.config import TemplateConfig
from SymbolicDSGE.regression.model_defaults import PySRParams
from SymbolicDSGE.regression.model_parametrizer import ModelParametrizer
from SymbolicDSGE.regression.symbolic_regression import SymbolicRegressor


class _FakeRegressor(SymbolicRegressor):
    def _load_params(self):
        # Prevent constructing real pysr.PySRRegressor in tests.
        return object()


def _make_regressor() -> _FakeRegressor:
    cfg = TemplateConfig(hessian_restriction="diag")
    p = ModelParametrizer(["x", "y"], PySRParams(precision=32), cfg)
    return _FakeRegressor(p)


def test_classify_expressions_partitions_by_hessian_compliance_and_ranks_top():
    x, y = sp.Symbol("x"), sp.Symbol("y")
    exprs = pd.DataFrame(
        {
            "loss": [0.1, 0.2, 0.3],
            "complexity": [1, 3, 2],
            "sympy_format": [x * y, x**2, x + y],
        },
        index=[10, 11, 12],
    )

    reg = _make_regressor()
    out = reg.classify_expressions(exprs)

    # diag mode: x*y and x+y are compliant; x**2 is not.
    assert set(out.qualified_expressions.index) == {10, 12}
    assert set(out.disqualified_expressions.index) == {11}
    assert set(out.top_candidates.index).issubset({10, 12})
    assert len(out.top_candidates) <= 5
    assert "total" in out.top_candidates.columns
