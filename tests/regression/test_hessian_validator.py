# type: ignore
from types import SimpleNamespace

import sympy as sp

from SymbolicDSGE.regression.config import TemplateConfig
from SymbolicDSGE.regression.hessian_validator import HessianValidator


def _validator(mode: str) -> HessianValidator:
    cfg = TemplateConfig(hessian_restriction=mode)  # type: ignore[arg-type]
    parametrizer = SimpleNamespace(config=cfg)
    return HessianValidator(parametrizer=parametrizer)  # type: ignore[arg-type]


def test_hessian_returns_zero_matrix_for_constant():
    h = HessianValidator._hessian(sp.Integer(3))
    assert h == sp.Matrix([[0]])


def test_hessian_ignores_dummy_symbols():
    x = sp.Symbol("x")
    c = sp.Dummy("C_1")
    h = HessianValidator._hessian(x + c)
    assert h == sp.Matrix([[0]])


def test_diag_mode_allows_cross_terms_and_rejects_quadratic_terms():
    x, y = sp.Symbol("x"), sp.Symbol("y")
    v = _validator("diag")
    assert v.hessian_compliant(x * y)
    assert not v.hessian_compliant(x**2)


def test_full_mode_only_allows_affine_expressions():
    x, y = sp.Symbol("x"), sp.Symbol("y")
    v = _validator("full")
    assert v.hessian_compliant(x + y + 1)
    assert not v.hessian_compliant(x * y)


def test_free_mode_accepts_any_expression():
    x, y = sp.Symbol("x"), sp.Symbol("y")
    v = _validator("free")
    assert v.hessian_compliant(x * y + x**2 + y**3)
