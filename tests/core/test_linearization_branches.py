"""Branch coverage for linearization guards."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import sympy as sp

from SymbolicDSGE.core.linearization import (
    LinearizationMethod,
    Linearizer,
    VariableTransformSpec,
)


def test_reconstruct_requires_steady_state():
    spec = VariableTransformSpec(
        original=sp.Function("c"),
        transformed=sp.Function("c_hat"),
        method=LinearizationMethod.LOG,
        steady_state=None,
    )
    with pytest.raises(ValueError, match="missing a steady state"):
        spec.reconstruct(sp.Symbol("z"))


def test_reconstruct_none_method_passthrough():
    spec = VariableTransformSpec(
        original=sp.Function("c"),
        transformed=sp.Function("c_hat"),
        method=LinearizationMethod.NONE,
        steady_state=None,
    )
    z = sp.Symbol("z")
    assert spec.reconstruct(z) == z


def test_linearizer_rejects_already_linearized_config():
    conf = SimpleNamespace(symbolically_linearized=True)
    with pytest.raises(ValueError, match="already symbolically linearized"):
        Linearizer(conf)


def test_linearizer_requires_inputs():
    with pytest.raises(TypeError, match="requires either a ModelConfig"):
        Linearizer()
