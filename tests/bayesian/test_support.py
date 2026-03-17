# type: ignore
from __future__ import annotations

import numpy as np
import pytest
from numpy import float64

from SymbolicDSGE.bayesian.support import (
    OutOfSupportError,
    Support,
    UnsetSupportError,
    _at_boundary_scalar,
    _at_boundary_vectorized,
    _contains_scalar,
    _contains_vectorized,
    bounded,
)


def test_support_contains_scalar_and_vector_respects_inclusive_bounds():
    support = Support(float64(0.0), float64(1.0))

    assert support.contains(float64(0.0))
    assert support.contains(float64(1.0))
    assert support.contains(np.array([0.0, 0.5, 1.0], dtype=np.float64))
    assert not support.contains(float64(-0.1))
    assert not support.contains(np.array([0.5, 1.1], dtype=np.float64))


def test_support_contains_scalar_and_vector_respects_exclusive_bounds():
    support = Support(
        float64(0.0),
        float64(1.0),
        low_inclusive=False,
        high_inclusive=False,
    )

    assert support.contains(float64(0.5))
    assert not support.contains(float64(0.0))
    assert not support.contains(float64(1.0))
    assert not support.contains(np.array([0.25, 1.0], dtype=np.float64))


def test_support_at_boundary_and_metadata_helpers():
    support = Support(float64(-2.0), float64(3.0))

    assert support.at_boundary(float64(-2.0), "low")
    assert support.at_boundary(float64(3.0), "high")
    assert support.at_boundary(np.array([-2.0, 0.0], dtype=np.float64), "low")
    assert support.at_boundary(np.array([0.0, 3.0], dtype=np.float64), "high")
    assert support.is_finite
    assert (support << Support(float64(-1.0), float64(2.0))) is True
    assert (Support(float64(-1.0), float64(2.0)) << support) is False
    assert (Support(float64(-1.0), float64(2.0))).__rlshift__(support) is True


def test_support_equality_rejects_non_support_operands():
    support = Support(float64(0.0), float64(1.0))

    assert support == Support(float64(0.0), float64(1.0))
    assert support != Support(float64(0.0), float64(2.0))
    with pytest.raises(NotImplementedError):
        _ = support == object()


class _BoundedDummy:
    def __init__(self):
        self.support = Support(float64(0.0), float64(1.0))
        self.maps_to = Support(float64(-1.0), float64(1.0))

    @bounded
    def on_support(self, x):
        return x

    @bounded(domain="maps_to")
    def on_maps_to(self, x):
        return x


class _UnsetDummy:
    @bounded
    def f(self, x):
        return x


def test_bounded_decorator_uses_support_and_maps_to_domains():
    dummy = _BoundedDummy()

    assert dummy.on_support(float64(0.4)) == pytest.approx(0.4)
    assert dummy.on_maps_to(float64(-0.2)) == pytest.approx(-0.2)

    with pytest.raises(OutOfSupportError, match="out of support"):
        dummy.on_support(float64(1.5))
    with pytest.raises(OutOfSupportError, match="out of support"):
        dummy.on_maps_to(float64(2.0))


def test_bounded_decorator_raises_when_target_support_is_missing():
    with pytest.raises(UnsetSupportError, match="without a support function"):
        _UnsetDummy().f(float64(0.0))


def test_support_numba_helpers_python_paths_cover_inclusive_exclusive_cases():
    contains_scalar = _contains_scalar.py_func
    contains_vectorized = _contains_vectorized.py_func
    at_boundary_scalar = _at_boundary_scalar.py_func
    at_boundary_vectorized = _at_boundary_vectorized.py_func

    assert contains_scalar(float64(0.0), float64(1.0), True, True, float64(0.0))
    assert not contains_scalar(float64(0.0), float64(1.0), False, True, float64(0.0))
    assert not contains_scalar(float64(0.0), float64(1.0), True, False, float64(1.0))
    assert contains_scalar(float64(0.0), float64(1.0), False, False, float64(0.5))

    vec = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    assert contains_vectorized(float64(0.0), float64(1.0), True, True, vec)
    assert not contains_vectorized(
        float64(0.0),
        float64(1.0),
        False,
        True,
        np.array([0.0, 0.2], dtype=np.float64),
    )
    assert not contains_vectorized(
        float64(0.0),
        float64(1.0),
        True,
        False,
        np.array([0.2, 1.0], dtype=np.float64),
    )

    assert at_boundary_scalar(float64(0.0), "low", float64(0.0))
    assert at_boundary_scalar(float64(1.0), "high", float64(1.0))
    assert at_boundary_vectorized(
        np.array([0.0, 0.4], dtype=np.float64), "low", float64(0.0)
    )
    assert at_boundary_vectorized(
        np.array([0.2, 1.0], dtype=np.float64), "high", float64(1.0)
    )
