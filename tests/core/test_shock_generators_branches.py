"""Branch coverage for the shock generators.

Targets the untested paths: the scipy fallback route, the PSD eigh fallback in
``_gaussian_factor``, the numpy string-family closures (including error
branches), ``_get_dist`` dispatch, and ``_jsonable``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import (
    norm,
    multivariate_normal as mnorm,
    t as scipy_t,
    multivariate_t as mt,
)

from SymbolicDSGE.core import shock_generators as S
from SymbolicDSGE.core.shock_generators import Shock


def test_abstract_shock_array_scipy_route():
    out = S.abstract_shock_array(16, 0, norm)
    assert out.shape == (16,)
    assert out.dtype == np.float64


def test_gaussian_factor_eigh_fallback_on_psd():
    # [[1,1],[1,1]] is PSD but singular -> cholesky raises -> eigh branch.
    factor = S._gaussian_factor(np.array([[1.0, 1.0], [1.0, 1.0]]))
    draws = S._draw_normal(64, 0, np.zeros(2), factor)
    assert draws.shape == (64, 2)
    # rank-1 covariance: the two columns are perfectly correlated.
    assert np.corrcoef(draws.T)[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_draw_fn_scipy_path_univariate_and_multivariate():
    # a live scipy object bypasses the numpy fast path (dist is not str)
    draw = Shock(dist=norm).draw_fn(10, False)
    assert draw(np.zeros(1), np.array([[1.0]]), 0).shape == (10, 1)

    mv = Shock(dist=mnorm, dist_kwargs={"mean": [0.0, 0.0]})
    out = mv.draw_fn(8, True)(np.zeros(2), np.eye(2), 0)
    assert out.shape[0] == 8


def test_numpy_draw_families_and_errors():
    # t without df raises
    with pytest.raises(ValueError, match="require 'df'"):
        Shock(dist="t").draw_fn(5, False)
    # t univariate closure
    gt = Shock(dist="t", dist_kwargs={"df": 5}).draw_fn(12, False)
    assert gt(np.zeros(1), np.array([[1.0]]), 0).shape == (12, 1)
    # t multivariate closure
    gtm = Shock(dist="t", dist_kwargs={"df": 5}).draw_fn(12, True)
    assert gtm(np.zeros(2), np.eye(2), 0).shape == (12, 2)
    # uniform univariate closure
    gu = Shock(dist="uni").draw_fn(9, False)
    assert gu(np.zeros(1), np.array([[2.0]]), 0).shape == (9, 1)
    # uniform multivariate is not implemented
    with pytest.raises(NotImplementedError):
        Shock(dist="uni").draw_fn(9, True)


def test_get_dist_dispatch():
    assert Shock(dist=norm)._get_dist(False) is norm


def test_jsonable_branches():
    assert S._jsonable(np.float64(2.0)) == pytest.approx(2.0)
    assert S._jsonable([np.float64(1.0), 2]) == [pytest.approx(1.0), 2]
    out = S._jsonable({"a": np.int64(3)})
    assert out == {"a": 3}
