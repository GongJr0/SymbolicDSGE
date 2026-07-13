"""Branch coverage for the shock generators.

Targets the untested paths: the scipy fallback route, the PSD eigh fallback in
``_gaussian_factor``, the numpy string-family closures (including error
branches), ``_get_dist`` dispatch, ``place_shocks`` edges, and ``_jsonable``.
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
    draws = S._draw_normal_mv(64, 0, None, np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert draws.shape == (64, 2)
    # rank-1 covariance: the two columns are perfectly correlated.
    assert np.corrcoef(draws.T)[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_shock_generator_scipy_path_univariate_and_multivariate():
    # a live scipy object bypasses the numpy fast path (dist is not str)
    gen = Shock(dist=norm).shock_generator(10)
    assert gen(1.0).shape == (10,)

    mv = Shock(dist=mnorm, multivar=True, dist_kwargs={"mean": [0.0, 0.0]})
    gen_mv = mv.shock_generator(8)
    out = gen_mv(np.eye(2))
    assert out.shape[0] == 8


def test_numpy_generator_families_and_errors():
    # t without df raises
    with pytest.raises(ValueError, match="require 'df'"):
        Shock(dist="t").shock_generator(5)
    # t univariate closure
    gt = Shock(dist="t", dist_kwargs={"df": 5}).shock_generator(12)
    assert gt(1.0).shape == (12,)
    # t multivariate closure
    gtm = Shock(dist="t", multivar=True, dist_kwargs={"df": 5}).shock_generator(12)
    assert gtm(np.eye(2)).shape == (12, 2)
    # uniform univariate closure
    gu = Shock(dist="uni").shock_generator(9)
    assert gu(2.0).shape == (9,)
    # uniform multivariate is not implemented
    with pytest.raises(NotImplementedError):
        Shock(dist="uni", multivar=True).shock_generator(9)


def test_get_dist_dispatch():
    assert Shock(dist="norm", multivar=True)._get_dist() is mnorm
    assert Shock(dist="t", multivar=False)._get_dist() is scipy_t
    assert Shock(dist="t", multivar=True)._get_dist() is mt
    with pytest.raises(NotImplementedError):
        Shock(dist="uni", multivar=True)._get_dist()
    # a raw scipy object passes straight through
    assert Shock(dist=norm)._get_dist() is norm


def test_place_shocks_empty_and_multivar_bounds():
    # empty spec, univariate -> zeros(T,)
    z = Shock(dist="norm").place_shocks({}, 5)
    assert z.shape == (5,) and np.all(z == 0.0)
    # empty spec, multivar -> zeros(T, 0)
    zm = Shock(dist="norm", multivar=True).place_shocks({}, 5)
    assert zm.shape == (5, 0)
    # multivar with a provided array + valid placement
    arr = np.zeros((4, 2), dtype=np.float64)
    placed = Shock(dist="norm", multivar=True, shock_arr=arr).place_shocks(
        {(1, 0): 3.0}, 4
    )
    assert placed[1, 0] == 3.0
    # multivar dimension out of bounds against the provided array
    with pytest.raises(IndexError, match="out of bounds for K="):
        Shock(dist="norm", multivar=True, shock_arr=arr).place_shocks({(1, 5): 1.0}, 4)


def test_jsonable_branches():
    assert S._jsonable(np.float64(2.0)) == pytest.approx(2.0)
    assert S._jsonable([np.float64(1.0), 2]) == [pytest.approx(1.0), 2]
    out = S._jsonable({"a": np.int64(3)})
    assert out == {"a": 3}
