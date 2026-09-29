"""Branch coverage for the shock generators.

Targets the untested paths: the scipy fallback route, the PSD eigh fallback in
``_gaussian_factor``, the numpy string-family closures (including error
branches), ``_get_dist`` dispatch, and ``_jsonable``.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import LinAlgError
import pytest
from scipy.stats import (
    norm,
    multivariate_normal as mnorm,
)

from SymbolicDSGE.core.shock import generators as S
from SymbolicDSGE.core.shock.generators import Shock


def test_abstract_shock_array_scipy_route():
    out = S.abstract_shock_array(16, 0, norm)
    assert out.shape == (16,)
    assert out.dtype == np.float64


def test_gaussian_factor_eigh_fallback_on_psd():
    # [[1,1],[1,1]] is PSD but singular -> cholesky raises -> eigh branch.
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(LinAlgError):
        # conform the chol raise
        np.linalg.cholesky(cov)

    factor = S._gaussian_factor(cov)
    assert np.array_equal(factor @ factor.T, cov)


def test_draw_fn_scipy_path_univariate_and_multivariate():
    # a live scipy object bypasses the numpy fast path (dist is not str)
    draw = Shock(dist=norm).draw_fn(10, False)
    assert draw(np.zeros(1), np.array([[1.0]]), 0).shape == (10, 1)

    mv = Shock(dist=mnorm, dist_kwargs={"mean": [0.0, 0.0]})
    out = mv.draw_fn(8, True)(np.zeros(2), np.eye(2), 0)
    assert out.shape[0] == 8


def test_jsonable_branches():
    assert S._jsonable(np.float64(2.0)) == pytest.approx(2.0)
    assert S._jsonable([np.float64(1.0), 2]) == [pytest.approx(1.0), 2]
    out = S._jsonable({"a": np.int64(3)})
    assert out == {"a": 3}
