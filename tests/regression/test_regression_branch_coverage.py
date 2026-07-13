"""Branch coverage for the ridge/lasso non-native dispatch and intercept paths.

The existing tests exercise the native/scalar path; these force the numba
(non-scalar) branch and the intercept-handling helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.regression.ridge import core as ridge_core
from SymbolicDSGE.regression.lasso import core as lasso_core


def _xy(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(24, 3))
    beta = np.array([1.0, -2.0, 0.5])
    y = X @ beta + rng.normal(scale=0.1, size=24)
    return np.ascontiguousarray(X), np.ascontiguousarray(y)


def test_ridge_non_native_branch(monkeypatch):
    monkeypatch.setattr(ridge_core, "_native_dims_ok", lambda n, p: False)
    X, y = _xy()
    r = ridge_core.ridge(X, y, alpha=1.0, intercept=True)
    assert r.coefficients.shape[0] == 4  # intercept + 3
    gs = ridge_core.ridge_gs(X, y, 0.01, 10.0, 5, intercept=True)
    assert gs.coefficients.shape[0] == 4


def test_lasso_intercept_and_non_native(monkeypatch):
    X, y = _xy(1)
    # intercept path helpers (_center_for_intercept / _restore_intercept_path)
    gs = lasso_core.lasso_gs(X, y, 0.01, 1.0, 5, intercept=True)
    assert gs.coefficients.shape[0] == 4

    # force the numba (non-scalar) dispatch in both entry points
    monkeypatch.setattr(lasso_core, "use_scalar_path", lambda n, k: False)
    r = lasso_core.lasso(X, y, alpha=0.05, intercept=True)
    assert r.coefficients.shape[0] == 4
    gs2 = lasso_core.lasso_gs(X, y, 0.01, 1.0, 5, intercept=True)
    assert gs2.coefficients.shape[0] == 4


def test_ridge_lasso_param_validation():
    X, y = _xy(2)
    with pytest.raises(ValueError, match="non-negative"):
        ridge_core.ridge(X, y, alpha=-1.0)
    with pytest.raises(ValueError, match="positive"):
        ridge_core.ridge_gs(X, y, -1.0, 10.0, 5)
    with pytest.raises(ValueError, match="positive"):
        ridge_core.ridge_gs(X, y, 0.1, 10.0, 0)
    with pytest.raises(ValueError, match="non-negative"):
        lasso_core.lasso(X, y, alpha=-1.0)
    with pytest.raises(ValueError, match="positive"):
        lasso_core.lasso_gs(X, y, -1.0, 1.0, 5)
    with pytest.raises(ValueError, match="positive"):
        lasso_core.lasso_gs(X, y, 0.1, 1.0, 0)
