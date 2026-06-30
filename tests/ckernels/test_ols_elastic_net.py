"""Parity tests: native OLS + elastic-net kernels vs the numba reference.

OLS (``chol_solve``) and the elastic-net kernels are plain ``@njit`` (no
fastmath), so the strict-IEEE C is bit-parity with numba up to libm. Inputs stay
in the small/medium regime (where production dispatch routes to native).
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.regression import (
    elastic_net_active_dof as native_en_dof,
    elastic_net_gram_cd as native_en_cd,
    elastic_net_gram_cd_path as native_en_path,
    ols_chol_solve as native_ols,
)
from SymbolicDSGE.regression.solvers import chol_solve as numba_chol_solve
from SymbolicDSGE.regression.elastic_net.core import (
    elastic_net_active_dof as numba_en_dof,
    elastic_net_gram_cd as numba_en_cd,
    elastic_net_gram_cd_path as numba_en_path,
    split_penalty,
)
from SymbolicDSGE.regression.utils import log_grid

RTOL = 1e-9
ATOL = 1e-10


def _design(n, p, seed, intercept):
    rng = np.random.default_rng(seed)
    cols = p - 1 if intercept else p
    x = rng.normal(size=(n, cols))
    X = np.ascontiguousarray(
        np.hstack([np.ones((n, 1)), x]) if intercept else x, dtype=np.float64
    )
    beta = rng.normal(size=p)
    y = np.ascontiguousarray(X @ beta + 0.3 * rng.normal(size=n), dtype=np.float64)
    return X, y


def _scaled_gram(n, k, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    beta = np.zeros(k)
    beta[: max(1, k // 2)] = rng.normal(size=max(1, k // 2))
    y = X @ beta + 0.25 * rng.normal(size=n)
    Xc = X - X.mean(0)
    yc = y - y.mean()
    G = np.ascontiguousarray((Xc.T @ Xc) / n, dtype=np.float64)
    g = np.ascontiguousarray((Xc.T @ yc) / n, dtype=np.float64)
    return G, g


# --- OLS ---------------------------------------------------------------------


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("p", [1, 2, 5, 9])
def test_ols_chol_solve_parity(intercept, p):
    X, y = _design(50, p, seed=p * 13 + intercept, intercept=intercept)
    n_coef, n_L, n_status = native_ols(X, y)
    b_coef, b_L, b_status = numba_chol_solve(X, y)
    assert int(n_status) == int(b_status) == 0
    np.testing.assert_allclose(n_coef, b_coef, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(n_L, b_L, rtol=RTOL, atol=ATOL)


def test_ols_chol_solve_rank_deficient_matches():
    n = 20
    base = np.random.default_rng(0).normal(size=(n, 1))
    X = np.ascontiguousarray(np.hstack([base, base]), dtype=np.float64)
    y = np.ascontiguousarray(np.random.default_rng(1).normal(size=n), dtype=np.float64)
    n_coef, n_L, n_status = native_ols(X, y)
    b_coef, b_L, b_status = numba_chol_solve(X, y)
    assert int(n_status) == int(b_status) == -1
    assert np.all(np.isnan(n_coef)) and np.all(np.isnan(b_coef))
    assert n_L.shape == b_L.shape == (0, 0)


# --- Elastic net -------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 5, 11])
@pytest.mark.parametrize("alpha,l1_ratio", [(0.02, 0.5), (0.1, 0.3), (0.5, 0.8)])
def test_en_gram_cd_and_dof_parity(k, alpha, l1_ratio):
    G, g = _scaled_gram(60, k, seed=k * 23 + int(alpha * 100))
    a1, a2 = split_penalty(np.float64(alpha), np.float64(l1_ratio))
    beta0 = np.zeros(k, dtype=np.float64)

    n_coef, n_status = native_en_cd(G, g, float(a1), float(a2), beta0, 1000, 1e-10)
    b_coef, b_status = numba_en_cd(G, g, a1, a2, beta0, 1000, np.float64(1e-10))
    assert int(n_status) == int(b_status) == 0
    np.testing.assert_allclose(n_coef, b_coef, rtol=RTOL, atol=ATOL)

    n_dof = native_en_dof(G, n_coef, float(a2), True, 1e-10)
    b_dof = numba_en_dof(G, b_coef, a2, True, np.float64(1e-10))
    np.testing.assert_allclose(float(n_dof), float(b_dof), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("k", [1, 3, 7])
@pytest.mark.parametrize("l1_ratio", [0.25, 0.6, 0.9])
def test_en_gram_cd_path_parity(k, l1_ratio):
    G, g = _scaled_gram(75, k, seed=k * 5 + int(l1_ratio * 10))
    grid = np.ascontiguousarray(log_grid(np.float64(1e-3), np.float64(1.0), 9))

    n_coefs, n_status = native_en_path(G, g, grid, float(l1_ratio), 1000, 1e-10)
    b_coefs, b_status = numba_en_path(
        G, g, grid, np.float64(l1_ratio), 1000, np.float64(1e-10)
    )

    np.testing.assert_array_equal(n_status, b_status)
    np.testing.assert_allclose(n_coefs, b_coefs, rtol=RTOL, atol=ATOL)
