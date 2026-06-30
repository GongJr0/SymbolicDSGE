"""Parity tests: native ridge kernels vs the numba reference.

The native C kernels in ``_ckernels/regression`` must match the compiled numba
helpers (``chol_solve_L2`` in regression.solvers, ``l2_grid_search`` in
regression.ridge.core) up to libm. The compiled njit kernels are the oracle.
Inputs are kept in the small/medium regime where numba's xtx_xty stays on its
manual branch (n < 1e5, p < 100) -- that is exactly where the native gram is
bit-parity, and where production dispatch routes to native.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.regression import (
    chol_solve_L2 as native_chol_solve_L2,
    ridge_grid_search as native_ridge_grid_search,
)
from SymbolicDSGE.regression.solvers import chol_solve_L2 as numba_chol_solve_L2
from SymbolicDSGE.regression.ridge.core import l2_grid_search as numba_grid_search
from SymbolicDSGE.regression.utils import get_criterion, log_grid

RTOL = 1e-10
ATOL = 1e-12

_CRITERION_CODES = {"aic": 1, "bic": 2, "loss": 3}


def _design(
    n: int, p: int, seed: int, intercept: bool
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cols = p - 1 if intercept else p
    x = rng.normal(size=(n, cols))
    X = np.ascontiguousarray(
        np.hstack([np.ones((n, 1)), x]) if intercept else x, dtype=np.float64
    )
    beta = rng.normal(size=p)
    y = np.ascontiguousarray(X @ beta + 0.3 * rng.normal(size=n), dtype=np.float64)
    return X, y


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("p", [1, 2, 4, 8])
@pytest.mark.parametrize("alpha", [0.0, 1e-4, 0.1, 1.0, 25.0])
def test_chol_solve_L2_parity(intercept, p, alpha):
    n = 40
    X, y = _design(n, p, seed=p * 100 + int(alpha), intercept=intercept)

    n_coef, n_L, n_dof, n_status = native_chol_solve_L2(X, y, float(alpha), intercept)
    b_coef, b_L, b_dof, b_status = numba_chol_solve_L2(
        X, y, np.float64(alpha), intercept
    )

    assert int(n_status) == int(b_status) == 0
    np.testing.assert_allclose(n_coef, b_coef, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(n_L, b_L, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(float(n_dof), float(b_dof), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("p", [120, 200])
def test_chol_solve_L2_wide_band_parity(p):
    # p in (100, 256]: the consolidated predicate keeps this on the manual/native
    # path (the old gate excluded p >= 100). Native and numba must still agree.
    n = 3 * p
    X, y = _design(n, p, seed=p, intercept=True)
    n_coef, _, n_dof, n_status = native_chol_solve_L2(X, y, 0.5, True)
    b_coef, _, b_dof, b_status = numba_chol_solve_L2(X, y, np.float64(0.5), True)
    assert int(n_status) == int(b_status) == 0
    np.testing.assert_allclose(n_coef, b_coef, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(float(n_dof), float(b_dof), rtol=RTOL, atol=ATOL)


def test_chol_solve_L2_rank_deficient_matches():
    # Duplicate column -> singular Gram even with no penalty.
    n = 20
    base = np.random.default_rng(0).normal(size=(n, 1))
    X = np.ascontiguousarray(np.hstack([base, base, base]), dtype=np.float64)
    y = np.ascontiguousarray(np.random.default_rng(1).normal(size=n), dtype=np.float64)

    n_coef, n_L, n_dof, n_status = native_chol_solve_L2(X, y, 0.0, False)
    b_coef, b_L, b_dof, b_status = numba_chol_solve_L2(X, y, np.float64(0.0), False)

    assert int(n_status) == int(b_status) == -1
    assert np.all(np.isnan(n_coef)) and np.all(np.isnan(b_coef))
    assert np.isnan(float(n_dof)) and np.isnan(float(b_dof))
    assert n_L.shape == b_L.shape == (0, 0)


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("criterion", ["aic", "bic", "loss"])
@pytest.mark.parametrize("p", [1, 3, 6])
def test_ridge_grid_search_parity(intercept, criterion, p):
    n, num = 50, 11
    start, stop = 1e-3, 50.0
    X, y = _design(n, p, seed=p * 7 + len(criterion), intercept=intercept)

    alphas = np.ascontiguousarray(
        log_grid(np.float64(start), np.float64(stop), num), dtype=np.float64
    )
    n_alpha, n_coef, n_obj, n_status = native_ridge_grid_search(
        X, y, alphas, _CRITERION_CODES[criterion], intercept
    )

    obj = get_criterion(criterion)
    b_alpha, b_coef, b_obj, b_status = numba_grid_search(
        X, y, np.float64(start), np.float64(stop), num, obj, intercept
    )

    assert int(n_status) == int(b_status) == 0
    np.testing.assert_allclose(float(n_alpha), float(b_alpha), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(n_coef, b_coef, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(float(n_obj), float(b_obj), rtol=RTOL, atol=ATOL)
