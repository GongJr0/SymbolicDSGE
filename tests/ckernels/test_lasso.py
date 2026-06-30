"""Parity tests: native lasso kernels vs the numba reference.

The numba lasso helpers are compiled with ``fastmath=True``; the native C is
strict IEEE, so the two are not bit-identical. Both converge to the same (unique)
lasso minimizer, so parity is checked at the solver tolerance, not ULP. Inputs
stay in the small/medium regime (where production dispatch routes to native).
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.regression import (
    lars_lasso_gram as native_lars,
    lasso_gram_cd as native_cd,
    lasso_path_eval as native_path_eval,
)
from SymbolicDSGE.regression.lasso.core import (
    lars_lasso_gram as numba_lars,
    lasso_gram_cd as numba_cd,
    lasso_path_eval as numba_path_eval,
)
from SymbolicDSGE.regression.utils import log_grid

# Converged-tolerance bar: the lasso minimizer is unique, but strict-vs-fastmath
# accumulation makes ULP parity unattainable.
RTOL = 1e-7
ATOL = 1e-8


def _scaled_gram(n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    beta = np.zeros(k)
    beta[: max(1, k // 2)] = rng.normal(size=max(1, k // 2))
    y = X @ beta + 0.25 * rng.normal(size=n)
    Xc = X - X.mean(0)
    yc = y - y.mean()
    G = np.ascontiguousarray((Xc.T @ Xc) / n, dtype=np.float64)
    c = np.ascontiguousarray((Xc.T @ yc) / n, dtype=np.float64)
    return G, c


@pytest.mark.parametrize("k", [1, 2, 5, 12])
@pytest.mark.parametrize("alpha", [1e-3, 0.02, 0.1, 0.5])
def test_lasso_gram_cd_parity(k, alpha):
    G, c = _scaled_gram(60, k, seed=k * 31 + int(alpha * 1000))
    n_coef, n_status = native_cd(G, c, float(alpha), 1000, 1e-10)
    b_coef, b_status = numba_cd(G, c, np.float64(alpha), 1000, np.float64(1e-10))
    assert int(n_status) == int(b_status) == 0
    np.testing.assert_allclose(n_coef, b_coef, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("k", [1, 3, 6, 10])
def test_lars_lasso_path_parity(k):
    G, c = _scaled_gram(80, k, seed=k * 17 + 4)

    n_lam, n_beta, n_status = native_lars(G, c, 500, 1e-12)
    b_lam, b_beta, b_status = numba_lars(G, c, 500, np.float64(1e-12))

    assert int(n_status) == int(b_status) == 0
    # Knot counts and the path itself agree for generic (non-degenerate) data.
    assert n_lam.shape == b_lam.shape
    np.testing.assert_allclose(n_lam, b_lam, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(n_beta, b_beta, rtol=RTOL, atol=ATOL)

    # Path evaluation parity on a descending lambda grid.
    grid = np.ascontiguousarray(
        log_grid(np.float64(1e-3), np.float64(1.0), 13)[::-1], dtype=np.float64
    )
    n_eval = native_path_eval(n_lam, n_beta, grid)
    b_eval = numba_path_eval(b_lam, b_beta, grid)
    np.testing.assert_allclose(n_eval, b_eval, rtol=RTOL, atol=ATOL)


def test_lasso_path_eval_matches_on_shared_path():
    # Feed both evaluators the *same* (numba) path so this isolates path_eval.
    G, c = _scaled_gram(70, 5, seed=99)
    lam, beta, _ = numba_lars(G, c, 500, np.float64(1e-12))
    grid = np.ascontiguousarray(
        log_grid(np.float64(1e-4), np.float64(2.0), 20)[::-1], dtype=np.float64
    )
    np.testing.assert_allclose(
        native_path_eval(lam, beta, grid),
        numba_path_eval(lam, beta, grid),
        rtol=1e-12,
        atol=1e-12,
    )
