"""Coverage for the Brown-Durbin-Evans recursive-residual kernel.

``recursive_residuals`` is the shared seed for both CUSUM variants but has no
direct test. This drives its three status branches and checks the exact-fit
property (a perfectly linear ``y`` yields zero recursive residuals).
"""

from __future__ import annotations

import numpy as np

from SymbolicDSGE._diag_tests import cusum_utils as C


def _design(T: int, p: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.empty((T, p), dtype=np.float64)
    X[:, 0] = 1.0
    if p > 1:
        X[:, 1:] = rng.normal(size=(T, p - 1))
    return np.ascontiguousarray(X)


def test_recursive_residuals_ok_and_length():
    X = _design(24, 2, seed=1)
    beta = np.array([0.5, -1.3], dtype=np.float64)
    rng = np.random.default_rng(2)
    y = np.ascontiguousarray(X @ beta + rng.normal(scale=0.5, size=24))

    status, w = C.recursive_residuals(y, X)
    assert status == C.OK
    assert w.shape == (24 - 2,)
    assert np.all(np.isfinite(w))


def test_recursive_residuals_exact_fit_is_zero():
    # y exactly linear in X: the seed recovers the true beta and every later
    # one-step prediction is exact, so every recursive residual is ~0.
    X = _design(18, 3, seed=3)
    beta = np.array([1.0, 2.0, -0.5], dtype=np.float64)
    y = np.ascontiguousarray(X @ beta)

    status, w = C.recursive_residuals(y, X)
    assert status == C.OK
    assert np.allclose(w, 0.0, atol=1e-8)


def test_recursive_residuals_insufficient_samples():
    # T <= p
    X = _design(2, 2, seed=4)
    y = np.ascontiguousarray(np.zeros(2, dtype=np.float64))
    status, w = C.recursive_residuals(y, X)
    assert status == C.INSUFFICIENT_SAMPLES
    assert w.size == 0

    # T == 0
    X0 = np.ascontiguousarray(np.empty((0, 2), dtype=np.float64))
    y0 = np.ascontiguousarray(np.empty(0, dtype=np.float64))
    status0, w0 = C.recursive_residuals(y0, X0)
    assert status0 == C.INSUFFICIENT_SAMPLES
    assert w0.size == 0


def test_recursive_residuals_bad_shape():
    X = _design(10, 2, seed=5)
    y = np.ascontiguousarray(np.zeros(9, dtype=np.float64))  # y.size != T
    status, w = C.recursive_residuals(y, X)
    assert status == C.BAD_SHAPE
    assert w.size == 0
