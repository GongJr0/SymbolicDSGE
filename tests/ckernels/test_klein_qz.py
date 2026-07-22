"""Parity: native ``klein_qz`` vs ``scipy.linalg.ordqz``.

``klein_qz`` (the pyx shim over ``klein_qz.c``) computes the ordered complex
generalized Schur (QZ) decomposition with the Klein 'ouc' ordering. Both it and
``ordqz`` reduce through the same LAPACK path (unordered ``zgges`` then a
``ztgsen`` reorder on the same 'outside unit circle' select set), so the ordered
factors agree entrywise -- tolerance, not bit-exact. ``ordqz`` returns
``(s, t, alpha, beta, q, z)``; indices ``[0, 1, 5]`` are the ``(s, t, z)`` the
native routine produces.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import ordqz

from SymbolicDSGE._ckernels.core import klein_qz

RTOL = 1e-8
ATOL = 1e-10


@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 8, 12])
def test_klein_qz_matches_ordqz(N):
    rng = np.random.default_rng(N * 131 + 7)
    a = rng.normal(size=(N, N))
    b = rng.normal(size=(N, N))

    s, t, z = klein_qz(a, b)
    os_, ot, _alpha, _beta, _q, oz = ordqz(a, b, sort="ouc", output="complex")

    np.testing.assert_allclose(s, os_, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(t, ot, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(z, oz, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("N", [2, 4, 6])
def test_klein_qz_invariants(N):
    """Schur-convention-free checks: Z unitary, and the ordered generalized
    eigenvalues diag(s)/diag(t) are sorted 'outside unit circle' first."""
    rng = np.random.default_rng(N * 977 + 3)
    a = rng.normal(size=(N, N))
    b = rng.normal(size=(N, N))

    s, t, z = klein_qz(a, b)

    np.testing.assert_allclose(z.conj().T @ z, np.eye(N), rtol=RTOL, atol=ATOL)

    eig = np.abs(np.diag(s)) / np.abs(np.diag(t))
    n_ouc = int(np.sum(eig > 1.0))
    # the ouc eigenvalues occupy the leading positions
    assert np.all(eig[:n_ouc] > 1.0)


def test_klein_qz_empty():
    s, t, z = klein_qz(np.zeros((0, 0)), np.zeros((0, 0)))
    assert s.shape == (0, 0)
    assert t.shape == (0, 0)
    assert z.shape == (0, 0)


def test_klein_qz_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        klein_qz(np.zeros((3, 2)), np.zeros((3, 2)))
