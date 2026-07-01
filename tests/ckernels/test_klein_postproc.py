"""Parity: native klein post-proc vs the numba ``_klein_postprocess`` (live path).

The native C kernel in ``_ckernels/core/klein_postproc.c`` consumes the ordered
generalized-Schur factors (from ``scipy.linalg.ordqz``) and must reproduce the
``f``, ``p``, ``stab``, ``eig`` of the numba reference. Both are fed the *same*
Schur factors, so the only difference is the post-proc arithmetic (hand-rolled
complex LU vs numpy/LAPACK) -- tolerance, not bit-exact.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import ordqz

from SymbolicDSGE._ckernels.core import klein_postprocess as native_kp
from SymbolicDSGE._linearsolve import _klein_postprocess as numba_kp

RTOL = 1e-8
ATOL = 1e-10


def _schur(seed: int, N: int):
    """Ordered complex Schur factors of a random real pencil (a, b)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(N, N))
    b = rng.normal(size=(N, N))
    s, t, _alpha, _beta, q, z = ordqz(a, b, sort="ouc", output="complex")
    cc = np.ascontiguousarray
    return cc(s), cc(t), cc(q), cc(z)


@pytest.mark.parametrize("N,n_states", [(3, 1), (4, 2), (5, 3), (6, 2), (6, 4)])
def test_klein_postproc_parity(N, n_states):
    s, t, q, z = _schur(seed=N * 100 + n_states, N=N)
    empty = np.empty((0, 0), dtype=np.complex128)

    nf, _n, npp, _l, nstab, neig = numba_kp(s, t, q, z, empty, empty, n_states)
    f, p, stab, eig = native_kp(s, t, z, n_states)

    assert int(stab) == int(nstab)
    np.testing.assert_allclose(f, nf, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(p, npp, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(eig, neig, rtol=RTOL, atol=ATOL)


def test_klein_postproc_shapes():
    s, t, _q, z = _schur(seed=7, N=5)
    f, p, stab, eig = native_kp(s, t, z, 3)
    assert f.shape == (2, 3)  # (n_costates, n_states)
    assert p.shape == (3, 3)
    assert eig.shape == (5,)
    assert isinstance(stab, int)


def test_klein_postproc_rejects_no_states():
    s, t, _q, z = _schur(seed=1, N=4)
    with pytest.raises(ValueError, match="n_states"):
        native_kp(s, t, z, 0)
    with pytest.raises(ValueError, match="matrix dimension"):
        native_kp(s, t, z, 5)
