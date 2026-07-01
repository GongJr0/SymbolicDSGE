"""Parity: native klein post-proc vs the numba twin ``_klein_postprocess_numba``.

The native C kernel in ``_ckernels/core/klein_postproc.c`` and the numba twin in
``core.klein`` are the two implementations ``klein_solve`` dispatches between, so
they must agree. Both consume the ordered generalized-Schur factors (from
``scipy.linalg.ordqz``) fed identically; the only difference is the post-proc
arithmetic (hand-rolled complex LU vs numpy/LAPACK) -- tolerance, not bit-exact.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import ordqz

from SymbolicDSGE._ckernels.core import klein_postprocess as native_kp
from SymbolicDSGE.core.klein import _klein_postprocess_numba as numba_kp

RTOL = 1e-8
ATOL = 1e-10


def _schur(seed: int, N: int):
    """Ordered complex Schur factors of a random real pencil (a, b)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(N, N))
    b = rng.normal(size=(N, N))
    s, t, _alpha, _beta, _q, z = ordqz(a, b, sort="ouc", output="complex")
    cc = np.ascontiguousarray
    return cc(s), cc(t), cc(z)


@pytest.mark.parametrize("N,n_states", [(3, 1), (4, 2), (5, 3), (6, 2), (6, 4)])
def test_klein_postproc_parity(N, n_states):
    s, t, z = _schur(seed=N * 100 + n_states, N=N)

    nf, npp, nstab, neig = numba_kp(s, t, z, n_states)
    f, p, stab, eig = native_kp(s, t, z, n_states)

    assert int(stab) == int(nstab)
    np.testing.assert_allclose(f, nf, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(p, npp, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(eig, neig, rtol=RTOL, atol=ATOL)


def test_klein_postproc_shapes():
    s, t, z = _schur(seed=7, N=5)
    f, p, stab, eig = native_kp(s, t, z, 3)
    assert f.shape == (2, 3)  # (n_costates, n_states)
    assert p.shape == (3, 3)
    assert eig.shape == (5,)
    assert isinstance(stab, int)


def test_klein_postproc_rejects_no_states():
    s, t, z = _schur(seed=1, N=4)
    with pytest.raises(ValueError, match="n_states"):
        native_kp(s, t, z, 0)
    with pytest.raises(ValueError, match="matrix dimension"):
        native_kp(s, t, z, 5)
