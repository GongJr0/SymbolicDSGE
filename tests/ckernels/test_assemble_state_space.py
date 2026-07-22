"""Parity: native ``assemble_state_space`` vs the reference block assembly.

The native kernel (``_ckernels/core/core.c``) builds the first-order state space
``(A, B)`` from the Klein policy ``(p, f)``; it replaced the pure-Python
``_assemble_state_space`` in ``core/solver.py``. The oracle below is that deleted
body, with ``np.real`` in place of ``real_if_close`` (the native kernel always
takes the real part).
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.core import assemble_state_space

RTOL = 1e-10
ATOL = 1e-12


def _oracle(p, f, n_s, n_u, n_exo):
    p = np.asarray(p, dtype=np.complex128)
    f = np.asarray(f, dtype=np.complex128)
    A = np.real(np.block([[p, np.zeros((n_s, n_u))], [f @ p, np.zeros((n_u, n_u))]]))
    B_state = np.vstack([np.eye(n_exo), np.zeros((n_s - n_exo, n_exo))]).astype(
        np.float64
    )
    B = np.real(np.vstack([B_state, f @ B_state]))
    return A, B


@pytest.mark.parametrize(
    "n_state,n_ctrl,n_exog",
    [(1, 0, 1), (2, 1, 1), (3, 2, 2), (4, 3, 2), (5, 4, 5), (6, 0, 3)],
)
def test_assemble_state_space_parity(n_state, n_ctrl, n_exog):
    rng = np.random.default_rng(n_state * 100 + n_ctrl * 10 + n_exog)
    p = rng.normal(size=(n_state, n_state)) + 1j * rng.normal(size=(n_state, n_state))
    f = rng.normal(size=(n_ctrl, n_state)) + 1j * rng.normal(size=(n_ctrl, n_state))

    A, B = assemble_state_space(p, f, n_state, n_ctrl, n_exog)
    Ao, Bo = _oracle(p, f, n_state, n_ctrl, n_exog)

    assert A.shape == (n_state + n_ctrl, n_state + n_ctrl)
    assert B.shape == (n_state + n_ctrl, n_exog)
    np.testing.assert_allclose(A, Ao, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(B, Bo, rtol=RTOL, atol=ATOL)


def test_assemble_state_space_real_part_only():
    """A purely imaginary policy collapses to zero (real part taken)."""
    n_state, n_ctrl, n_exog = 3, 2, 2
    p = 1j * np.ones((n_state, n_state))
    f = 1j * np.ones((n_ctrl, n_state))
    A, B = assemble_state_space(p, f, n_state, n_ctrl, n_exog)
    # top-left = Re(p) = 0; bottom-left = Re(f@p) = Re(sum of i*i) = -n_state
    np.testing.assert_allclose(A[:n_state, :n_state], 0.0, atol=ATOL)
    np.testing.assert_allclose(A[n_state:, :n_state], -n_state, atol=ATOL)
