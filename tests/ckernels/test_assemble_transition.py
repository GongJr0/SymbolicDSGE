"""Parity: native ``assemble_transition`` vs the reference block assembly.

The native kernel (``_ckernels/core/core.c``) builds the first-order transition
``A`` from the solved rule ``(p, f)``. It assembles only ``A``: the shock
loading is one solve spanning every variable, which the pencil stage emits
whole, so no product of ``f`` with a state loading reproduces it.

``A`` is the rule scattered into the state columns rather than ``[[p, 0], [f@p,
0]]``. A state is a variable occurring at ``t-1``, so ``(p, f)`` already maps
``y_{t-1}`` to ``y_t`` and a control responds to the same lagged state the
transition does, not to the state at ``t``.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.core import assemble_transition

RTOL = 1e-10
ATOL = 1e-12


def _oracle(p, f, n_s, n_u):
    return np.block(
        [
            [np.asarray(p, dtype=np.float64), np.zeros((n_s, n_u))],
            [np.asarray(f, dtype=np.float64), np.zeros((n_u, n_u))],
        ]
    )


@pytest.mark.parametrize(
    "n_state,n_ctrl",
    [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 0)],
)
def test_assemble_transition_parity(n_state, n_ctrl):
    rng = np.random.default_rng(n_state * 100 + n_ctrl * 10)
    p = rng.normal(size=(n_state, n_state))
    f = rng.normal(size=(n_ctrl, n_state))

    A = assemble_transition(p, f, n_state, n_ctrl)

    assert A.shape == (n_state + n_ctrl, n_state + n_ctrl)
    np.testing.assert_allclose(A, _oracle(p, f, n_state, n_ctrl), rtol=RTOL, atol=ATOL)


def test_control_columns_are_empty():
    """Nothing reads a control one period on, so its column contributes nothing."""
    n_state, n_ctrl = 3, 2
    rng = np.random.default_rng(0)
    A = assemble_transition(
        rng.normal(size=(n_state, n_state)),
        rng.normal(size=(n_ctrl, n_state)),
        n_state,
        n_ctrl,
    )

    assert np.abs(A[:, n_state:]).max() == 0.0
