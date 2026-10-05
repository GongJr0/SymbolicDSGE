# type: ignore
"""The exchange search itself, on triples built so the answer is known.

``klein_reorder_argmax`` reads ``(s, t, z)`` and never the pencil they came
from, so a test can hand it any upper-triangular pair with any unitary ``z`` and
still drive the real LAPACK path. The triples here are diagonal with
well-separated real eigenvalues, which is what makes the expected selection
exact rather than approximate: for a diagonal pencil with distinct roots the
one-dimensional deflating subspaces are unique, so an exchange permutes the
columns of ``z`` up to a phase, and a 1-norm rcond is invariant under both. A
non-diagonal leading block would let ``ztgexc`` hand back a different unitary
basis of the same subspace, which the 1-norm is not invariant to.

The neighbourhood is single swaps, one leading slot out and one excess slot in,
so a selection two evictions from the baseline is unreachable by construction.
That is what ``argmax`` means here, and the enumeration below matches it.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.core import klein_reorder, klein_z11
from SymbolicDSGE._ckernels.core._solve_errors import SolveStatus

OK = 0

#: The gate in ``klein_z11_pair``: an inverse carrying no digits is a rank
#: failure rather than a bad score.
RCOND_FLOOR = 1e-9

LAM = [0.9, 0.5, -0.3, 0.1, 2.5]
S_DIAG = [1.0, 1.3, 0.7, 1.7, 1.1]


def _pencil(lam, s_diag):
    s = np.asarray(s_diag, dtype=np.complex128)
    t = s * np.asarray(lam, dtype=np.complex128)
    return np.asfortranarray(np.diag(s)), np.asfortranarray(np.diag(t))


def _unitary(rng, nd, blind_first=False):
    m = rng.normal(size=(nd, nd)) + 1j * rng.normal(size=(nd, nd))
    if blind_first:
        # qr puts Q[:, 0] parallel to m[:, 0], so parking the first column off
        # the leading rows leaves the baseline z11 with a zero column.
        m[:, 0] = 0.0
        m[nd - 1, 0] = 1.0
    q, _ = np.linalg.qr(m)
    return np.asfortranarray(q)


def _unitary_blind_row(rng, nd, sdim, row):
    """Unitary whose first ``sdim`` columns vanish on ``row``, which leaves every
    selection drawn from them singular. The last column takes up the slack, so
    this needs ``nd == sdim + 1``."""
    e = np.zeros(nd, dtype=np.complex128)
    e[row] = 1.0
    m = rng.normal(size=(nd, sdim)) + 1j * rng.normal(size=(nd, sdim))
    m -= np.outer(e, e.conj() @ m)
    q, _ = np.linalg.qr(m)
    return np.asfortranarray(np.hstack([q, e.reshape(nd, 1)]))


def _norm1(a):
    return np.abs(a).sum(axis=0).max()


def _score(z, n_s, cols):
    """The verdict of ``klein_z11_pair`` in numpy: zero for all it rejects."""
    z11 = z[:n_s][:, list(cols)]
    with np.errstate(all="ignore"):
        try:
            inv = np.linalg.inv(z11)
        except np.linalg.LinAlgError:
            return 0.0
        if not np.isfinite(inv).all():
            return 0.0
        r = 1.0 / (_norm1(z11) * _norm1(inv))
    return r if r > RCOND_FLOOR else 0.0


def _candidates(n_s, sdim):
    """The baseline, then the single swaps in the order the C loops take them."""
    yield (0, 0), tuple(range(n_s))
    for k in range(1, n_s + 1):
        for j in range(n_s + 1, sdim + 1):
            cols = [c for c in range(n_s) if c != k - 1] + [j - 1]
            yield (k, j), tuple(sorted(cols))


def _expected(z, n_s, sdim):
    """The winner under the C acceptance rule, which is strict: the baseline
    holds a tie, and among the swaps the earliest holds it."""
    best_key, best_cols, best = (0, 0), tuple(range(n_s)), 0.0
    scores = {}
    for key, cols in _candidates(n_s, sdim):
        scores[key] = _score(z, n_s, cols)
        if scores[key] > best:
            best_key, best_cols, best = key, cols, scores[key]
    return best_key, best_cols, best, scores


def _selected_roots(s, t, n_s):
    return np.sort((np.diag(t) / np.diag(s))[:n_s].real)


def _expected_roots(lam, cols):
    return np.sort(np.asarray(lam, dtype=np.float64)[list(cols)])


def test_the_search_returns_the_best_conditioned_single_swap():
    rng = np.random.default_rng(542)
    s, t = _pencil(LAM, S_DIAG)
    z = _unitary(rng, 5)
    n_s, sdim = 2, 4

    key, cols, best, scores = _expected(z, n_s, sdim)
    rc, s_out, t_out, z_out = klein_reorder(s, t, z, n_s, sdim)

    assert rc == OK
    np.testing.assert_allclose(_score(z_out, n_s, range(n_s)), best, rtol=1e-10)
    np.testing.assert_allclose(
        _selected_roots(s_out, t_out, n_s), _expected_roots(LAM, cols), atol=1e-12
    )

    # What makes this an argmax and not a first acceptable: nothing else ties it.
    rejected = [r for other, r in scores.items() if other != key]
    assert rejected
    assert max(rejected) < best


def test_a_singular_baseline_is_rescued_by_an_exchange():
    rng = np.random.default_rng(1138)
    s, t = _pencil(LAM, S_DIAG)
    z = _unitary(rng, 5, blind_first=True)
    n_s, sdim = 2, 4

    # Where the ordering out of the QZ leaves us: no rule to read off at all.
    rc_base, _, _, rcond_base = klein_z11(s, z, n_s)
    assert rc_base == SolveStatus.RANK_FAIL
    assert rcond_base == 0.0

    key, cols, best, scores = _expected(z, n_s, sdim)
    assert key != (0, 0)
    assert best > 0.0

    rc, s_out, t_out, z_out = klein_reorder(s, t, z, n_s, sdim)
    assert rc == OK
    np.testing.assert_allclose(
        _selected_roots(s_out, t_out, n_s), _expected_roots(LAM, cols), atol=1e-12
    )

    # Evicting the second slot keeps the blind column, which stays singular
    # whichever excess slot joins it.
    assert scores[(2, 3)] == 0.0
    assert scores[(2, 4)] == 0.0

    # The pair builder and the search agree on what won.
    rc_win, _, _, rcond_win = klein_z11(s_out, z_out, n_s)
    assert rc_win == OK
    np.testing.assert_allclose(rcond_win, best, rtol=1e-10)


def test_nothing_viable_keeps_the_baseline_verdict():
    rng = np.random.default_rng(99)
    lam, s_diag = [0.9, 0.5, -0.3, 2.5], [1.0, 1.3, 0.7, 1.1]
    s, t = _pencil(lam, s_diag)
    n_s, sdim = 2, 3
    z = _unitary_blind_row(rng, 4, sdim, row=n_s - 1)

    _, _, best, scores = _expected(z, n_s, sdim)
    assert best == 0.0
    assert set(scores.values()) == {0.0}

    rc, s_out, t_out, z_out = klein_reorder(s, t, z, n_s, sdim)

    assert rc == SolveStatus.RANK_FAIL
    # Nothing beat the baseline, so the exit re-applies no exchange.
    np.testing.assert_array_equal(s_out, s)
    np.testing.assert_array_equal(t_out, t)
    np.testing.assert_array_equal(z_out, z)


def test_the_caller_keeps_its_triple():
    """The routine writes s/t/z in place, so the entry hands it copies."""
    rng = np.random.default_rng(7)
    s, t = _pencil(LAM, S_DIAG)
    z = _unitary(rng, 5)
    before = (s.copy(), t.copy(), z.copy())

    klein_reorder(s, t, z, 2, 4)

    for given, kept in zip((s, t, z), before):
        np.testing.assert_array_equal(given, kept)


@pytest.mark.parametrize(
    ("nspred", "sdim"),
    [(0, 3), (2, 2), (3, 2), (2, 6)],
    ids=["no-states", "nothing-in-excess", "sdim-under-nspred", "sdim-over-nd"],
)
def test_rejects_an_entry_the_routine_assumes_away(nspred, sdim):
    rng = np.random.default_rng(3)
    s, t = _pencil(LAM, S_DIAG)
    z = _unitary(rng, 5)

    with pytest.raises(ValueError, match="0 < nspred < sdim <= nd"):
        klein_reorder(s, t, z, nspred, sdim)


def test_rejects_a_triple_that_does_not_line_up():
    rng = np.random.default_rng(3)
    s, t = _pencil(LAM, S_DIAG)
    z = _unitary(rng, 5)

    with pytest.raises(ValueError, match="identically shaped"):
        klein_reorder(s, t, z[:, :4], 2, 4)

    with pytest.raises(ValueError, match="identically shaped"):
        klein_z11(s, z[:, :4], 2)
