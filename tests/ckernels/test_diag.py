"""Parity tests for the native diagnostic-test kernels (_ckernels.diag).

Each native kernel is checked against its numba oracle on full-rank designs, and
the rank-deficient path is checked to return DIAG_FALLBACK so the consumer
recomputes through numba. Gram matrices square the condition number of the
design, so the tolerance here is looser than the kalman parity suite's 1e-9.
"""

from __future__ import annotations

import numpy as np
import pytest

diag = pytest.importorskip("SymbolicDSGE._ckernels.diag")

from SymbolicDSGE._diag_tests import breusch_godfrey as bg_mod
from SymbolicDSGE._diag_tests import breusch_pagan as bp_mod
from SymbolicDSGE._diag_tests import chow as chow_mod
from SymbolicDSGE._diag_tests import cusum as cusum_mod
from SymbolicDSGE._diag_tests import cusumsq as cusumsq_mod

RTOL = 1e-7
ATOL = 1e-9


def _design(rng, T=160, k=3):
    """Full-rank design with an intercept plus a stable response."""
    X = np.column_stack([np.ones(T), rng.standard_normal((T, k - 1))])
    beta = rng.standard_normal(k)
    y = X @ beta + rng.standard_normal(T)
    return np.ascontiguousarray(y), np.ascontiguousarray(X)


def _collinear_design(rng, T=160):
    """Rank-deficient design: a duplicated regressor column."""
    a = rng.standard_normal(T)
    X = np.column_stack([np.ones(T), a, a])
    y = a + rng.standard_normal(T)
    return np.ascontiguousarray(y), np.ascontiguousarray(X)


# ---------------------------------------------------------------- full-rank parity


def test_bg_stat_parity():
    rng = np.random.default_rng(0)
    y, X = _design(rng)
    eps = np.ascontiguousarray(rng.standard_normal(X.shape[0]))
    for lags in (1, 2, 4):
        ns, nstat = diag.bg_stat(eps, X, lags)
        rs, rstat = bg_mod._bg_stat_numba(eps, X, lags)
        assert ns == rs == 0
        assert np.isclose(nstat, rstat, rtol=RTOL, atol=ATOL)


def test_bp_aux_parity():
    rng = np.random.default_rng(1)
    _, X = _design(rng)
    eps = np.ascontiguousarray(rng.standard_normal(X.shape[0]))
    ns, nrss, ntss = diag.bp_aux(eps, X)
    rs, rrss, rtss = bp_mod.bp_aux(eps, X)
    assert ns == rs == 0
    assert np.isclose(nrss, rrss, rtol=RTOL, atol=ATOL)
    assert np.isclose(ntss, rtss, rtol=RTOL, atol=ATOL)


def test_chow_stat_parity():
    rng = np.random.default_rng(2)
    y, X = _design(rng)
    for t_break in (40, 80, 120):
        ns, nstat = diag.chow_stat(y, X, t_break)
        rs, rstat = chow_mod._chow_stat_numba(y, X, t_break)
        assert ns == rs == 0
        assert np.isclose(nstat, rstat, rtol=RTOL, atol=ATOL)


def test_cusum_series_parity():
    rng = np.random.default_rng(3)
    y, X = _design(rng)
    ns, nser = diag.cusum_series(y, X)
    rs, rser = cusum_mod._cusum_series_numba(y, X)
    assert ns == rs == 0
    np.testing.assert_allclose(nser, rser, rtol=RTOL, atol=ATOL)


def test_cusum_stat_parity():
    rng = np.random.default_rng(4)
    y, X = _design(rng)
    ns, nstat = diag.cusum_stat(y, X)
    rs, rstat = cusum_mod._cusum_stat_numba(y, X)
    assert ns == rs == 0
    assert np.isclose(nstat, rstat, rtol=RTOL, atol=ATOL)


def test_cusumsq_stat_parity():
    rng = np.random.default_rng(5)
    y, X = _design(rng)
    ns, nn, nstat = diag.cusumsq_stat(y, X)
    rs, rn, rstat = cusumsq_mod._cusumsq_stat_numba(y, X)
    assert ns == rs == 0
    assert nn == rn
    assert np.isclose(nstat, rstat, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------- fallback signal


def test_rank_deficient_returns_fallback():
    """A collinear design makes the Cholesky path fail; the native kernels must
    signal DIAG_FALLBACK rather than producing a (wrong) result."""
    rng = np.random.default_rng(6)
    y, X = _collinear_design(rng)
    eps = np.ascontiguousarray(rng.standard_normal(X.shape[0]))

    assert diag.bg_stat(eps, X, 2)[0] == diag.FALLBACK
    assert diag.bp_aux(eps, X)[0] == diag.FALLBACK
    assert diag.chow_stat(y, X, 80)[0] == diag.FALLBACK
    assert diag.cusum_series(y, X)[0] == diag.FALLBACK
    assert diag.cusum_stat(y, X)[0] == diag.FALLBACK
    assert diag.cusumsq_stat(y, X)[0] == diag.FALLBACK


def _same_result(a, b):
    """Compare (status, stat) results, treating nan == nan."""
    (sa, va), (sb, vb) = a, b
    assert sa == sb
    if np.isnan(va) or np.isnan(vb):
        assert np.isnan(va) and np.isnan(vb)
    else:
        assert np.isclose(va, vb, rtol=RTOL, atol=ATOL)


def test_rank_deficient_dispatch_matches_numba():
    """On a rank-deficient design the public dispatch must transparently fall
    back to the numba (lstsq) path, producing the same result as pure numba."""
    rng = np.random.default_rng(7)
    y, X = _collinear_design(rng)
    eps = np.ascontiguousarray(rng.standard_normal(X.shape[0]))

    _same_result(bg_mod.bg_stat(eps, X, 2), bg_mod._bg_stat_numba(eps, X, 2))
    _same_result(bp_mod.bp_stat(eps, X), bp_mod._bp_stat_numba(eps, X))
    _same_result(chow_mod._chow_stat(y, X, 80), chow_mod._chow_stat_numba(y, X, 80))
