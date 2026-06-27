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


def _design(rng, T=160, k=3, intercept=True):
    """Full-rank design plus a stable response.

    ``intercept`` controls whether a leading column of ones is included. The
    Breusch-Godfrey kernel builds its own intercept into the auxiliary design,
    so it must be fed an intercept-free X — otherwise the augmented matrix has
    two identical columns of ones and is rank-deficient (cond ~1e16), which
    makes the chol-vs-lstsq path build-dependent rather than a parity check.
    """
    if intercept:
        X = np.column_stack([np.ones(T), rng.standard_normal((T, k - 1))])
    else:
        X = rng.standard_normal((T, k))
    beta = rng.standard_normal(X.shape[1])
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
    # bg adds its own intercept; feed an intercept-free design to keep it full rank.
    y, X = _design(rng, intercept=False)
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


# ---------------------------------------------------------------- HAC estimator

from SymbolicDSGE._diag_tests.hac_covariance import (  # noqa: E402
    _BARTLETT,
    _PARZEN,
    _QS,
    jit_hac_estimator_matmul,
)

# Golden HAC long-run covariances for the centered design below at bandwidth 2,
# mirroring tests/diag_tests/test_hac_covariance.py::GOLDEN_HAC_CENTERED. These
# anchor the native kernel to the same absolute values the numba side is pinned
# to, rather than relying only on transitivity through the parity sweep below.
_HAC_GOLDEN_R = np.array(
    [[1.0, 2.0], [2.0, -1.0], [0.0, 1.0], [3.0, 0.0], [-1.0, 2.0]],
    dtype=np.float64,
)
_HAC_GOLDEN = {
    _BARTLETT: np.array(
        [
            [0.6666666666666667, -0.5599999999999998],
            [-0.5599999999999998, 0.6453333333333331],
        ],
        dtype=np.float64,
    ),
    _PARZEN: np.array(
        [
            [0.5629629629629630, -0.3733333333333333],
            [-0.3733333333333333, 0.6079999999999999],
        ],
        dtype=np.float64,
    ),
    _QS: np.array(
        [
            [0.4104387018488457, -0.4840134757411693],
            [-0.4840134757411693, 0.5017280910104844],
        ],
        dtype=np.float64,
    ),
}


@pytest.mark.parametrize("kernel_id", [_BARTLETT, _PARZEN, _QS])
def test_hac_estimator_matches_golden(kernel_id):
    """Native HAC long-run covariance matches the pinned golden fixtures (an
    absolute anchor independent of the numba reference)."""
    centered = np.ascontiguousarray(_HAC_GOLDEN_R - _HAC_GOLDEN_R.mean(axis=0))
    out = diag.hac_estimator_matmul(centered, kernel_id, 2)
    np.testing.assert_allclose(out, _HAC_GOLDEN[kernel_id], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kernel_id", [_BARTLETT, _PARZEN, _QS])
@pytest.mark.parametrize("shape", [(160, 3), (200, 2), (80, 4), (50, 1)])
@pytest.mark.parametrize("lags", [0, 1, 4, 9])
def test_hac_estimator_parity(kernel_id, shape, lags):
    """Native HAC matches the numba jit_hac_estimator_matmul across kernels,
    bandwidths, and shapes -- broad coverage beyond the golden fixtures."""
    n, p = shape
    rng = np.random.default_rng([kernel_id, n, p, lags])
    r = np.ascontiguousarray(rng.standard_normal((n, p)))
    native = diag.hac_estimator_matmul(r, kernel_id, lags)
    ref = jit_hac_estimator_matmul(r, kernel_id, lags)
    np.testing.assert_allclose(native, ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------- Wald helpers

from SymbolicDSGE._diag_tests.wald_test import (  # noqa: E402
    jit_fill_symmetric_target_vec,
    jit_symmetric_outer_prod_2dim,
    jit_wald_stat_from_mean_and_cov,
)


@pytest.mark.parametrize("q", [1, 2, 5])
def test_wald_stat_parity(q):
    """Native Wald statistic matches numba on a positive-definite omega."""
    rng = np.random.default_rng([q, 11])
    mean = np.ascontiguousarray(rng.standard_normal(q))
    target = np.ascontiguousarray(rng.standard_normal(q))
    m = rng.standard_normal((q, q))
    omega = np.ascontiguousarray(m @ m.T + q * np.eye(q))  # SPD, well-conditioned
    ns, nstat = diag.wald_stat_from_mean_and_cov(mean, target, omega, 100)
    rs, rstat, rdf = jit_wald_stat_from_mean_and_cov(mean, target, omega, 100)
    assert ns == rs == 0
    assert rdf == q
    assert np.isclose(nstat, rstat, rtol=RTOL, atol=ATOL)


def test_wald_stat_fallback_on_non_pd():
    """A non-PD omega makes the native Cholesky path signal DIAG_FALLBACK."""
    q = 3
    mean = np.ascontiguousarray(np.ones(q))
    target = np.ascontiguousarray(np.zeros(q))
    omega = np.zeros((q, q), dtype=np.float64)
    assert diag.wald_stat_from_mean_and_cov(mean, target, omega, 50)[0] == diag.FALLBACK


@pytest.mark.parametrize("shape", [(10, 1), (20, 3), (15, 4)])
def test_symmetric_outer_prod_parity(shape):
    """Native per-row vech outer product matches numba."""
    n, p = shape
    rng = np.random.default_rng([n, p])
    x = np.ascontiguousarray(rng.standard_normal((n, p)))
    ns, nout = diag.symmetric_outer_prod_2dim(x)
    rout = np.empty((n, p * (p + 1) // 2), dtype=np.float64)
    rs = jit_symmetric_outer_prod_2dim(x, rout)
    assert ns == rs == 0
    np.testing.assert_allclose(nout, rout, rtol=RTOL, atol=ATOL)


def test_fill_symmetric_target_vec_parity():
    """Native vech packing matches numba; asymmetric input is rejected."""
    rng = np.random.default_rng(99)
    p = 4
    a = rng.standard_normal((p, p))
    sym = np.ascontiguousarray(a + a.T)
    ns, nvec = diag.fill_symmetric_target_vec(sym, 1e-8, 1e-5)
    rvec = np.empty(p * (p + 1) // 2, dtype=np.float64)
    rs = jit_fill_symmetric_target_vec(sym, rvec)
    assert ns == rs == 0
    np.testing.assert_allclose(nvec, rvec, rtol=RTOL, atol=ATOL)

    asym = np.ascontiguousarray(np.array([[1.0, 0.1], [0.2, 1.0]]))
    assert diag.fill_symmetric_target_vec(asym, 1e-8, 1e-5)[0] != 0


from SymbolicDSGE._diag_tests.moment_calculation_utils import (  # noqa: E402
    jit_fill_centered,
    jit_fill_mean_ax0,
)


@pytest.mark.parametrize("shape", [(10, 1), (20, 3), (50, 4)])
def test_fill_mean_and_centered_parity(shape):
    """Native column-mean and centering match numba (bit-exact loops)."""
    n, p = shape
    rng = np.random.default_rng([n, p, 7])
    x = np.ascontiguousarray(rng.standard_normal((n, p)))

    nmean = diag.fill_mean_ax0(x)
    rmean = np.empty(p, dtype=np.float64)
    jit_fill_mean_ax0(x, rmean)
    np.testing.assert_allclose(nmean, rmean, rtol=RTOL, atol=ATOL)

    ncentered = diag.fill_centered_ax0(x, nmean)
    rcentered = np.empty((n, p), dtype=np.float64)
    jit_fill_centered(x, rmean, rcentered)
    np.testing.assert_allclose(ncentered, rcentered, rtol=RTOL, atol=ATOL)
