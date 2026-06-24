"""Parity tests for the native linear Kalman hot loop vs the numba reference.

Skips when the ``_ckernels.kalman`` extension is not built.
"""

import numpy as np
import pytest

from SymbolicDSGE.kalman.filter import _kalman_hot_loop

kf = pytest.importorskip("SymbolicDSGE._ckernels.kalman")

# Cross-compiler FP (log/sqrt/division last-ULP) accumulates over T steps, so
# compare tightly rather than bit-exact.
_RTOL = 1e-9
_ATOL = 1e-9

_HIST = (
    "x_pred",
    "x_filt",
    "P_pred",
    "P_filt",
    "y_pred",
    "y_filt",
    "innov",
    "std_innov",
    "S",
)


def _c(z: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(z, dtype=np.float64)


def _make_system(n: int, m: int, k: int, T: int, seed: int):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    A *= 0.5 / np.max(np.abs(np.linalg.eigvals(A)))  # spectral radius 0.5 (stable)
    B = rng.standard_normal((n, k))
    C = rng.standard_normal((m, n))
    d = rng.standard_normal(m)
    Qh = rng.standard_normal((k, k))
    Rh = rng.standard_normal((m, m))
    Ph = rng.standard_normal((n, n))
    Q = Qh @ Qh.T + np.eye(k)
    R = Rh @ Rh.T + np.eye(m)
    P0 = Ph @ Ph.T + np.eye(n)
    P0 = 0.5 * (P0 + P0.T)
    y = rng.standard_normal((T, m))
    x0 = rng.standard_normal(n)
    return tuple(_c(z) for z in (A, B, C, d, Q, R, y, x0, P0))


@pytest.mark.parametrize(
    "n, m, k, T", [(2, 1, 1, 20), (4, 2, 2, 50), (3, 3, 1, 30), (5, 2, 3, 40)]
)
@pytest.mark.parametrize("symmetrize", [True, False])
@pytest.mark.parametrize("return_shocks", [True, False])
@pytest.mark.parametrize("store_history", [True, False])
def test_kalman_matches_numba(
    n: int,
    m: int,
    k: int,
    T: int,
    symmetrize: bool,
    return_shocks: bool,
    store_history: bool,
) -> None:
    A, B, C, d, Q, R, y, x0, P0 = _make_system(
        n, m, k, T, seed=n * 1000 + m * 100 + k * 10 + T
    )
    args = (
        T,
        (n, m, k),
        A,
        B,
        C,
        d,
        Q,
        R,
        y,
        x0,
        P0,
        symmetrize,
        0.0,
        return_shocks,
        store_history,
    )

    _, _, out_ref = _kalman_hot_loop(*args)
    _, _, out_native = kf.kalman_hot_loop(*args)

    for i, name in enumerate(_HIST):
        np.testing.assert_allclose(
            out_native[i], out_ref[i], rtol=_RTOL, atol=_ATOL, err_msg=name
        )
    # eps_hat (index 9)
    np.testing.assert_allclose(
        out_native[9], out_ref[9], rtol=_RTOL, atol=_ATOL, err_msg="eps_hat"
    )
    # loglik (index 10)
    np.testing.assert_allclose(
        out_native[10], out_ref[10], rtol=_RTOL, atol=_ATOL, err_msg="loglik"
    )


def test_kalman_non_pd_raises() -> None:
    # m=1 with a negative R forces S = C P_pred C^T + R < 0, so the Cholesky
    # fails -> both paths raise LinAlgError (-> MatrixConditionError upstream).
    n, m, k, T = 1, 1, 1, 5
    rng = np.random.default_rng(0)
    args = (
        T,
        (n, m, k),
        _c([[0.5]]),
        _c([[1.0]]),
        _c([[1.0]]),
        _c([0.0]),
        _c([[1.0]]),
        _c([[-10.0]]),  # negative "covariance" -> non-PD innovation covariance
        _c(rng.standard_normal((T, m))),
        _c([0.0]),
        _c([[1.0]]),
        True,
        0.0,
        False,
        True,
    )
    with pytest.raises(np.linalg.LinAlgError):
        kf.kalman_hot_loop(*args)
    with pytest.raises(np.linalg.LinAlgError):
        _kalman_hot_loop(*args)
