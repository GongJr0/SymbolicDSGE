"""Parity tests for the native linear Kalman hot loop vs the numpy oracle.

Skips when the ``_ckernels.kalman`` extension is not built.
"""

import numpy as np
import pytest
from numba import cfunc, types

from SymbolicDSGE.kalman.errors import ErrorCode
from _oracles.kalman import kf_reference

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


_MEAS_SIG = types.void(
    types.CPointer(types.float64),
    types.CPointer(types.float64),
    types.CPointer(types.float64),
)


@cfunc(_MEAS_SIG)
def _ukf_measure_first_var(model_vars, _params, out):
    out[0] = model_vars[0]


@pytest.mark.parametrize(
    "n, m, k, T", [(2, 1, 1, 20), (4, 2, 2, 50), (3, 3, 1, 30), (5, 2, 3, 40)]
)
@pytest.mark.parametrize("symmetrize", [True, False])
@pytest.mark.parametrize("return_shocks", [True, False])
@pytest.mark.parametrize("store_history", [True, False])
def test_kalman_matches_oracle(
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

    out_ref = kf_reference(*args)
    _status, out_native = kf.kalman_hot_loop(*args)

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


def test_kalman_non_pd_returns_error_code() -> None:
    # m=1 with a negative R forces S = C P_pred C^T + R < 0, so the Cholesky
    # fails. The native hot loop returns the MATRIX_CONDITION status code (the
    # caller maps it to MatrixConditionError); it does not raise here.
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
    status, _ = kf.kalman_hot_loop(*args)
    assert status == ErrorCode.MATRIX_CONDITION


def test_ukf_returns_projected_model_variable_history() -> None:
    T = 4
    hx = _c([[0.55]])
    gx = _c([[2.0]])
    bx = _c([[1.0]])
    hxx = _c([[[0.1]]])
    gxx = _c([[[0.4]]])
    hss = _c([0.03])
    gss = _c([0.2])
    steady_state = _c([10.0, 20.0])
    params = _c([])
    Q = _c([[0.05]])
    R = _c([[0.25]])
    obs = _c(np.zeros((T, 1)))
    z0 = _c([0.1, 0.0])
    P0 = _c(np.diag([0.2, 0.3]))

    _, out = kf.ukf_hot_loop(
        _ukf_measure_first_var.address,
        hx,
        gx,
        bx,
        hxx,
        gxx,
        hss,
        gss,
        steady_state,
        params,
        Q,
        R,
        obs,
        z0,
        P0,
        1.0,
        2.0,
        1.0,
        1e-12,
        True,
        True,
    )
    (
        x1_pred,
        x2_pred,
        x1_filt,
        x2_filt,
        x_pred,
        x_filt,
        P_pred,
        P_filt,
        _y_pred,
        _y_filt,
        _innov,
        _std_innov,
        _S,
        _loglik,
    ) = out

    assert x_pred.shape == (T, 2)
    assert x_filt.shape == (T, 2)
    np.testing.assert_allclose(
        x_pred[:, 0],
        steady_state[0] + x1_pred[:, 0] + x2_pred[:, 0],
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        x_filt[:, 0],
        steady_state[0] + x1_filt[:, 0] + x2_filt[:, 0],
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        x_pred[:, 1],
        steady_state[1]
        + 0.5 * gss[0]
        + gx[0, 0] * (x1_pred[:, 0] + x2_pred[:, 0])
        + 0.5 * gxx[0, 0, 0] * (P_pred[:, 0, 0] + x1_pred[:, 0] ** 2),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        x_filt[:, 1],
        steady_state[1]
        + 0.5 * gss[0]
        + gx[0, 0] * (x1_filt[:, 0] + x2_filt[:, 0])
        + 0.5 * gxx[0, 0, 0] * (P_filt[:, 0, 0] + x1_filt[:, 0] ** 2),
        rtol=_RTOL,
        atol=_ATOL,
    )
