# type: ignore
from __future__ import annotations

import numpy as np
import pytest
from numba import cfunc, njit, types
from numpy import float64

import SymbolicDSGE.kalman.filter as filter_module
from SymbolicDSGE.kalman.errors import ErrorCode
from SymbolicDSGE.kalman.filter import (
    KalmanFilter,
    MatrixConditionError,
    ShapeMismatchError,
)
from _oracles.kalman import ekf_reference


def _linear_system_1d():
    A = np.array([[0.9]], dtype=float64)
    B = np.array([[1.0]], dtype=float64)
    C = np.array([[1.0]], dtype=float64)
    d = np.array([0.0], dtype=float64)
    Q = np.array([[0.04]], dtype=float64)
    R = np.array([[0.01]], dtype=float64)
    return A, B, C, d, Q, R


@pytest.mark.parametrize(
    ("mutator", "expected_name"),
    [
        (lambda A, B, Q, R, C, d: (A, B.reshape(1, 2), Q, R, C, d), "B"),
        (lambda A, B, Q, R, C, d: (A, B, np.eye(2, dtype=float64), R, C, d), "Q"),
        (lambda A, B, Q, R, C, d: (A, B, Q, np.eye(2, dtype=float64), C, d), "R"),
        (lambda A, B, Q, R, C, d: (A, B, Q, R, np.ones((2, 2), dtype=float64), d), "C"),
        (lambda A, B, Q, R, C, d: (A, B, Q, R, C, np.zeros((2,), dtype=float64)), "d"),
    ],
)
def test_shape_validate_covers_all_matrix_arguments(mutator, expected_name):
    A = np.eye(2, dtype=float64)
    B = np.ones((2, 1), dtype=float64)
    Q = np.eye(1, dtype=float64)
    R = np.eye(1, dtype=float64)
    C = np.ones((1, 2), dtype=float64)
    d = np.zeros((1,), dtype=float64)

    with pytest.raises(ShapeMismatchError, match=expected_name):
        KalmanFilter._shape_validate(*mutator(A, B, Q, R, C, d), nmk=(2, 1, 1))


_EKF_MEAS_SIG = types.void(
    types.CPointer(types.float64),
    types.CPointer(types.float64),
    types.CPointer(types.float64),
)


@cfunc(_EKF_MEAS_SIG)
def _ekf_nl_meas(x, params, out):
    # y0 = x0 + p0 * x1^2 (nonlinear); y1 = x1 (linear)
    out[0] = x[0] + params[0] * x[1] * x[1]
    out[1] = x[1]


@cfunc(_EKF_MEAS_SIG)
def _ekf_nl_jac(x, params, out):
    # dh/dx, row-major (m=2, n=2)
    out[0] = 1.0
    out[1] = 2.0 * params[0] * x[1]
    out[2] = 0.0
    out[3] = 1.0


def _ekf_nl_h(x, params):
    return np.array([x[0] + params[0] * x[1] ** 2, x[1]], dtype=float64)


def _ekf_nl_H(x, params):
    return np.array([[1.0, 2.0 * params[0] * x[1]], [0.0, 1.0]], dtype=float64)


def test_run_extended_raw_matches_ekf_oracle():
    # Native EKF (meas/jac @cfunc addresses) must match the pure-numpy oracle fed
    # the same nonlinear measurement as Python callables.
    A = np.array([[0.5, 0.1], [0.0, 0.7]], dtype=float64)
    B = np.eye(2, dtype=float64)
    Q = np.diag([0.04, 0.09]).astype(float64)
    R = np.diag([0.01, 0.02]).astype(float64)
    params = np.array([0.3], dtype=float64)
    rng = np.random.default_rng(0)
    y = rng.normal(size=(6, 2)).astype(float64)
    x0 = np.array([0.2, -0.1], dtype=float64)
    P0 = np.diag([0.5, 0.3]).astype(float64)

    native = KalmanFilter.run_extended_raw(
        _ekf_nl_meas.address,
        _ekf_nl_jac.address,
        A,
        B,
        params,
        Q,
        R,
        y,
        x0=x0,
        P0=P0,
        return_shocks=True,
    )
    ref = ekf_reference(
        A,
        B,
        _ekf_nl_h,
        _ekf_nl_H,
        params,
        Q,
        R,
        y,
        x0=x0,
        P0=P0,
        return_shocks=True,
    )

    np.testing.assert_allclose(native.x_pred, ref.x_pred)
    np.testing.assert_allclose(native.x_filt, ref.x_filt)
    np.testing.assert_allclose(native.P_pred, ref.P_pred)
    np.testing.assert_allclose(native.P_filt, ref.P_filt)
    np.testing.assert_allclose(native.y_pred, ref.y_pred)
    np.testing.assert_allclose(native.y_filt, ref.y_filt)
    np.testing.assert_allclose(native.innov, ref.innov)
    np.testing.assert_allclose(native.std_innov, ref.std_innov)
    np.testing.assert_allclose(native.S, ref.S)
    np.testing.assert_allclose(native.eps_hat, ref.eps_hat)
    assert native.loglik == pytest.approx(float(ref.loglik))

    # Intrinsic whitening relations on the native output: std_innov = L^{-1} v,
    # and its squared norm is the Mahalanobis distance v^T S^{-1} v.
    L0 = np.linalg.cholesky(native.S[0])
    np.testing.assert_allclose(
        native.std_innov[0], np.linalg.solve(L0, native.innov[0])
    )
    np.testing.assert_allclose(
        native.std_innov[0] @ native.std_innov[0],
        native.innov[0] @ np.linalg.solve(native.S[0], native.innov[0]),
    )


def test_run_converts_error_code_to_matrix_condition(monkeypatch):
    A, B, C, d, Q, R = _linear_system_1d()
    y = np.zeros((2, 1), dtype=float64)

    # The native hot loop returns (err, out); a nonzero status is mapped to the
    # matching in-house exception in one place. No numpy exception leaks out.
    def error_hot_loop(*args, **kwargs):
        return ErrorCode.MATRIX_CONDITION, ()

    monkeypatch.setattr(filter_module, "kalman_hot_loop", error_hot_loop)
    with pytest.raises(MatrixConditionError):
        KalmanFilter.run(A, B, C, d, Q, R, y)


def test_run_extended_converts_error_code_to_matrix_condition(monkeypatch):
    A, B, _, _, Q, R = _linear_system_1d()
    y = np.zeros((2, 1), dtype=float64)

    # A nonzero status from the native hot loop is mapped to the matching
    # in-house exception via the ErrorCode registry (mirrors the linear path).
    # meas/jac addresses are unused because the hot loop is monkeypatched.
    def error_ekf(*args, **kwargs):
        return ErrorCode.MATRIX_CONDITION, ()

    monkeypatch.setattr(filter_module, "ekf_hot_loop", error_ekf)
    with pytest.raises(MatrixConditionError):
        KalmanFilter.run_extended(
            1,
            1,
            A,
            B,
            np.array([0.0], dtype=float64),
            Q,
            R,
            y,
        )
