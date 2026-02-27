# type: ignore
from __future__ import annotations

import numpy as np
import pytest
from numpy import float64

from SymbolicDSGE.kalman.filter import (
    ComplexMatrixError,
    KalmanFilter,
    MatrixConditionError,
    ShapeMismatchError,
)


def _linear_system_1d():
    A = np.array([[0.9]], dtype=float64)
    B = np.array([[1.0]], dtype=float64)
    C = np.array([[1.0]], dtype=float64)
    d = np.array([0.0], dtype=float64)
    Q = np.array([[0.04]], dtype=float64)
    R = np.array([[0.01]], dtype=float64)
    return A, B, C, d, Q, R


def test_get_real_accepts_nearly_real_complex():
    mat = np.array([[1.0 + 1e-14j, 2.0 - 1e-14j]], dtype=np.complex128)
    out = KalmanFilter._get_real(mat, "M")

    assert out.dtype == float64
    assert np.allclose(out, np.array([[1.0, 2.0]], dtype=float64))


def test_get_real_rejects_significant_imaginary_parts():
    mat = np.array([[1.0 + 0.5j]], dtype=np.complex128)
    with pytest.raises(ComplexMatrixError):
        KalmanFilter._get_real(mat, "bad")


def test_shape_validate_raises_on_bad_A_shape():
    A = np.eye(2, dtype=float64)
    B = np.ones((2, 1), dtype=float64)
    Q = np.eye(1, dtype=float64)
    R = np.eye(1, dtype=float64)
    C = np.ones((1, 2), dtype=float64)
    d = np.zeros((1,), dtype=float64)

    with pytest.raises(ShapeMismatchError):
        KalmanFilter._shape_validate(A[:1], B, Q, R, C, d, nmk=(2, 1, 1))


def test_sym_returns_explicitly_symmetric_matrix():
    P = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float64)
    S = KalmanFilter._sym(P)
    assert np.allclose(S, S.T)
    assert np.allclose(S, np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float64))


def test_chol_and_chol_solve_paths():
    S = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=float64)
    B = np.array([[1.0], [2.0]], dtype=float64)
    L = KalmanFilter._chol(S, jit=0.0)

    x_chol = KalmanFilter._chol_solve(L, S, B)
    x_np = np.linalg.solve(S, B)

    assert L is not None
    assert np.allclose(x_chol, x_np)


def test_chol_with_zero_jitter_returns_none_on_non_pd():
    S = np.array([[0.0]], dtype=float64)
    L = KalmanFilter._chol(S, jit=0.0)
    assert L is None


def test_chol_uses_jitter_when_needed():
    S = np.array([[0.0]], dtype=float64)
    L = KalmanFilter._chol(S, jit=1e-8)
    assert L is not None
    assert np.allclose(L, np.array([[1e-4]], dtype=float64))


def test_chol_solve_raises_on_ill_conditioned_matrix_without_chol():
    S = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-15]], dtype=float64)
    B = np.array([[1.0], [1.0]], dtype=float64)
    with pytest.raises(MatrixConditionError):
        KalmanFilter._chol_solve(None, S, B)


def test_logdet_with_and_without_cholesky():
    S = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=float64)
    L = np.linalg.cholesky(S).astype(float64)

    ld_from_chol = KalmanFilter._logdet(L, S)
    ld_from_slogdet = KalmanFilter._logdet(None, S)

    assert np.allclose(ld_from_chol, np.log(6.0))
    assert np.allclose(ld_from_slogdet, np.log(6.0))


def test_logdet_raises_when_slogdet_sign_nonpositive():
    S = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float64)
    with pytest.raises(np.linalg.LinAlgError):
        KalmanFilter._logdet(None, S)


def test_run_linear_outputs_shapes_and_first_prediction():
    A, B, C, d, Q, R = _linear_system_1d()
    y = np.zeros((5, 1), dtype=float64)
    x0 = np.array([2.0], dtype=float64)
    P0 = np.array([[1.0]], dtype=float64)

    out = KalmanFilter.run(A, B, C, d, Q, R, y, x0=x0, P0=P0)

    assert out.x_pred.shape == (5, 1)
    assert out.x_filt.shape == (5, 1)
    assert out.P_pred.shape == (5, 1, 1)
    assert out.P_filt.shape == (5, 1, 1)
    assert out.y_pred.shape == (5, 1)
    assert out.y_filt.shape == (5, 1)
    assert out.innov.shape == (5, 1)
    assert out.S.shape == (5, 1, 1)
    assert out.eps_hat is None
    assert np.isfinite(out.loglik)

    # x_{0| -1} = A x0
    assert np.allclose(out.x_pred[0], A @ x0)
    assert np.allclose(out.y_pred[0], C @ out.x_pred[0] + d)
    assert np.allclose(out.P_filt, np.transpose(out.P_filt, (0, 2, 1)))


def test_run_linear_return_shocks_and_complex_inputs():
    A, B, C, d, Q, R = _linear_system_1d()
    y = np.zeros((4, 1), dtype=float64)

    out = KalmanFilter.run(
        A.astype(np.complex128),
        B.astype(np.complex128),
        C.astype(np.complex128),
        d.astype(np.complex128),
        Q.astype(np.complex128),
        R.astype(np.complex128),
        y.astype(np.complex128),
        return_shocks=True,
    )

    assert out.eps_hat is not None
    assert out.eps_hat.shape == (4, 1)


def test_run_linear_raises_on_singular_innovation_covariance():
    A = np.eye(1, dtype=float64)
    B = np.eye(1, dtype=float64)
    C = np.eye(1, dtype=float64)
    d = np.zeros((1,), dtype=float64)
    Q = np.zeros((1, 1), dtype=float64)
    R = np.zeros((1, 1), dtype=float64)
    y = np.zeros((2, 1), dtype=float64)

    with pytest.raises(MatrixConditionError):
        KalmanFilter.run(A, B, C, d, Q, R, y, P0=np.zeros((1, 1), dtype=float64))


def test_run_extended_matches_linear_when_measurement_is_linear():
    A, B, C, d, Q, R = _linear_system_1d()
    y = np.zeros((5, 1), dtype=float64)
    x0 = np.array([1.5], dtype=float64)
    P0 = np.array([[0.7]], dtype=float64)
    calib = np.array([], dtype=float64)

    linear = KalmanFilter.run(A, B, C, d, Q, R, y, x0=x0, P0=P0)
    extended = KalmanFilter.run_extended(
        A,
        B,
        h=lambda x: np.array([x], dtype=float64),
        H_jac=lambda x: np.array([[1.0]], dtype=float64),
        calib_params=calib,
        Q=Q,
        R=R,
        y=y,
        x0=x0,
        P0=P0,
    )

    assert np.allclose(extended.x_pred, linear.x_pred)
    assert np.allclose(extended.x_filt, linear.x_filt)
    assert np.allclose(extended.y_pred, linear.y_pred)
    assert np.allclose(extended.P_filt, linear.P_filt)
    assert np.allclose(extended.loglik, linear.loglik)


def test_run_extended_compute_y_filt_false_and_return_shocks():
    A, B, _, _, Q, R = _linear_system_1d()
    y = np.zeros((4, 1), dtype=float64)
    calib = np.array([], dtype=float64)

    out = KalmanFilter.run_extended(
        A,
        B,
        h=lambda x: np.array([x], dtype=float64),
        H_jac=lambda x: np.array([[1.0]], dtype=float64),
        calib_params=calib,
        Q=Q,
        R=R,
        y=y,
        compute_y_filt=False,
        return_shocks=True,
    )

    assert np.array_equal(out.y_filt, out.y_pred)
    assert out.eps_hat is not None
    assert out.eps_hat.shape == (4, 1)
