# type: ignore
from __future__ import annotations

import numpy as np
import pytest
from numba import njit
from numpy import float64

import SymbolicDSGE.kalman.filter as filter_module
from SymbolicDSGE.kalman.filter import (
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


def test_low_level_linear_algebra_helpers_match_numpy():
    S = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=float64)
    b = np.array([1.0, 2.0], dtype=float64)
    B = np.array([[1.0, 0.5], [2.0, -1.0]], dtype=float64)
    L = filter_module._chol_shifted(S, jit=0.0)

    assert np.allclose(L @ L.T, S)
    assert np.allclose(filter_module._forward_subst_vec(L, b), np.linalg.solve(L, b))
    assert np.allclose(
        filter_module._backward_subst_vec(L.T, b),
        np.linalg.solve(L.T, b),
    )
    assert np.allclose(
        filter_module._chol_solve_vec(L, b),
        np.linalg.solve(S, b),
    )
    assert np.allclose(
        filter_module._chol_solve_mat(L, B),
        np.linalg.solve(S, B),
    )
    assert filter_module._logdet_from_chol(L) == pytest.approx(np.log(np.linalg.det(S)))


def test_direct_kalman_hot_loop_covers_return_shocks_branch():
    A, B, C, d, Q, R = _linear_system_1d()
    y = np.full((4, 1), 0.25, dtype=float64)
    x0 = np.array([0.0], dtype=float64)
    P0 = np.array([[1.0]], dtype=float64)

    err, _, out = filter_module._kalman_hot_loop(
        4,
        (1, 1, 1),
        A,
        B,
        C,
        d,
        Q,
        R,
        y,
        x0,
        P0,
        False,
        0.0,
        True,
    )
    assert err == filter_module.OK
    x_pred, x_filt, P_pred, P_filt, y_pred, y_filt, innov, S_hist, eps_hat, loglik = out
    assert x_pred.shape == (4, 1)
    assert x_filt.shape == (4, 1)
    assert P_pred.shape == (4, 1, 1)
    assert P_filt.shape == (4, 1, 1)
    assert y_pred.shape == (4, 1)
    assert y_filt.shape == (4, 1)
    assert innov.shape == (4, 1)
    assert S_hist.shape == (4, 1, 1)
    assert eps_hat.shape == (4, 1)
    assert np.any(np.abs(eps_hat) > 0.0)
    assert np.isfinite(loglik)


def test_direct_extended_hot_loops_cover_python_and_numba_paths():
    A, B, _, _, Q, R = _linear_system_1d()
    y = np.full((3, 1), 0.1, dtype=float64)
    x0 = np.array([0.0], dtype=float64)
    P0 = np.array([[1.0]], dtype=float64)
    calib = np.array([0.05], dtype=float64)

    def h_scalar(x, bias):
        return np.array([x + bias], dtype=float64)

    def H_scalar(x, bias):
        return np.array([[1.0]], dtype=float64)

    err_py, _, out_py = filter_module._ekf_hot_loop_python(
        3,
        (1, 1, 1),
        A,
        B,
        h_scalar,
        H_scalar,
        calib,
        Q,
        R,
        y,
        x0,
        P0,
        True,
        0.0,
        True,
        True,
    )
    assert err_py == filter_module.OK
    assert out_py[5].shape == (3, 1)
    assert out_py[8].shape == (3, 1)
    assert np.any(np.abs(out_py[8]) > 0.0)

    @njit
    def h_array(x, params):
        return np.array([x[0] + params[0]], dtype=float64)

    @njit
    def H_array(x, params):
        return np.array([[1.0]], dtype=float64)

    err_nb, _, out_nb = filter_module._ekf_hot_loop_numba(
        3,
        (1, 1, 1),
        A,
        B,
        h_array,
        H_array,
        calib,
        Q,
        R,
        y,
        x0,
        P0,
        True,
        0.0,
        False,
        True,
    )
    assert err_nb == filter_module.OK
    assert np.array_equal(out_nb[5], np.zeros((3, 1), dtype=float64))
    assert out_nb[8].shape == (3, 1)
    assert np.any(np.abs(out_nb[8]) > 0.0)


def test_run_converts_internal_linalg_and_error_codes(monkeypatch):
    A, B, C, d, Q, R = _linear_system_1d()
    y = np.zeros((2, 1), dtype=float64)

    def raising_hot_loop(*args, **kwargs):
        raise np.linalg.LinAlgError("boom")

    monkeypatch.setattr(filter_module, "_kalman_hot_loop", raising_hot_loop)
    with pytest.raises(MatrixConditionError):
        KalmanFilter.run(A, B, C, d, Q, R, y)

    def error_hot_loop(*args, **kwargs):
        return filter_module.ERR_COND, (123.0,), ()

    monkeypatch.setattr(filter_module, "_kalman_hot_loop", error_hot_loop)
    with pytest.raises(MatrixConditionError, match="123.0"):
        KalmanFilter.run(A, B, C, d, Q, R, y)


def test_run_extended_converts_internal_linalg_and_error_codes(monkeypatch):
    A, B, _, _, Q, R = _linear_system_1d()
    y = np.zeros((2, 1), dtype=float64)

    def h_scalar(x, bias):
        return np.array([x + bias], dtype=float64)

    def H_scalar(x, bias):
        return np.array([[1.0]], dtype=float64)

    def raising_ekf(*args, **kwargs):
        raise np.linalg.LinAlgError("boom")

    monkeypatch.setattr(filter_module, "_ekf_hot_loop_python", raising_ekf)
    with pytest.raises(MatrixConditionError):
        KalmanFilter.run_extended(
            A,
            B,
            h_scalar,
            H_scalar,
            np.array([0.0], dtype=float64),
            Q,
            R,
            y,
        )

    def error_ekf(*args, **kwargs):
        return filter_module.ERR_COND, (321.0,), ()

    monkeypatch.setattr(filter_module, "_ekf_hot_loop_python", error_ekf)
    with pytest.raises(MatrixConditionError, match="321.0"):
        KalmanFilter.run_extended(
            A,
            B,
            h_scalar,
            H_scalar,
            np.array([0.0], dtype=float64),
            Q,
            R,
            y,
        )
