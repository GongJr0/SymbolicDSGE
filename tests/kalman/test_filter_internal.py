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


def test_write_into_linear_algebra_helpers_match_numpy():
    S = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=float64)
    b = np.array([1.0, 2.0], dtype=float64)
    B_rows = np.array([[1.0, 0.5], [2.0, -1.0]], dtype=float64)

    sym_input = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=float64)
    np.testing.assert_allclose(
        filter_module._sym.py_func(sym_input),
        np.array([[1.0, 2.5], [2.5, 4.0]], dtype=float64),
    )

    L_old = filter_module._chol_shifted.py_func(S, jit=0.0)
    L_old_jitter = filter_module._chol_shifted.py_func(
        np.array([[0.0]], dtype=float64), jit=1e-8
    )
    np.testing.assert_allclose(L_old @ L_old.T, S)
    np.testing.assert_allclose(L_old_jitter, np.array([[1e-4]], dtype=float64))
    np.testing.assert_allclose(
        filter_module._forward_subst_vec.py_func(L_old, b),
        np.linalg.solve(L_old, b),
    )
    np.testing.assert_allclose(
        filter_module._backward_subst_vec.py_func(L_old.T, b),
        np.linalg.solve(L_old.T, b),
    )
    np.testing.assert_allclose(
        filter_module._chol_solve_mat.py_func(L_old, B_rows.T),
        np.linalg.solve(S, B_rows.T),
    )

    zero_target = np.full((2, 2), 7.0, dtype=float64)
    filter_module._zero_mat_into.py_func(zero_target)
    np.testing.assert_allclose(zero_target, np.zeros((2, 2), dtype=float64))

    sym_target = sym_input.copy()
    filter_module._sym_inplace.py_func(sym_target)
    np.testing.assert_allclose(sym_target, filter_module._sym.py_func(sym_input))

    L = np.empty_like(S)
    filter_module._chol_shifted_into.py_func(S, 0.0, L)
    np.testing.assert_allclose(L @ L.T, S)

    L_jitter = np.empty((1, 1), dtype=float64)
    filter_module._chol_shifted_into.py_func(
        np.array([[0.0]], dtype=float64), 1e-8, L_jitter
    )
    np.testing.assert_allclose(L_jitter, np.array([[1e-4]], dtype=float64))

    with pytest.raises(np.linalg.LinAlgError):
        filter_module._chol_shifted_into.py_func(
            np.array([[0.0]], dtype=float64), 0.0, L_jitter
        )

    forward = np.empty_like(b)
    backward = np.empty_like(b)
    filter_module._forward_subst_vec_into.py_func(L, b, forward)
    filter_module._backward_subst_chol_t_vec_into.py_func(L, forward, backward)
    np.testing.assert_allclose(forward, np.linalg.solve(L, b))
    np.testing.assert_allclose(backward, np.linalg.solve(S, b))

    chol_solve_rows = np.empty_like(B_rows)
    for row in range(B_rows.shape[0]):
        filter_module._chol_solve_row_into.py_func(
            L,
            B_rows,
            row,
            np.empty_like(b),
            np.empty_like(b),
            chol_solve_rows,
        )
    np.testing.assert_allclose(chol_solve_rows, np.linalg.solve(S, B_rows.T).T)

    assert filter_module._logdet_from_chol.py_func(L) == pytest.approx(
        np.log(np.linalg.det(S))
    )


def test_write_into_kalman_covariance_helpers_match_numpy():
    A = np.array([[0.8, 0.1], [0.0, 0.6]], dtype=float64)
    B = np.array([[1.0, 0.2], [0.5, -0.4]], dtype=float64)
    C = np.array([[1.0, 0.0], [0.2, 0.9]], dtype=float64)
    Q = np.array([[0.5, 0.1], [0.1, 0.3]], dtype=float64)
    R = np.array([[0.2, 0.05], [0.05, 0.4]], dtype=float64)
    P_prev = np.array([[1.5, 0.2], [0.2, 0.8]], dtype=float64)
    x = np.array([0.4, -0.2], dtype=float64)
    d = np.array([0.1, -0.3], dtype=float64)
    v = np.array([0.25, -0.1], dtype=float64)

    mat_out = np.empty((2, 2), dtype=float64)
    filter_module._matmul_into.py_func(A, P_prev, mat_out)
    np.testing.assert_allclose(mat_out, A @ P_prev)

    abt_out = np.empty((2, 2), dtype=float64)
    filter_module._matmul_abt_into.py_func(A, B, abt_out)
    np.testing.assert_allclose(abt_out, A @ B.T)

    abt_plus_out = np.empty((2, 2), dtype=float64)
    filter_module._matmul_abt_plus_c_into.py_func(A, B, R, abt_plus_out)
    np.testing.assert_allclose(abt_plus_out, A @ B.T + R)

    vec_out = np.empty((2,), dtype=float64)
    filter_module._matvec_into.py_func(A, x, vec_out)
    np.testing.assert_allclose(vec_out, A @ x)
    filter_module._matvec_plus_vec_into.py_func(C, x, d, vec_out)
    np.testing.assert_allclose(vec_out, C @ x + d)

    row_diff = np.empty((2,), dtype=float64)
    y = np.array([[0.0, 1.0], [0.5, -0.7]], dtype=float64)
    filter_module._row_minus_vec_into.py_func(y, 1, vec_out, row_diff)
    np.testing.assert_allclose(row_diff, y[1] - vec_out)
    assert filter_module._dot_vec.py_func(row_diff, vec_out) == pytest.approx(
        row_diff @ vec_out
    )

    BQBT = np.empty((2, 2), dtype=float64)
    filter_module._build_bqbt_into.py_func(B, Q, np.empty((2, 2), dtype=float64), BQBT)
    np.testing.assert_allclose(BQBT, B @ Q @ B.T)
    np.testing.assert_allclose(BQBT, BQBT.T)

    P_pred = np.empty((2, 2), dtype=float64)
    filter_module._predict_cov_into.py_func(
        A, P_prev, BQBT, np.empty((2, 2), dtype=float64), P_pred
    )
    np.testing.assert_allclose(P_pred, A @ P_prev @ A.T + BQBT)

    S = np.empty((2, 2), dtype=float64)
    filter_module._measurement_cov_into.py_func(
        C, P_pred, R, np.empty((2, 2), dtype=float64), S
    )
    np.testing.assert_allclose(S, C @ P_pred @ C.T + R)

    PCt = np.empty((2, 2), dtype=float64)
    filter_module._pc_t_into.py_func(P_pred, C, PCt)
    np.testing.assert_allclose(PCt, P_pred @ C.T)

    L = np.linalg.cholesky(S).astype(float64)
    K = np.empty((2, 2), dtype=float64)
    filter_module._gain_from_pc_t_into.py_func(
        L, PCt, np.empty((2,), dtype=float64), np.empty((2,), dtype=float64), K
    )
    np.testing.assert_allclose(K, np.linalg.solve(S, PCt.T).T)

    x_filt = np.empty((2,), dtype=float64)
    filter_module._state_update_into.py_func(x, K, v, x_filt)
    np.testing.assert_allclose(x_filt, x + K @ v)

    I_minus = np.empty((2, 2), dtype=float64)
    filter_module._identity_minus_into.py_func(K @ C, I_minus)
    np.testing.assert_allclose(I_minus, np.eye(2, dtype=float64) - K @ C)

    P_filt = np.empty((2, 2), dtype=float64)
    filter_module._joseph_cov_into.py_func(
        K,
        C,
        P_pred,
        R,
        np.empty((2, 2), dtype=float64),
        np.empty((2, 2), dtype=float64),
        np.empty((2, 2), dtype=float64),
        np.empty((2, 2), dtype=float64),
        P_filt,
    )
    expected_p_filt = I_minus @ P_pred @ I_minus.T + K @ R @ K.T
    np.testing.assert_allclose(P_filt, expected_p_filt)

    shock_projection = np.empty((2, 2), dtype=float64)
    filter_module._build_shock_projection_into.py_func(
        B, C, Q, np.empty((2, 2), dtype=float64), shock_projection
    )
    np.testing.assert_allclose(shock_projection, Q @ (B.T @ C.T))


def test_direct_kalman_hot_loop_covers_return_shocks_branch():
    A, B, C, d, Q, R = _linear_system_1d()
    y = np.full((4, 1), 0.25, dtype=float64)
    x0 = np.array([0.0], dtype=float64)
    P0 = np.array([[1.0]], dtype=float64)

    err, _, out = filter_module._kalman_hot_loop.py_func(
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
    (
        x_pred,
        x_filt,
        P_pred,
        P_filt,
        y_pred,
        y_filt,
        innov,
        std_innov,
        S_hist,
        eps_hat,
        loglik,
    ) = out
    assert x_pred.shape == (4, 1)
    assert x_filt.shape == (4, 1)
    assert P_pred.shape == (4, 1, 1)
    assert P_filt.shape == (4, 1, 1)
    assert y_pred.shape == (4, 1)
    assert y_filt.shape == (4, 1)
    assert innov.shape == (4, 1)
    assert std_innov.shape == (4, 1)
    assert S_hist.shape == (4, 1, 1)
    assert eps_hat.shape == (4, 1)
    assert np.any(np.abs(eps_hat) > 0.0)
    assert np.isfinite(loglik)

    L0 = np.linalg.cholesky(S_hist[0])
    expected_std0 = np.linalg.solve(L0, innov[0])
    assert np.allclose(std_innov[0], expected_std0)
    assert np.allclose(
        std_innov[0] @ std_innov[0],
        innov[0] @ np.linalg.solve(S_hist[0], innov[0]),
    )

    err_sym, _, out_sym = filter_module._kalman_hot_loop.py_func(
        2,
        (1, 1, 1),
        A,
        B,
        C,
        d,
        Q,
        R,
        y[:2],
        x0,
        P0,
        True,
        0.0,
        False,
    )
    assert err_sym == filter_module.OK
    assert np.allclose(out_sym[2], np.transpose(out_sym[2], (0, 2, 1)))
    assert np.allclose(out_sym[3], np.transpose(out_sym[3], (0, 2, 1)))


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
    assert out_py[7].shape == (3, 1)
    assert out_py[9].shape == (3, 1)
    assert np.any(np.abs(out_py[9]) > 0.0)

    innov_py = out_py[6]
    std_innov_py = out_py[7]
    S_py = out_py[8]
    L_py0 = np.linalg.cholesky(S_py[0])
    assert np.allclose(std_innov_py[0], np.linalg.solve(L_py0, innov_py[0]))
    assert np.allclose(
        std_innov_py[0] @ std_innov_py[0],
        innov_py[0] @ np.linalg.solve(S_py[0], innov_py[0]),
    )

    @njit
    def h_array(x, params):
        return np.array([x[0] + params[0]], dtype=float64)

    @njit
    def H_array(x, params):
        return np.array([[1.0]], dtype=float64)

    err_nb, _, out_nb = filter_module._ekf_hot_loop_numba.py_func(
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
    assert out_nb[7].shape == (3, 1)
    assert out_nb[9].shape == (3, 1)
    assert np.any(np.abs(out_nb[9]) > 0.0)

    innov_nb = out_nb[6]
    std_innov_nb = out_nb[7]
    S_nb = out_nb[8]
    L_nb0 = np.linalg.cholesky(S_nb[0])
    assert np.allclose(std_innov_nb[0], np.linalg.solve(L_nb0, innov_nb[0]))
    assert np.allclose(
        std_innov_nb[0] @ std_innov_nb[0],
        innov_nb[0] @ np.linalg.solve(S_nb[0], innov_nb[0]),
    )

    err_nb_y, _, out_nb_y = filter_module._ekf_hot_loop_numba.py_func(
        1,
        (1, 1, 1),
        A,
        B,
        h_array,
        H_array,
        calib,
        Q,
        R,
        y[:1],
        x0,
        P0,
        True,
        0.0,
        True,
        False,
    )
    assert err_nb_y == filter_module.OK
    assert np.isfinite(out_nb_y[5]).all()


def test_run_extended_uses_numba_array_dispatch_branch():
    A, B, _, _, Q, R = _linear_system_1d()
    y = np.full((2, 1), 0.1, dtype=float64)
    calib = np.array([0.05], dtype=float64)

    @njit
    def h_array(x, params):
        return np.array([x[0] + params[0]], dtype=float64)

    @njit
    def H_array(x, params):
        return np.array([[1.0]], dtype=float64)

    h_array._symbolicdsge_array_dispatch = True
    H_array._symbolicdsge_array_dispatch = True

    out = KalmanFilter.run_extended(
        A,
        B,
        h_array,
        H_array,
        calib,
        Q,
        R,
        y,
        compute_y_filt=True,
        return_shocks=True,
    )

    assert out.y_filt.shape == (2, 1)
    assert out.eps_hat is not None
    assert out.eps_hat.shape == (2, 1)
    assert np.isfinite(out.loglik)


def test_extended_dispatch_helpers_cover_array_dispatch_branches():
    def h_array(state, params):
        return np.array([state[0] + params[0]], dtype=float64)

    def H_array(state, params):
        return np.array([[1.0 + params[0] * 0.0]], dtype=float64)

    h_array._symbolicdsge_array_dispatch = True
    H_array._symbolicdsge_array_dispatch = True

    state = np.array([1.0], dtype=float64)
    params = np.array([0.25], dtype=float64)

    assert filter_module._is_numba_array_dispatch(h_array)
    np.testing.assert_allclose(
        filter_module._call_extended_measurement(h_array, state, params),
        np.array([1.25], dtype=float64),
    )
    np.testing.assert_allclose(
        filter_module._call_extended_jacobian(H_array, state, params),
        np.array([[1.0]], dtype=float64),
    )


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
