# type: ignore
from __future__ import annotations

import numpy as np
import pytest
from numba import cfunc, types
from numpy import float64

import SymbolicDSGE.filters.filter as filter_module
from SymbolicDSGE.filters.errors import (
    ErrorCode,
    MatrixFactorizationError,
    ShapeMismatchError,
)
from SymbolicDSGE.filters.filter import (
    _shape_validate,
    run_extended,
    run_linear,
)


def _linear_system_1d():
    A = np.array([[0.9]], dtype=float64)
    B = np.array([[1.0]], dtype=float64)
    C = np.array([[1.0]], dtype=float64)
    d = np.array([0.0], dtype=float64)
    Q = np.array([[0.04]], dtype=float64)
    R = np.array([[0.01]], dtype=float64)
    ss = np.zeros(1, dtype=float64)
    return A, B, C, d, Q, R, ss


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
        _shape_validate(*mutator(A, B, Q, R, C, d), nmk=(2, 1, 1))


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


def test_run_extended_whitens_innovations_by_their_covariance():
    A = np.array([[0.5, 0.1], [0.0, 0.7]], dtype=float64)
    B = np.eye(2, dtype=float64)
    Q = np.diag([0.04, 0.09]).astype(float64)
    R = np.diag([0.01, 0.02]).astype(float64)
    params = np.array([0.3], dtype=float64)
    rng = np.random.default_rng(0)
    y = rng.normal(size=(6, 2)).astype(float64)
    x0 = np.array([0.2, -0.1], dtype=float64)
    P0 = np.diag([0.5, 0.3]).astype(float64)
    ss = np.array([0.4, -0.25], dtype=float64)

    native = run_extended(
        _ekf_nl_meas.address,
        _ekf_nl_jac.address,
        A,
        B,
        params,
        Q,
        R,
        ss,
        y,
        x0=x0,
        P0=P0,
    )

    L0 = np.linalg.cholesky(native.S[0])
    np.testing.assert_allclose(
        native.std_innov[0], np.linalg.solve(L0, native.innov[0])
    )
    np.testing.assert_allclose(
        native.std_innov[0] @ native.std_innov[0],
        native.innov[0] @ np.linalg.solve(native.S[0], native.innov[0]),
    )


def test_run_converts_error_code_to_matrix_condition(monkeypatch):
    A, B, C, d, Q, R, ss = _linear_system_1d()
    y = np.zeros((2, 1), dtype=float64)

    def error_kf(*args, **kwargs):
        return ErrorCode.S_FACTORIZATION, ()

    monkeypatch.setattr(filter_module, "kf_forward", error_kf)
    with pytest.raises(MatrixFactorizationError):
        run_linear(A, B, C, d, Q, R, ss, y)


def test_run_extended_converts_error_code_to_matrix_condition(monkeypatch):
    A, B, _, _, Q, R, ss = _linear_system_1d()
    y = np.zeros((2, 1), dtype=float64)

    def error_ekf(*args, **kwargs):
        return ErrorCode.S_FACTORIZATION, ()

    monkeypatch.setattr(filter_module, "ekf_forward", error_ekf)
    with pytest.raises(MatrixFactorizationError):
        run_extended(
            1,
            1,
            A,
            B,
            np.array([0.0], dtype=float64),
            Q,
            R,
            ss,
            y,
        )


# ---------------------------------------------------------------------------
# raise_on_error API (linear + extended).
#
# The native hot loops return (err: int, out: tuple). The runners map a nonzero
# ``err`` to the in-house exception when ``_raise_on_error`` is True, and
# otherwise carry the code on the result's scalar ``status`` field so batch
# callers (e.g. an estimation search) tally failures without a try/except. There
# is deliberately no LinAlgError or other numpy-exception handling: the native
# kernel reports every per-iteration failure as a status code, never as a raised
# numpy error. Pre-loop setup checks (shape/dtype) still raise, by design.
# ---------------------------------------------------------------------------


def _placeholder_out_11():
    # The runner unpacks 11 array/scalar fields from ``out``; only ``status``
    # is under test here, so the payload contents are irrelevant.
    return (None,) * 11


def test_run_status_carries_error_code(monkeypatch):
    A, B, C, d, Q, R, ss = _linear_system_1d()
    y = np.zeros((2, 1), dtype=float64)

    # A clean run reports success on ``status``.
    assert run_linear(A, B, C, d, Q, R, ss, y).status == 0

    def error_kf(*args, **kwargs):
        return ErrorCode.S_FACTORIZATION, _placeholder_out_11()

    monkeypatch.setattr(filter_module, "kf_forward", error_kf)

    # Default: a nonzero status raises the mapped exception.
    with pytest.raises(MatrixFactorizationError):
        run_linear(A, B, C, d, Q, R, ss, y)

    # _raise_on_error=False: the code is carried on ``status`` instead of raising.
    res = run_linear(A, B, C, d, Q, R, ss, y, _raise_on_error=False)
    assert res.status == int(ErrorCode.S_FACTORIZATION)


def test_run_extended_status_carries_error_code(monkeypatch):
    A, B, _, _, Q, R, ss = _linear_system_1d()
    y = np.zeros((2, 1), dtype=float64)
    calib = np.array([0.0], dtype=float64)

    def error_ekf(*args, **kwargs):
        return ErrorCode.S_FACTORIZATION, _placeholder_out_11()

    monkeypatch.setattr(filter_module, "ekf_forward", error_ekf)

    with pytest.raises(MatrixFactorizationError):
        run_extended(1, 1, A, B, calib, Q, R, ss, y)

    res = run_extended(1, 1, A, B, calib, Q, R, ss, y, _raise_on_error=False)
    assert res.status == int(ErrorCode.S_FACTORIZATION)
