# type: ignore
from __future__ import annotations

import numpy as np
import pytest
from numba import cfunc, types
from numpy import float64
from sympy import Symbol

from SymbolicDSGE.core.model_parser import make_R
from SymbolicDSGE.kalman.errors import ErrorCode, get_error_constructor
from SymbolicDSGE.kalman.filter import (
    ComplexMatrixError,
    FilterRawResult,
    KalmanFilter,
    MatrixConditionError,
    ShapeMismatchError,
    UnscentedFilterRawResult,
)


def _linear_system_1d():
    A = np.array([[0.9]], dtype=float64)
    B = np.array([[1.0]], dtype=float64)
    C = np.array([[1.0]], dtype=float64)
    d = np.array([0.0], dtype=float64)
    Q = np.array([[0.04]], dtype=float64)
    R = np.array([[0.01]], dtype=float64)
    return A, B, C, d, Q, R


_UKF_MEAS_SIG = types.void(
    types.CPointer(types.float64),
    types.CPointer(types.float64),
    types.CPointer(types.float64),
)


@cfunc(_UKF_MEAS_SIG)
def _ukf_measurement(model_vars, params, out):
    out[0] = model_vars[0] + params[0] * model_vars[1]


# 1-D linear measurement h(x) = x and its (constant) jacobian, as @cfuncs for the
# EKF address-based API. The linear measurement lets the EKF be checked against
# the exact linear Kalman filter.
@cfunc(_UKF_MEAS_SIG)
def _ekf_lin_meas(x, params, out):
    out[0] = x[0]


@cfunc(_UKF_MEAS_SIG)
def _ekf_lin_jac(x, params, out):
    out[0] = 1.0


@cfunc(_UKF_MEAS_SIG)
def _ekf_bias_meas(x, params, out):
    out[0] = x[0] + params[0]


def _ukf_system_1d():
    return {
        "meas_addr": _ukf_measurement.address,
        "hx": np.array([[0.65]], dtype=float64),
        "gx": np.array([[1.5]], dtype=float64),
        "bx": np.array([[1.0]], dtype=float64),
        "hxx": np.array([[[0.05]]], dtype=float64),
        "gxx": np.array([[[0.1]]], dtype=float64),
        "hss": np.array([0.02], dtype=float64),
        "gss": np.array([0.04], dtype=float64),
        "steady_state": np.array([2.0, 3.0], dtype=float64),
        "calib_params": np.array([0.25], dtype=float64),
        "Q": np.array([[0.03]], dtype=float64),
        "R": np.array([[0.2]], dtype=float64),
        "y": np.array([[2.8], [2.7], [2.65], [2.6]], dtype=float64),
        "z0": np.array([0.1, 0.0], dtype=float64),
        "P0": np.diag(np.array([0.4, 0.2], dtype=float64)),
        "alpha": 1.0,
        "beta": 2.0,
        "kappa": 1.0,
        "jitter": 1e-12,
        "symmetrize": True,
    }


def test_make_r_builds_covariance_from_std_and_corr_maps():
    obs_a = Symbol("ObsA")
    obs_b = Symbol("ObsB")

    out = make_R(
        [obs_a, obs_b],
        {obs_a: float64(2.0), obs_b: float64(3.0)},
        {frozenset({obs_a, obs_b}): float64(0.25)},
    )

    np.testing.assert_allclose(
        out,
        np.array([[4.0, 1.5], [1.5, 9.0]], dtype=float64),
    )


def test_error_code_dispatch_maps_known_errors_and_rejects_unknown():
    assert get_error_constructor(ErrorCode.COMPLEX_MATRIX) is ComplexMatrixError
    assert get_error_constructor(ErrorCode.SHAPE_MISMATCH) is ShapeMismatchError
    assert get_error_constructor(ErrorCode.MATRIX_CONDITION) is MatrixConditionError
    assert get_error_constructor(ErrorCode.LINALG_ERROR) is np.linalg.LinAlgError

    with pytest.raises(ValueError, match="Unknown error code"):
        get_error_constructor(ErrorCode.SUCCESS)


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


def test_run_raw_linear_matches_public_result():
    A, B, C, d, Q, R = _linear_system_1d()
    y = np.zeros((5, 1), dtype=float64)
    x0 = np.array([2.0], dtype=float64)
    P0 = np.array([[1.0]], dtype=float64)

    raw = KalmanFilter.run_raw(
        A,
        B,
        C,
        d,
        Q,
        R,
        y,
        x0=x0,
        P0=P0,
        return_shocks=True,
    )
    public = KalmanFilter.run(
        A,
        B,
        C,
        d,
        Q,
        R,
        y,
        x0=x0,
        P0=P0,
        return_shocks=True,
    )

    assert isinstance(raw, FilterRawResult)
    assert raw.eps_hat is not None
    np.testing.assert_allclose(raw.x_pred, public.x_pred)
    np.testing.assert_allclose(raw.x_filt, public.x_filt)
    np.testing.assert_allclose(raw.innov, public.innov)
    np.testing.assert_allclose(raw.std_innov, public.std_innov)
    np.testing.assert_allclose(raw.S, public.S)
    assert raw.loglik == pytest.approx(public.loglik)


def test_run_linear_can_skip_history_storage_for_loglik_only_path():
    A, B, C, d, Q, R = _linear_system_1d()
    y = np.zeros((5, 1), dtype=float64)
    x0 = np.array([2.0], dtype=float64)
    P0 = np.array([[1.0]], dtype=float64)

    full = KalmanFilter.run(A, B, C, d, Q, R, y, x0=x0, P0=P0)
    minimal = KalmanFilter.run(
        A,
        B,
        C,
        d,
        Q,
        R,
        y,
        x0=x0,
        P0=P0,
        _store_history=False,
    )

    assert np.allclose(minimal.loglik, full.loglik)
    assert minimal.x_pred.shape == (0, 1)
    assert minimal.x_filt.shape == (0, 1)
    assert minimal.P_pred.shape == (0, 1, 1)
    assert minimal.P_filt.shape == (0, 1, 1)
    assert minimal.y_pred.shape == (0, 1)
    assert minimal.y_filt.shape == (0, 1)
    assert minimal.innov.shape == (0, 1)
    assert minimal.std_innov.shape == (0, 1)
    assert minimal.S.shape == (0, 1, 1)
    assert minimal.eps_hat is None


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


def test_run_unscented_outputs_shapes_and_projected_fields():
    pytest.importorskip("SymbolicDSGE._ckernels.kalman")
    kwargs = _ukf_system_1d()

    out = KalmanFilter.run_unscented(**kwargs)

    T = kwargs["y"].shape[0]
    n_state = kwargs["hx"].shape[0]
    n_var = kwargs["steady_state"].shape[0]
    n_obs = kwargs["y"].shape[1]
    nz = 2 * n_state
    assert out.x_pred.shape == (T, n_var)
    assert out.x_filt.shape == (T, n_var)
    assert out.x1_pred.shape == (T, n_state)
    assert out.x2_pred.shape == (T, n_state)
    assert out.x1_filt.shape == (T, n_state)
    assert out.x2_filt.shape == (T, n_state)
    assert out.P_pred.shape == (T, nz, nz)
    assert out.P_filt.shape == (T, nz, nz)
    assert out.y_pred.shape == (T, n_obs)
    assert out.y_filt.shape == (T, n_obs)
    assert out.innov.shape == (T, n_obs)
    assert out.std_innov.shape == (T, n_obs)
    assert out.S.shape == (T, n_obs, n_obs)
    assert np.isfinite(out.loglik)

    steady_state = kwargs["steady_state"]
    params = kwargs["calib_params"]
    np.testing.assert_allclose(
        out.x_pred[:, 0],
        steady_state[0] + out.x1_pred[:, 0] + out.x2_pred[:, 0],
    )
    np.testing.assert_allclose(
        out.x_filt[:, 0],
        steady_state[0] + out.x1_filt[:, 0] + out.x2_filt[:, 0],
    )
    np.testing.assert_allclose(
        out.y_filt[:, 0],
        out.x_filt[:, 0] + params[0] * out.x_filt[:, 1],
    )
    np.testing.assert_allclose(out.P_filt, np.transpose(out.P_filt, (0, 2, 1)))


def test_run_raw_unscented_matches_public_result():
    pytest.importorskip("SymbolicDSGE._ckernels.kalman")
    kwargs = _ukf_system_1d()

    raw = KalmanFilter.run_unscented_raw(**kwargs)
    public = KalmanFilter.run_unscented(**kwargs)

    assert isinstance(raw, UnscentedFilterRawResult)
    np.testing.assert_allclose(raw.x_pred, public.x_pred)
    np.testing.assert_allclose(raw.x_filt, public.x_filt)
    np.testing.assert_allclose(raw.x1_pred, public.x1_pred)
    np.testing.assert_allclose(raw.x2_pred, public.x2_pred)
    np.testing.assert_allclose(raw.y_pred, public.y_pred)
    np.testing.assert_allclose(raw.y_filt, public.y_filt)
    assert raw.loglik == pytest.approx(public.loglik)


def test_run_unscented_can_skip_history_storage_for_loglik_only_path():
    pytest.importorskip("SymbolicDSGE._ckernels.kalman")
    kwargs = _ukf_system_1d()

    full = KalmanFilter.run_unscented(**kwargs)
    minimal = KalmanFilter.run_unscented(**kwargs, _store_history=False)

    assert np.allclose(minimal.loglik, full.loglik)
    assert minimal.x_pred.shape == (0, 2)
    assert minimal.x_filt.shape == (0, 2)
    assert minimal.x1_pred.shape == (0, 1)
    assert minimal.x2_pred.shape == (0, 1)
    assert minimal.x1_filt.shape == (0, 1)
    assert minimal.x2_filt.shape == (0, 1)
    assert minimal.P_pred.shape == (0, 2, 2)
    assert minimal.P_filt.shape == (0, 2, 2)
    assert minimal.y_pred.shape == (0, 1)
    assert minimal.y_filt.shape == (0, 1)
    assert minimal.innov.shape == (0, 1)
    assert minimal.std_innov.shape == (0, 1)
    assert minimal.S.shape == (0, 1, 1)


def test_run_unscented_rejects_degenerate_inputs():
    pytest.importorskip("SymbolicDSGE._ckernels.kalman")
    kwargs = _ukf_system_1d()

    with pytest.raises(ValueError, match="meas_addr"):
        KalmanFilter.run_unscented(**(kwargs | {"meas_addr": 0}))

    with pytest.raises(ShapeMismatchError, match="hxx"):
        KalmanFilter.run_unscented(
            **(kwargs | {"hxx": np.zeros((1, 1), dtype=float64)})
        )

    with pytest.raises(MatrixConditionError):
        KalmanFilter.run_unscented(
            **(
                kwargs
                | {
                    "P0": np.zeros((2, 2), dtype=float64),
                    "Q": np.zeros((1, 1), dtype=float64),
                    "jitter": 0.0,
                }
            )
        )


def test_run_extended_matches_linear_when_measurement_is_linear():
    A, B, C, d, Q, R = _linear_system_1d()
    y = np.zeros((5, 1), dtype=float64)
    x0 = np.array([1.5], dtype=float64)
    P0 = np.array([[0.7]], dtype=float64)
    calib = np.array([], dtype=float64)

    linear = KalmanFilter.run(A, B, C, d, Q, R, y, x0=x0, P0=P0)
    extended = KalmanFilter.run_extended(
        meas_addr=_ekf_lin_meas.address,
        jac_addr=_ekf_lin_jac.address,
        A=A,
        B=B,
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


def test_run_raw_extended_matches_public_result():
    A, B, _, _, Q, R = _linear_system_1d()
    y = np.zeros((5, 1), dtype=float64)
    x0 = np.array([1.5], dtype=float64)
    P0 = np.array([[0.7]], dtype=float64)
    calib = np.array([0.0], dtype=float64)

    raw = KalmanFilter.run_extended_raw(
        _ekf_bias_meas.address,
        _ekf_lin_jac.address,
        A,
        B,
        calib,
        Q,
        R,
        y,
        x0=x0,
        P0=P0,
    )
    public = KalmanFilter.run_extended(
        _ekf_bias_meas.address,
        _ekf_lin_jac.address,
        A,
        B,
        calib,
        Q,
        R,
        y,
        x0=x0,
        P0=P0,
    )

    assert isinstance(raw, FilterRawResult)
    np.testing.assert_allclose(raw.x_pred, public.x_pred)
    np.testing.assert_allclose(raw.x_filt, public.x_filt)
    np.testing.assert_allclose(raw.y_pred, public.y_pred)
    np.testing.assert_allclose(raw.innov, public.innov)
    assert raw.loglik == pytest.approx(public.loglik)


def test_run_extended_can_skip_history_storage_for_loglik_only_path():
    A, B, _, _, Q, R = _linear_system_1d()
    y = np.zeros((5, 1), dtype=float64)
    calib = np.array([], dtype=float64)

    kwargs = dict(
        meas_addr=_ekf_lin_meas.address,
        jac_addr=_ekf_lin_jac.address,
        A=A,
        B=B,
        calib_params=calib,
        Q=Q,
        R=R,
        y=y,
        compute_y_filt=False,
    )
    full = KalmanFilter.run_extended(**kwargs)
    minimal = KalmanFilter.run_extended(**kwargs, _store_history=False)

    assert np.allclose(minimal.loglik, full.loglik)
    assert minimal.x_pred.shape == (0, 1)
    assert minimal.x_filt.shape == (0, 1)
    assert minimal.P_pred.shape == (0, 1, 1)
    assert minimal.P_filt.shape == (0, 1, 1)
    assert minimal.y_pred.shape == (0, 1)
    assert minimal.y_filt.shape == (0, 1)
    assert minimal.innov.shape == (0, 1)
    assert minimal.std_innov.shape == (0, 1)
    assert minimal.S.shape == (0, 1, 1)
    assert minimal.eps_hat is None


def test_run_extended_compute_y_filt_false_and_return_shocks():
    A, B, _, _, Q, R = _linear_system_1d()
    y = np.zeros((4, 1), dtype=float64)
    calib = np.array([], dtype=float64)
    out_true = KalmanFilter.run_extended(
        meas_addr=_ekf_lin_meas.address,
        jac_addr=_ekf_lin_jac.address,
        A=A,
        B=B,
        calib_params=calib,
        Q=Q,
        R=R,
        y=y,
        compute_y_filt=True,
        return_shocks=True,
    )

    out = KalmanFilter.run_extended(
        meas_addr=_ekf_lin_meas.address,
        jac_addr=_ekf_lin_jac.address,
        A=A,
        B=B,
        calib_params=calib,
        Q=Q,
        R=R,
        y=y,
        compute_y_filt=False,
        return_shocks=True,
    )

    assert np.array_equal(out.y_filt, np.zeros_like(out.y_filt))
    assert np.allclose(out.loglik, out_true.loglik)
    assert out.eps_hat is not None
    assert out.eps_hat.shape == (4, 1)
