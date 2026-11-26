# type: ignore
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numba import cfunc, types
from numpy import float64

from SymbolicDSGE.filters.errors import (
    ErrorCode,
    HyperparameterError,
    MatrixFactorizationError,
    ShapeMismatchError,
    get_error_constructor,
)
from SymbolicDSGE.filters.filter import (
    _shape_validate,
    run_extended,
    run_linear,
    run_unscented,
    smooth_extended,
    smooth_linear,
    smooth_unscented,
)
from SymbolicDSGE.filters.gaussian import ExtendedKalman, LinearKalman, UnscentedKalman


def _linear_system_1d():
    A = np.array([[0.9]], dtype=float64)
    B = np.array([[1.0]], dtype=float64)
    C = np.array([[1.0]], dtype=float64)
    d = np.array([0.0], dtype=float64)
    Q = np.array([[0.04]], dtype=float64)
    R = np.array([[0.01]], dtype=float64)
    # A zero expansion point, so the state series this system reports are the
    # recursion's own output and the assertions below read as the recursion.
    # The offset itself is covered on `_system` below and end to end in
    # tests/filters/test_resolvers.py.
    ss = np.zeros(1, dtype=float64)
    return A, B, C, d, Q, R, ss


def _system(steady_state=(1.5, -0.7, 2.0, 0.4), curvature=0.0):
    """Four states, two observables, and three shocks."""
    ss = np.array(steady_state, dtype=float64)
    C = np.array([[1.0, 0.5, 0.0, -0.4], [0.0, 0.0, 0.0, 0.0]], dtype=float64)
    d = np.array([0.3, -0.2], dtype=float64)
    return {
        "A": np.array(
            [
                [0.6, 0.1, 0.0, 0.0],
                [0.2, 0.5, 0.1, 0.0],
                [0.0, 0.3, 0.4, 0.1],
                [0.1, 0.0, 0.2, 0.3],
            ],
            dtype=float64,
        ),
        "B": np.array(
            [[1.0, 0.2, 0.0], [0.0, 0.8, 0.0], [0.3, 0.0, 0.0], [0.0, 0.5, 0.0]],
            dtype=float64,
        ),
        "C": C,
        "d": d,
        "meas_addr": _ekf_system_meas.address,
        "jac_addr": _ekf_system_jac.address,
        "calib_params": np.concatenate([[curvature], ss, d, C.ravel()]),
        "Q": np.array(
            [[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.25]], dtype=float64
        ),
        "R": np.array([[0.02, 0.005], [0.005, 0.03]], dtype=float64),
        "steady_state": ss,
        "y": np.random.default_rng(0).normal(size=(8, 2)),
        "x0": np.array([0.2, -0.1, 0.05, 0.3], dtype=float64),
        "P0": np.diag(np.array([0.5, 0.4, 0.3, 0.2], dtype=float64)),
    }


def _kf_system(**kwargs):
    system = _system(**kwargs)
    for key in ("meas_addr", "jac_addr", "calib_params"):
        system.pop(key)
    return system


def _ekf_system(**kwargs):
    system = _system(**kwargs)
    for key in ("C", "d"):
        system.pop(key)
    return system


def _ill_conditioned_P0():
    """Eigenvalues 1 through 1e14 under a fixed reflection, so no axis is aligned."""
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=float64)
    U = np.eye(4, dtype=float64) - 2.0 * np.outer(v, v) / (v @ v)
    P0 = U @ np.diag(np.logspace(0.0, 14.0, 4)) @ U.T
    return 0.5 * (P0 + P0.T)


_MEAS_SIG = types.void(
    types.CPointer(types.float64),
    types.CPointer(types.float64),
    types.CPointer(types.float64),
)


@cfunc(_MEAS_SIG)
def _ukf_measurement(model_vars, params, out):
    out[0] = model_vars[0] + params[0] * model_vars[1]


# 1-D linear measurement h(x) = x and its (constant) jacobian, as @cfuncs for the
# EKF address-based API. The linear measurement lets the EKF be checked against
# the exact linear Kalman filter.
@cfunc(_MEAS_SIG)
def _ekf_lin_meas(x, params, out):
    out[0] = x[0]


@cfunc(_MEAS_SIG)
def _ekf_lin_jac(x, params, out):
    out[0] = 1.0


@cfunc(_MEAS_SIG)
def _ekf_bias_meas(x, params, out):
    out[0] = x[0] + params[0]


# `_system`'s measurement: curvature, expansion point (4), intercepts (2), and
# loadings (2 x 4, row-major) in calib_params.
@cfunc(_MEAS_SIG)
def _ekf_system_meas(x, params, out):
    for i in range(2):
        acc = params[5 + i]
        for j in range(4):
            acc += params[7 + 4 * i + j] * (x[j] - params[1 + j])
        out[i] = acc
    u = x[1] - params[2]
    out[0] += params[0] * u * u


@cfunc(_MEAS_SIG)
def _ekf_system_jac(x, params, out):
    for k in range(8):
        out[k] = params[7 + k]
    out[1] += 2.0 * params[0] * (x[1] - params[2])


_KERNELS = {
    "linear": (run_linear, smooth_linear, _kf_system),
    "extended": (run_extended, smooth_extended, _ekf_system),
}


def _ukf_system_1d():
    return {
        "meas_addr": _ukf_measurement.address,
        "hx": np.array([[0.65]], dtype=float64),
        "gx": np.array([[1.5]], dtype=float64),
        # bu spans every variable: a control responds to an innovation
        # contemporaneously, which a state-only loading cannot express.
        "bu": np.array([[1.0], [-0.4]], dtype=float64),
        "hxx": np.array([[[0.05]]], dtype=float64),
        "gxx": np.array([[[0.1]]], dtype=float64),
        "hxu": np.array([[[0.06]]], dtype=float64),
        "gxu": np.array([[[-0.03]]], dtype=float64),
        "huu": np.array([[[0.08]]], dtype=float64),
        "guu": np.array([[[0.02]]], dtype=float64),
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


def test_error_code_dispatch_maps_known_errors_and_rejects_unknown():
    assert get_error_constructor(ErrorCode.P_FACTORIZATION) is MatrixFactorizationError
    assert get_error_constructor(ErrorCode.Q_FACTORIZATION) is MatrixFactorizationError
    assert get_error_constructor(ErrorCode.S_FACTORIZATION) is MatrixFactorizationError
    assert get_error_constructor(ErrorCode.BAD_HYPERPARAMS) is HyperparameterError

    with pytest.raises(ValueError, match="Unknown error code"):
        get_error_constructor(ErrorCode.SUCCESS)


def test_shape_validate_raises_on_bad_A_shape():
    A = np.eye(2, dtype=float64)
    B = np.ones((2, 1), dtype=float64)
    Q = np.eye(1, dtype=float64)
    R = np.eye(1, dtype=float64)
    C = np.ones((1, 2), dtype=float64)
    d = np.zeros((1,), dtype=float64)

    with pytest.raises(ShapeMismatchError):
        _shape_validate(A[:1], B, Q, R, C, d, nmk=(2, 1, 1))


def test_run_linear_outputs_shapes_and_first_prediction():
    A, B, C, d, Q, R, ss = _linear_system_1d()
    y = np.zeros((5, 1), dtype=float64)
    x0 = np.array([2.0], dtype=float64)
    P0 = np.array([[1.0]], dtype=float64)

    out = run_linear(A, B, C, d, Q, R, ss, y, x0=x0, P0=P0)

    assert out.x_pred.shape == (5, 1)
    assert out.x_filt.shape == (5, 1)
    assert out.P_pred.shape == (5, 1, 1)
    assert out.P_filt.shape == (5, 1, 1)
    assert out.y_pred.shape == (5, 1)
    assert out.y_filt.shape == (5, 1)
    assert out.innov.shape == (5, 1)
    assert out.S.shape == (5, 1, 1)
    assert out.eps_hat.shape == (5, 1)
    assert np.isfinite(out.loglik)

    # x0 is the prior for the first observed state, so the first prediction is
    # x0 itself; A only enters from the second period on.
    assert np.allclose(out.x_pred[0], x0)
    assert np.allclose(out.P_pred[0], P0)
    assert np.allclose(out.x_pred[1], A @ out.x_filt[0])
    assert np.allclose(out.y_pred[0], C @ out.x_pred[0] + d)
    assert np.allclose(out.P_filt, np.transpose(out.P_filt, (0, 2, 1)))


def test_run_linear_can_skip_history_storage_for_loglik_only_path():
    A, B, C, d, Q, R, ss = _linear_system_1d()
    y = np.zeros((5, 1), dtype=float64)
    x0 = np.array([2.0], dtype=float64)
    P0 = np.array([[1.0]], dtype=float64)

    full = run_linear(A, B, C, d, Q, R, ss, y, x0=x0, P0=P0)
    minimal = run_linear(
        A,
        B,
        C,
        d,
        Q,
        R,
        ss,
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
    assert minimal.eps_hat.shape == (0, 1)


def test_run_linear_return_shocks():
    A, B, C, d, Q, R, ss = _linear_system_1d()
    y = np.zeros((4, 1), dtype=float64)

    out = run_linear(A, B, C, d, Q, R, ss, y)

    assert out.eps_hat is not None
    assert out.eps_hat.shape == (4, 1)


def test_run_linear_raises_on_singular_innovation_covariance():
    A = np.eye(1, dtype=float64)
    B = np.eye(1, dtype=float64)
    C = np.eye(1, dtype=float64)
    d = np.zeros((1,), dtype=float64)
    Q = np.zeros((1, 1), dtype=float64)
    R = np.zeros((1, 1), dtype=float64)
    ss = np.zeros(1, dtype=float64)
    y = np.zeros((2, 1), dtype=float64)

    with pytest.raises(MatrixFactorizationError):
        run_linear(A, B, C, d, Q, R, ss, y, P0=np.zeros((1, 1), dtype=float64))


@pytest.mark.parametrize("mode", ["linear", "extended"])
def test_symmetrize_makes_every_covariance_exactly_symmetric(mode):
    _, smooth, build = _KERNELS[mode]
    out = smooth(**build(), symmetrize=True)
    fwd = out.filter_result

    for P in (fwd.P_pred, fwd.P_filt, fwd.S, out.P_smooth):
        assert np.array_equal(P, np.transpose(P, (0, 2, 1)))


@pytest.mark.parametrize("joseph_cov", [False, True])
@pytest.mark.parametrize("mode", ["linear", "extended"])
def test_unsymmetrized_covariance_leaves_symmetry_on_an_ill_conditioned_prior(
    mode, joseph_cov
):
    # At a well-conditioned prior the drift is rounding and may be zero; at this
    # scale it is orders above the threshold.
    run, _, build = _KERNELS[mode]
    system = build() | {"P0": _ill_conditioned_P0()}
    out = run(**system, symmetrize=False, joseph_cov=joseph_cov)

    assert np.abs(out.P_filt - np.transpose(out.P_filt, (0, 2, 1))).max() > 1e-12


@pytest.mark.parametrize("symmetrize", [False, True])
@pytest.mark.parametrize("mode", ["linear", "extended"])
def test_joseph_update_departs_from_the_standard_one_on_an_ill_conditioned_prior(
    mode, symmetrize
):
    # The two forms are equal in exact arithmetic, so only a prior this badly
    # conditioned separates them by more than rounding.
    run, _, build = _KERNELS[mode]
    system = build() | {"P0": _ill_conditioned_P0()}
    standard = run(**system, symmetrize=symmetrize, joseph_cov=False)
    joseph = run(**system, symmetrize=symmetrize, joseph_cov=True)

    assert abs(joseph.loglik - standard.loglik) > 1e-7


# The curvature makes the point the extended filter's jacobian is evaluated at
# observable; the linear view has no callable to carry it.
@pytest.mark.parametrize(
    ("mode", "curvature"),
    [("linear", 0.0), ("extended", 0.3)],
    ids=["linear", "extended"],
)
def test_steady_state_shifts_the_state_series_and_nothing_else(mode, curvature):
    _, smooth, build = _KERNELS[mode]
    system = build(curvature=curvature)
    ss = system["steady_state"]
    shifted = smooth(**system)
    origin = smooth(**build(steady_state=(0.0, 0.0, 0.0, 0.0), curvature=curvature))
    fwd, fwd0 = shifted.filter_result, origin.filter_result

    assert np.abs(fwd.x_pred - fwd0.x_pred - ss).max() < 1e-12
    assert np.abs(fwd.x_filt - fwd0.x_filt - ss).max() < 1e-12
    assert np.abs(shifted.x_smooth - origin.x_smooth - ss).max() < 1e-12

    for name in ("P_pred", "P_filt", "y_pred", "y_filt", "innov", "std_innov", "S"):
        assert np.abs(getattr(fwd, name) - getattr(fwd0, name)).max() < 1e-12, name
    assert np.abs(fwd.eps_hat - fwd0.eps_hat).max() < 1e-12
    assert np.abs(shifted.P_smooth - origin.P_smooth).max() < 1e-12
    assert np.abs(shifted.eps_smooth - origin.eps_smooth).max() < 1e-12
    assert abs(fwd.loglik - fwd0.loglik) < 1e-12


@pytest.mark.parametrize("mode", ["linear", "extended"])
def test_states_observables_and_shocks_stay_on_their_own_axes(mode):
    _, smooth, build = _KERNELS[mode]
    system = build()
    y, R = system["y"], system["R"]
    d = _kf_system()["d"]
    out = smooth(**system)
    fwd = out.filter_result
    T = len(y)

    for P in (fwd.P_pred, fwd.P_filt, out.P_smooth):
        assert P.shape == (T, 4, 4)
    for x in (fwd.x_pred, fwd.x_filt, out.x_smooth):
        assert x.shape == (T, 4)
    for v in (fwd.y_pred, fwd.y_filt, fwd.innov, fwd.std_innov):
        assert v.shape == (T, 2)
    assert fwd.S.shape == (T, 2, 2)
    assert fwd.eps_hat.shape == (T, 3)
    assert out.eps_smooth.shape == (T, 3)

    # The observable that loads on no state is predicted by its intercept alone,
    # and its innovation covariance row is its measurement noise row.
    assert np.abs(fwd.y_pred[:, 1] - d[1]).max() < 1e-12
    assert np.abs(fwd.innov[:, 1] - (y[:, 1] - d[1])).max() < 1e-12
    assert np.abs(fwd.S[:, 1, :] - R[1]).max() < 1e-12

    # The shock that loads on no state is never inferred, while the others are.
    assert np.abs(fwd.eps_hat[:, 2]).max() < 1e-12
    assert np.abs(out.eps_smooth[:, 2]).max() < 1e-12
    assert np.abs(fwd.eps_hat[:, :2]).max() > 0.1


def test_run_unscented_outputs_shapes_and_projected_fields():
    pytest.importorskip("SymbolicDSGE._ckernels.kalman")
    kwargs = _ukf_system_1d()

    out = run_unscented(**kwargs)

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


def test_run_unscented_can_skip_history_storage_for_loglik_only_path():
    pytest.importorskip("SymbolicDSGE._ckernels.kalman")
    kwargs = _ukf_system_1d()

    full = run_unscented(**kwargs)
    minimal = run_unscented(**kwargs, _store_history=False)

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
        run_unscented(**(kwargs | {"meas_addr": 0}))


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("hx", np.zeros((1, 2), dtype=float64)),
        ("bu", np.zeros((3, 1), dtype=float64)),
        ("gx", np.zeros((1, 2), dtype=float64)),
        ("hxx", np.zeros((1, 1), dtype=float64)),
        ("gxx", np.zeros((1, 1), dtype=float64)),
        ("hxu", np.zeros((1, 1), dtype=float64)),
        ("gxu", np.zeros((1, 1), dtype=float64)),
        ("huu", np.zeros((1, 1), dtype=float64)),
        ("guu", np.zeros((1, 1), dtype=float64)),
        ("hss", np.zeros(2, dtype=float64)),
        ("gss", np.zeros(2, dtype=float64)),
        ("steady_state", np.zeros(3, dtype=float64)),
        ("Q", np.zeros((2, 2), dtype=float64)),
        ("y", np.zeros(4, dtype=float64)),
        ("R", np.zeros((2, 2), dtype=float64)),
        ("z0", np.zeros(3, dtype=float64)),
    ],
)
def test_run_unscented_rejects_every_misshaped_argument(name, bad):
    with pytest.raises(ShapeMismatchError, match=f"'{name}'"):
        run_unscented(**(_ukf_system_1d() | {name: bad}))


def test_run_unscented_reports_each_kernel_failure_by_its_own_code():
    kwargs = _ukf_system_1d()
    cases = [
        # The augmented dimension is 2 n_state + n_exog = 3, so this scale is zero.
        ({"kappa": -3.0}, ErrorCode.BAD_HYPERPARAMS, HyperparameterError),
        (
            {"Q": np.array([[-1.0]])},
            ErrorCode.Q_FACTORIZATION,
            MatrixFactorizationError,
        ),
        (
            {"P0": -np.eye(2, dtype=float64), "jitter": 0.0},
            ErrorCode.P_FACTORIZATION,
            MatrixFactorizationError,
        ),
        (
            {"R": np.array([[-10.0]])},
            ErrorCode.S_FACTORIZATION,
            MatrixFactorizationError,
        ),
    ]
    for override, code, error in cases:
        system = kwargs | override
        assert run_unscented(**system, _raise_on_error=False).status == int(code)
        with pytest.raises(error):
            run_unscented(**system)


def test_run_extended_matches_linear_when_measurement_is_linear():
    A, B, C, d, Q, R, ss = _linear_system_1d()
    y = np.zeros((5, 1), dtype=float64)
    x0 = np.array([1.5], dtype=float64)
    P0 = np.array([[0.7]], dtype=float64)
    calib = np.array([], dtype=float64)

    linear = run_linear(A, B, C, d, Q, R, ss, y, x0=x0, P0=P0)
    extended = run_extended(
        meas_addr=_ekf_lin_meas.address,
        jac_addr=_ekf_lin_jac.address,
        A=A,
        B=B,
        calib_params=calib,
        Q=Q,
        R=R,
        steady_state=ss,
        y=y,
        x0=x0,
        P0=P0,
    )

    assert np.allclose(extended.x_pred, linear.x_pred)
    assert np.allclose(extended.x_filt, linear.x_filt)
    assert np.allclose(extended.y_pred, linear.y_pred)
    assert np.allclose(extended.P_filt, linear.P_filt)
    assert np.allclose(extended.loglik, linear.loglik)


def test_extended_matches_linear_on_a_levels_model(solved_rbc_so):
    # The rbc observable is `c_obs: c(t)`, affine in the model variables, so the
    # extended filter relinearizes onto the same (C, d) the linear filter is
    # handed and the two must agree period by period. The data is arbitrary:
    # both filters read the same y, and S never depends on it, so agreement is
    # required whatever it holds.
    #
    # `test_run_extended_matches_linear_when_measurement_is_linear` above makes
    # the same claim against the kernels directly, where the caller supplies
    # (A, B, C, d) and no model, so there is no steady state to expand around.
    # This is the model-level case, and it only bites when that steady state is
    # away from the origin: the measurement cfunc is a function of levels, so a
    # filter that evaluates it at deviations drops `h(ss)` from every prediction.
    compiled = solved_rbc_so.compiled
    obs = list(compiled.observable_names)
    ss = np.asarray(solved_rbc_so.policy.steady_state, dtype=np.float64)

    # Guard the premise: at ss == 0 both conventions coincide and the test below
    # cannot fail for the reason it exists.
    assert np.max(np.abs(ss)) > 1.0

    rng = np.random.default_rng(0)
    y = pd.DataFrame(rng.normal(size=(24, len(obs))), columns=obs)
    R = np.eye(len(obs), dtype=np.float64) * 1e-4

    linear = LinearKalman().filter(solved_rbc_so, y, R=R)
    extended = ExtendedKalman().filter(solved_rbc_so, y, R=R)

    np.testing.assert_allclose(extended.y_pred, linear.y_pred, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(extended.innov, linear.innov, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(extended.x_filt, linear.x_filt, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(extended.loglik, linear.loglik, rtol=1e-10, atol=1e-12)


def test_run_extended_can_skip_history_storage_for_loglik_only_path():
    A, B, _, _, Q, R, ss = _linear_system_1d()
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
        steady_state=ss,
        y=y,
    )
    full = run_extended(**kwargs)
    minimal = run_extended(**kwargs, _store_history=False)

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
    assert minimal.eps_hat.shape == (0, 1)


def _linear_ukf_pair():
    """A model with no curvature, as the same system in both filters' terms.

    ``_ukf_measurement`` is ``vars[0] + params[0] * vars[1]``, which is affine,
    so the linear filter reads it as ``d + C x`` with ``d`` carrying the steady
    state the UKF applies inside its projection.

    The UKF's transition is [[hx, 0], [gx, 0]] once every quadratic block is
    zero, and its shock loading is ``bu`` whole: exactly the ``A``/``B`` the
    linear filter takes. The control column of ``A`` is zero, so neither the
    control entry of ``x0`` nor the control block of the linear ``P0`` reaches
    the recursion.

    The two priors are dated one period apart, and deliberately so. A UKF's
    ``z0`` is the state *before* the first observation, because an observable
    does not exist until the law of motion has run: it depends on this period's
    innovation. A linear filter's ``x0`` is the prior for the first observed
    state, because its state vector already contains the observables. Dynare
    carries the same split between ``gaussian_filter.m`` and
    ``kalman_filter.m`` for the same reason. So the linear side is handed
    ``z0`` advanced by one prediction, which is what makes the two describe the
    same data.
    """
    hx = np.array([[0.65]], dtype=float64)
    gx = np.array([[1.5]], dtype=float64)
    bu = np.array([[1.0], [-0.4]], dtype=float64)
    steady_state = np.array([2.0, 3.0], dtype=float64)
    params = np.array([0.25], dtype=float64)
    Q = np.array([[0.03]], dtype=float64)
    R = np.array([[0.2]], dtype=float64)
    y = np.array([[2.8], [2.7], [2.65], [2.6]], dtype=float64)
    p0_state = 0.4

    unscented = {
        "meas_addr": _ukf_measurement.address,
        "hx": hx,
        "gx": gx,
        "bu": bu,
        "hxx": np.zeros((1, 1, 1), dtype=float64),
        "gxx": np.zeros((1, 1, 1), dtype=float64),
        "hxu": np.zeros((1, 1, 1), dtype=float64),
        "gxu": np.zeros((1, 1, 1), dtype=float64),
        "huu": np.zeros((1, 1, 1), dtype=float64),
        "guu": np.zeros((1, 1, 1), dtype=float64),
        "hss": np.zeros(1, dtype=float64),
        "gss": np.zeros(1, dtype=float64),
        "steady_state": steady_state,
        "calib_params": params,
        "Q": Q,
        "R": R,
        "y": y,
        # x2 is identically zero on a linear model, so its prior variance is
        # too. The jittered Cholesky is what makes that factorable, and it is
        # the only reason the two runs are not bit-identical.
        "z0": np.array([0.1, 0.0], dtype=float64),
        "P0": np.diag(np.array([p0_state, 0.0], dtype=float64)),
        "alpha": 1.0,
        "beta": 2.0,
        "kappa": 1.0,
        "jitter": 1e-12,
        "symmetrize": True,
    }
    A = np.array([[0.65, 0.0], [1.5, 0.0]], dtype=float64)
    z0 = unscented["z0"]
    P0_ukf = unscented["P0"]
    linear = {
        "A": A,
        "B": bu,
        "C": np.array([[1.0, params[0]]], dtype=float64),
        "d": np.array([steady_state[0] + params[0] * steady_state[1]], dtype=float64),
        "Q": Q,
        "R": R,
        "steady_state": steady_state,
        "y": y,
        "x0": A @ z0,
        "P0": A @ P0_ukf @ A.T + bu @ Q @ bu.T,
        "symmetrize": True,
        "jitter": 1e-12,
    }
    return unscented, linear, steady_state


def test_unscented_reduces_to_the_linear_filter_without_curvature():
    """With every second-order block zero the UKF is the linear filter.

    The unscented transform is exact for an affine map, so this is an equality
    and not an approximation. It is the only check that reaches the filter's
    arithmetic rather than its shapes, and it pins the pieces a pruned
    recursion is easiest to get wrong: that a control responds to this period's
    innovation at all, and that the innovation enters as a sigma-point
    coordinate rather than as an additive covariance.
    """
    pytest.importorskip("SymbolicDSGE._ckernels.kalman")
    unscented, linear, _ = _linear_ukf_pair()

    ukf = run_unscented(**unscented)
    kf = run_linear(**linear)

    assert ukf.loglik == pytest.approx(kf.loglik, rel=1e-9, abs=1e-9)
    np.testing.assert_allclose(ukf.y_pred, kf.y_pred, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(ukf.innov, kf.innov, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(ukf.S, kf.S, rtol=1e-8, atol=1e-9)

    # Both filters report levels, so the two series are directly comparable.
    # The control row is the payload: it is where a contemporaneous shock
    # response would go missing.
    np.testing.assert_allclose(ukf.x_filt, kf.x_filt, rtol=1e-7, atol=1e-9)


def test_unscented_without_curvature_still_moves_with_the_shock_loading():
    """Guards the test above against passing on a degenerate system.

    If the control row of ``bu`` were dropped, the equivalence would still hold
    for any model whose controls do not respond to innovations. Perturbing that
    row has to move both filters, and by the same amount.
    """
    pytest.importorskip("SymbolicDSGE._ckernels.kalman")
    unscented, linear, _ = _linear_ukf_pair()
    base = run_unscented(**unscented).loglik

    bumped = np.array([[1.0], [-1.9]], dtype=float64)
    unscented["bu"] = bumped
    linear["B"] = bumped
    # The linear prior is derived from the UKF's, so it moves with the loading.
    linear["P0"] = (
        linear["A"] @ unscented["P0"] @ linear["A"].T
        + bumped @ unscented["Q"] @ bumped.T
    )
    moved_ukf = run_unscented(**unscented).loglik
    moved_kf = run_linear(**linear).loglik

    assert not np.isclose(moved_ukf, base, rtol=1e-6)
    assert moved_ukf == pytest.approx(moved_kf, rel=1e-9, abs=1e-9)


def test_unscented_keeps_states_controls_observables_and_shocks_on_their_own_axes(
    solved_multishock, multishock_c_k_panel
):
    obs, y = multishock_c_k_panel
    out = UnscentedKalman().smooth(solved_multishock, y, obs)
    fwd = out.filter_result
    compiled = solved_multishock.compiled
    T, ns, nv = len(y), compiled.n_state, compiled.n_var
    nz, m, k = 2 * ns, len(obs), len(compiled.shock_names)

    for x in (fwd.x_pred, fwd.x_filt, out.x_smooth):
        assert x.shape == (T, nv)
    for x in (fwd.x1_pred, fwd.x2_pred, fwd.x1_filt, fwd.x2_filt):
        assert x.shape == (T, ns)
    for x in (out.x1_smooth, out.x2_smooth):
        assert x.shape == (T, ns)
    for P in (fwd.P_pred, fwd.P_filt, out.P_smooth):
        assert P.shape == (T, nz, nz)
    for v in (fwd.y_pred, fwd.y_filt, fwd.innov, fwd.std_innov):
        assert v.shape == (T, m)
    assert fwd.S.shape == (T, m, m)
    assert fwd.eps_hat.shape == (T, k)
    assert out.eps_smooth.shape == (T, k)


def test_unscented_smoother_runs_the_filter_forward_and_ends_on_its_last_period(
    solved_multishock, multishock_c_k_panel
):
    obs, y = multishock_c_k_panel
    out = UnscentedKalman().smooth(solved_multishock, y, obs)
    fwd = out.filter_result
    filtered = UnscentedKalman().filter(solved_multishock, y, obs)

    for name in (
        "x_pred",
        "x_filt",
        "x1_pred",
        "x2_pred",
        "x1_filt",
        "x2_filt",
        "P_pred",
        "P_filt",
        "y_pred",
        "y_filt",
        "innov",
        "std_innov",
        "S",
        "eps_hat",
    ):
        assert np.array_equal(getattr(fwd, name), getattr(filtered, name)), name
    assert fwd.loglik == filtered.loglik

    # Nothing comes after the last observation, so smoothing leaves it filtered.
    assert np.array_equal(out.x_smooth[-1], fwd.x_filt[-1])
    assert np.array_equal(out.x1_smooth[-1], fwd.x1_filt[-1])
    assert np.array_equal(out.x2_smooth[-1], fwd.x2_filt[-1])
    assert np.array_equal(out.P_smooth[-1], fwd.P_filt[-1])
    assert np.array_equal(out.eps_smooth[-1], fwd.eps_hat[-1])


def test_unscented_smoother_reduces_to_the_linear_smoother_without_curvature():
    unscented, linear, _ = _linear_ukf_pair()

    ukf = smooth_unscented(**unscented)
    kf = smooth_linear(**linear)

    np.testing.assert_allclose(ukf.x_smooth, kf.x_smooth, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(ukf.eps_smooth, kf.eps_smooth, rtol=1e-7, atol=1e-9)


def test_unscented_weights_are_inert_without_curvature():
    # The unscented transform is exact for an affine map whatever its weights.
    unscented, _, _ = _linear_ukf_pair()
    base = run_unscented(**unscented).loglik

    for alpha, beta, kappa in ((0.5, 2.0, 0.0), (1.0, 0.0, 3.0)):
        moved = run_unscented(
            **(unscented | {"alpha": alpha, "beta": beta, "kappa": kappa})
        ).loglik
        assert abs(moved - base) < 1e-12


def test_unscented_weights_reach_the_kernel_with_curvature(
    solved_multishock, multishock_c_k_panel
):
    obs, y = multishock_c_k_panel
    base = UnscentedKalman().filter(solved_multishock, y, obs).loglik

    for alpha, beta, kappa in ((0.5, 2.0, 0.0), (1.0, 0.0, 3.0)):
        moved = UnscentedKalman(alpha=alpha, beta=beta, kappa=kappa).filter(
            solved_multishock, y, obs
        )
        assert abs(moved.loglik - base) > 1e-4


def test_unscented_symmetrize_makes_every_covariance_exactly_symmetric(
    solved_multishock, multishock_c_k_panel
):
    obs, y = multishock_c_k_panel
    out = UnscentedKalman(symmetrize=True).smooth(solved_multishock, y, obs)
    fwd = out.filter_result

    for P in (fwd.P_pred, fwd.P_filt, fwd.S, out.P_smooth):
        assert np.array_equal(P, np.transpose(P, (0, 2, 1)))
