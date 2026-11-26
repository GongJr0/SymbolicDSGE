"""Native Kalman kernels, called directly."""

import numpy as np
from numba import cfunc, types

from SymbolicDSGE.filters.errors import ErrorCode
from SymbolicDSGE._ckernels.kalman import (
    kf_forward,
    ukf_forward,
)

_RTOL = 1e-9
_ATOL = 1e-9


def _c(z: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(z, dtype=np.float64)


_MEAS_SIG = types.void(
    types.CPointer(types.float64),
    types.CPointer(types.float64),
    types.CPointer(types.float64),
)


@cfunc(_MEAS_SIG)
def _ukf_measure_first_var(model_vars, _params, out):
    out[0] = model_vars[0]


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
        _c([0.0]),
        _c(rng.standard_normal((T, m))),
        _c([0.0]),
        _c([[1.0]]),
        True,
        True,
        0.0,
        True,
    )
    status, _ = kf_forward(*args)
    assert status == ErrorCode.S_FACTORIZATION


def test_ukf_returns_model_variable_history() -> None:
    T = 4
    hx = _c([[0.55]])
    gx = _c([[2.0]])
    # bu spans every variable: the control takes its own shock loading.
    bu = _c([[1.0], [-0.3]])
    hxx = _c([[[0.1]]])
    gxx = _c([[[0.4]]])
    hxu = _c([[[0.05]]])
    gxu = _c([[[-0.02]]])
    huu = _c([[[0.07]]])
    guu = _c([[[0.03]]])
    hss = _c([0.03])
    gss = _c([0.2])
    steady_state = _c([10.0, 20.0])
    params = _c([])
    Q = _c([[0.05]])
    R = _c([[0.25]])
    obs = _c(np.zeros((T, 1)))
    z0 = _c([0.1, 0.0])
    P0 = _c(np.diag([0.2, 0.3]))

    _, out = ukf_forward(
        _ukf_measure_first_var.address,
        hx,
        gx,
        bu,
        hxx,
        gxx,
        hxu,
        gxu,
        huu,
        guu,
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
        _eps_hat,
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
    # The control column is no longer a closed form in the pruned state. It is
    # the filter's own output, carrying this period's innovation through the
    # sigma set, so a control that responds to a shock cannot be reproduced by
    # projecting x1/x2. Restating a formula here is what made this assertion
    # track the implementation rather than the contract; its content is pinned
    # against the linear Kalman filter in tests/kalman/test_filter.py.
    #
    # What still holds exactly is that projecting is *not* what happens: with a
    # nonzero control row in bu, the two differ.
    naive = (
        steady_state[1]
        + 0.5 * gss[0]
        + gx[0, 0] * (x1_filt[:, 0] + x2_filt[:, 0])
        + 0.5 * gxx[0, 0, 0] * (P_filt[:, 0, 0] + x1_filt[:, 0] ** 2)
    )
    assert bu[1, 0] != 0.0
    assert not np.allclose(x_filt[:, 1], naive, rtol=1e-6, atol=1e-9)
    assert np.all(np.isfinite(x_pred[:, 1]))
    assert np.all(np.isfinite(x_filt[:, 1]))
