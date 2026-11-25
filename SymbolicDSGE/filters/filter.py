"""Native passthroughs for the Kalman filter recursions, including linear, extended, and unscented implementations."""

from .._ckernels.kalman import (
    stationary_covariance,
    kf_forward,
    kf_smooth,
    ekf_forward,
    ekf_smooth,
    ukf_forward,
    ukf_smooth,
)
from .errors import (
    ErrorCode,
    ShapeMismatchError,
    get_error_constructor,
)
from dataclasses import dataclass
import numpy as np
from numpy import float64
from numpy.typing import NDArray

from typing import Tuple, TypeVar, Generic

NDF = NDArray[float64]


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Filter outputs and the state/observable descriptors that go with them.

    Returned by :meth:`KalmanFilter.run` and by ``SolvedModel.kalman`` for the linear
    and extended filter modes. :class:`UnscentedFilterResult` extends it with the
    first- and second-order components of the unscented state estimate.

    State paths are reported in levels: the solved steady state is added to
    ``x_pred`` and ``x_filt`` before they are returned. Measurements are in levels
    throughout the recursion, as comparison against observed data requires.

    Linear and extended filtering read ``x0`` and ``P0`` as the prior mean and
    covariance for the first observed state. Unscented filtering reads them as the
    state and covariance before the first observation.

    Attributes
    ----------
    x_pred : NDF
        Predicted states over time, before each date's observation is incorporated.
    x_filt : NDF
        Filtered states over time, after each date's observation is incorporated.
    P_pred : NDF
        Predicted state covariance over time, matching ``x_pred``.
    P_filt : NDF
        Filtered state covariance over time, matching ``x_filt``.
    y_pred : NDF
        Predicted observables over time.
    y_filt : NDF
        Filtered observables over time.
    innov : NDF
        Observable innovations, the pre-update difference between the observation
        and its prediction.
    std_innov : NDF
        Innovations standardized by their covariance ``S``.
    S : NDF
        Innovation covariance over time.
    loglik : float64
        Log likelihood of the measurements.
    eps_hat : NDF
        Conditional estimates of the structural shocks given the observed data.
    status : int
        Error code from the recursion. Zero on success; a nonzero value is set
        instead of raising when the run suppressed errors.
    x1_pred : NDF
        First-order component of the predicted state. Unscented filter only.
    x2_pred : NDF
        Second-order component of the predicted state. Unscented filter only.
    x1_filt : NDF
        First-order component of the filtered state. Unscented filter only.
    x2_filt : NDF
        Second-order component of the filtered state. Unscented filter only.
    """

    x_pred: NDF
    x_filt: NDF

    P_pred: NDF
    P_filt: NDF

    y_pred: NDF  # y_{t|t-1} = C x_pred + d
    y_filt: NDF  # y_{t|t}   = C x_filt + d

    innov: NDF  # pre-update
    std_innov: NDF  # scaled by S
    S: NDF

    loglik: float64

    eps_hat: NDF
    status: int = 0


@dataclass(frozen=True, kw_only=True, slots=True)
class UnscentedFilterResult(FilterResult):
    x1_pred: NDF
    x2_pred: NDF
    x1_filt: NDF
    x2_filt: NDF


FilterT = TypeVar("FilterT", bound=FilterResult)


@dataclass(frozen=True, kw_only=True, slots=True)
class SmootherResult(Generic[FilterT]):
    filter_result: FilterT

    x_smooth: NDF
    P_smooth: NDF
    eps_smooth: NDF


@dataclass(frozen=True, kw_only=True, slots=True)
class UnscentedSmootherResult(SmootherResult[UnscentedFilterResult]):
    x1_smooth: NDF
    x2_smooth: NDF


def _shape_validate(
    A: NDF,
    B: NDF,
    Q: NDF,
    R: NDF,
    C: NDF | None,
    d: NDF | None,
    nmk: Tuple[int, int, int],
) -> None:
    n, m, k = nmk
    if A.shape != (n, n):
        raise ShapeMismatchError("A", f"({n}, {n})", str(A.shape))
    if B.shape != (n, k):
        raise ShapeMismatchError("B", f"({n}, {k})", str(B.shape))
    if Q.shape != (k, k):
        raise ShapeMismatchError("Q", f"({k}, {k})", str(Q.shape))
    if R.shape != (m, m):
        raise ShapeMismatchError("R", f"({m}, {m})", str(R.shape))

    if C is not None:
        if C.shape != (m, n):
            raise ShapeMismatchError("C", f"({m}, {n})", str(C.shape))
    if d is not None:
        if d.shape != (m,):
            raise ShapeMismatchError("d", f"({m},)", str(d.shape))


def _ukf_shape_validate(
    hx: NDF,
    gx: NDF,
    bu: NDF,
    hxx: NDF,
    gxx: NDF,
    hxu: NDF,
    gxu: NDF,
    huu: NDF,
    guu: NDF,
    hss: NDF,
    gss: NDF,
    steady_state: NDF,
    Q: NDF,
    R: NDF,
    y: NDF,
    z0: NDF,
) -> None:
    if hx.ndim != 2 or hx.shape[0] != hx.shape[1]:
        raise ShapeMismatchError("hx", "(n_state, n_state)", str(hx.shape))

    n_state = hx.shape[0]
    n_var = n_state + gx.shape[0]
    n_z = 2 * n_state

    if bu.ndim != 2 or bu.shape[0] != n_var:
        raise ShapeMismatchError("bu", f"({n_var}, n_exog)", str(bu.shape))
    n_exog = bu.shape[1]

    if gx.ndim != 2 or gx.shape[1] != n_state:
        raise ShapeMismatchError("gx", f"(n_ctrl, {n_state})", str(gx.shape))
    n_ctrl = gx.shape[0]
    n_var = n_state + n_ctrl

    if hxx.shape != (n_state, n_state, n_state):
        raise ShapeMismatchError(
            "hxx", f"({n_state}, {n_state}, {n_state})", str(hxx.shape)
        )
    if gxx.shape != (n_ctrl, n_state, n_state):
        raise ShapeMismatchError(
            "gxx", f"({n_ctrl}, {n_state}, {n_state})", str(gxx.shape)
        )
    if hxu.shape != (n_state, n_state, n_exog):
        raise ShapeMismatchError(
            "hxu", f"({n_state}, {n_state}, {n_exog})", str(hxu.shape)
        )
    if gxu.shape != (n_ctrl, n_state, n_exog):
        raise ShapeMismatchError(
            "gxu", f"({n_ctrl}, {n_state}, {n_exog})", str(gxu.shape)
        )
    if huu.shape != (n_state, n_exog, n_exog):
        raise ShapeMismatchError(
            "huu", f"({n_state}, {n_exog}, {n_exog})", str(huu.shape)
        )
    if guu.shape != (n_ctrl, n_exog, n_exog):
        raise ShapeMismatchError(
            "guu", f"({n_ctrl}, {n_exog}, {n_exog})", str(guu.shape)
        )
    if hss.shape != (n_state,):
        raise ShapeMismatchError("hss", f"({n_state},)", str(hss.shape))
    if gss.shape != (n_ctrl,):
        raise ShapeMismatchError("gss", f"({n_ctrl},)", str(gss.shape))
    if steady_state.shape != (n_var,):
        raise ShapeMismatchError("steady_state", f"({n_var},)", str(steady_state.shape))
    if Q.shape != (n_exog, n_exog):
        raise ShapeMismatchError("Q", f"({n_exog}, {n_exog})", str(Q.shape))
    if y.ndim != 2:
        raise ShapeMismatchError("y", "(T, n_obs)", str(y.shape))
    n_obs = y.shape[1]
    if R.shape != (n_obs, n_obs):
        raise ShapeMismatchError("R", f"({n_obs}, {n_obs})", str(R.shape))
    if z0.shape != (n_z,):
        raise ShapeMismatchError("z0", f"({n_z},)", str(z0.shape))


def _sym(P: NDF) -> NDF:
    return 0.5 * (P + P.T)


def _initialize_P0(
    P0: NDF | None, A: NDF, B: NDF, Q: NDF, n: int, symmetrize: bool
) -> NDF:
    """Initialize the state covariance matrix P0.

    If P0 is provided, it is used as is. If not, the stationary covariance
    is computed from A, B, and Q. If that fails, a large diagonal matrix is
    returned.
    """
    if P0 is not None:
        P0 = P0.reshape(n, n)
        if symmetrize:
            return _sym(P0)
        return P0
    else:
        err, P0 = stationary_covariance(A, B, Q)
        if err != 0:
            P0 = np.eye(n, dtype=float64)
        return P0


def _initialize_P0_unscented(
    P0: NDF | None,
    A: NDF,
    B: NDF,
    Q: NDF,
    n_state: int,
    symmetrize: bool,
) -> NDF:
    """Initialize the state covariance matrix P0 for the unscented Kalman filter.

    If P0 is provided, it is used as is. If not, a large diagonal matrix is
    returned.
    """
    n_z = 2 * n_state
    out = np.zeros((n_z, n_z), dtype=float64)
    if P0 is not None:
        if P0.shape == (n_z, n_z):
            out[:] = P0
        else:
            out[:n_state, :n_state] = P0.reshape(n_state, n_state)
        if symmetrize:
            out = _sym(out)
    else:
        err, P = stationary_covariance(A, B, Q)
        if err != 0:
            P = np.eye(n_state, dtype=float64)
        out[:n_state, :n_state] = P
    return out


def _prep_gaussian(
    A: NDF,
    B: NDF,
    Q: NDF,
    R: NDF,
    C: NDF | None,
    d: NDF | None,
    y: NDF,
    x0: NDF | None,
    P0: NDF | None,
    symmetrize: bool,
) -> tuple[int, tuple[int, int, int], NDF, NDF]:
    """Validate a linear or extended run and resolve its prior.

    Returns ``(T, (n, m, k), x0, P0)``. The extended filter passes ``C`` and
    ``d`` as None.
    """
    T, m = y.shape
    n = A.shape[0]
    k = B.shape[1]
    _shape_validate(A, B, Q, R, C, d, nmk=(n, m, k))
    x0 = x0.reshape(n) if x0 is not None else np.zeros((n,), dtype=float64)
    P0 = _initialize_P0(P0, A, B, Q, n, symmetrize)
    return T, (n, m, k), x0, P0


def _prep_unscented(
    meas_addr: int,
    hx: NDF,
    gx: NDF,
    bu: NDF,
    hxx: NDF,
    gxx: NDF,
    hxu: NDF,
    gxu: NDF,
    huu: NDF,
    guu: NDF,
    hss: NDF,
    gss: NDF,
    steady_state: NDF,
    Q: NDF,
    R: NDF,
    y: NDF,
    z0: NDF,
    P0: NDF | None,
    symmetrize: bool,
) -> tuple[NDF, NDF, NDF]:
    """Validate an unscented run and resolve its pruned prior.

    Returns ``(P0, Q, R)`` with ``P0`` in the pruned dimension and ``Q``/``R``
    symmetrized when requested.
    """
    if meas_addr == 0:
        raise ValueError("meas_addr must be a nonzero measurement cfunc address.")
    _ukf_shape_validate(
        hx, gx, bu, hxx, gxx, hxu, gxu, huu, guu, hss, gss, steady_state, Q, R, y, z0
    )

    n_state = hx.shape[0]
    n_z = 2 * n_state

    P0 = _initialize_P0_unscented(P0, hx, bu[:n_state, :], Q, n_state, symmetrize)
    if P0.shape != (n_z, n_z):
        raise ShapeMismatchError("P0", f"({n_z}, {n_z})", str(P0.shape))

    if symmetrize:
        Q = _sym(Q)  # pyright: ignore
        R = _sym(R)  # pyright: ignore
    return P0, Q, R


def _raise_status(err: int, raise_on_error: bool) -> None:
    if err != 0 and raise_on_error:
        raise get_error_constructor(ErrorCode(err))(err)


def _filter_result(
    out: tuple[NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, float64],
    status: int = 0,
) -> FilterResult:
    # fmt: off
    (
        x_pred, x_filt,
        P_pred, P_filt,
        y_pred, y_filt,
        v, u,
        S,
        eps_hat,
        loglik,
    ) = out
    return FilterResult(
        x_pred=x_pred, x_filt=x_filt,
        P_pred=P_pred, P_filt=P_filt,
        y_pred=y_pred, y_filt=y_filt,
        innov=v, std_innov=u,
        S=S,
        eps_hat=eps_hat,
        loglik=loglik,
        status=status,
    )
    # fmt: on


def _smoother_result(
    fwd: tuple[NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, float64],
    smooth: tuple[NDF, NDF, NDF],
    status: int = 0,
) -> SmootherResult[FilterResult]:
    fwd_res = _filter_result(fwd, status)
    x_smooth, P_smooth, eps_smooth = smooth
    return SmootherResult(
        filter_result=fwd_res,
        x_smooth=x_smooth,
        P_smooth=P_smooth,
        eps_smooth=eps_smooth,
    )


def _unscented_result(
    out: tuple[
        NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, float64
    ],
    status: int = 0,
) -> UnscentedFilterResult:
    # fmt: off
    (
        x1_pred, x2_pred,
        x1_filt, x2_filt,
        x_pred, x_filt,
        P_pred, P_filt,
        y_pred, y_filt,
        v, u,
        S,
        eps_hat,
        loglik,
    ) = out
    return UnscentedFilterResult(
        x_pred=x_pred, x_filt=x_filt,
        P_pred=P_pred, P_filt=P_filt,
        y_pred=y_pred, y_filt=y_filt,
        innov=v, std_innov=u,
        S=S,
        eps_hat=eps_hat,
        x1_pred=x1_pred, x2_pred=x2_pred,
        x1_filt=x1_filt, x2_filt=x2_filt,
        loglik=loglik,
        status=status,
    )
    # fmt: on


def _unscented_smoother_result(
    fwd: tuple[
        NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, float64
    ],
    smooth: tuple[NDF, NDF, NDF, NDF, NDF],
    status: int = 0,
) -> UnscentedSmootherResult:
    fwd_res = _unscented_result(fwd, status)
    x_smooth, P_smooth, eps_smooth, x1_smooth, x2_smooth = smooth
    return UnscentedSmootherResult(
        filter_result=fwd_res,
        x_smooth=x_smooth,
        P_smooth=P_smooth,
        eps_smooth=eps_smooth,
        x1_smooth=x1_smooth,
        x2_smooth=x2_smooth,
    )


def run_linear(
    A: NDF,
    B: NDF,
    C: NDF,
    d: NDF,
    Q: NDF,
    R: NDF,
    steady_state: NDF,
    y: NDF,
    x0: NDF | None = None,
    P0: NDF | None = None,
    symmetrize: bool = True,
    joseph_cov: bool = False,
    jitter: float = 0.0,
    _store_history: bool = True,
    _raise_on_error: bool = True,
) -> FilterResult:
    """Linear Gaussian Kalman filter run.

    Parameters
    ----------
    A : NDF
        State transition matrix. (SolvedModel.policy.A)
    B : NDF
        Shock loading matrix. (SolvedModel.policy.B)
    C : NDF
        Linear measurement coefficient matrix.
    d : NDF
        Linear measurement constant vector.
    Q : NDF
        Shock covariance matrix.
    R : NDF
        Measurement-noise covariance matrix.
    steady_state : NDF
        Steady-state vector of the state variables.
    y : NDF
        Observations in declared order. Shape (T, m) where T is the number of periods and m is the number of observed variables.
    x0 : NDF | None
        State vector at t-1 to initialize the filter.
    P0 : NDF | None
        State covariance matrix at t-1 to initialize the filter.
    symmetrize : bool
        Whether to symmetrize the covariance matrices at each step via (M + M.T)/2.
    joseph_cov : bool
        Whether to use the Joseph form of the covariance update.
    jitter : float
        Jitter to add when a cholesky decomposition fails. Set to 0.0 to disable.
    _store_history : bool
        Whether to store the full history of the filter. If False, the filter operates in likelihood-only mode.
    _raise_on_error : bool
        Whether to raise an exception if the filter encounters an error. If False, the filter will return a FilterResult with a nonzero :attr:`status` code.

    Returns
    -------
    FilterResult
        The result container of linear and extended Kalman filters.
        Contains predicted and filtered states, state covariances, measurements;
        innovations and standardized innovations; innovation covariances; log-likelihood; and shock estimates if requested.

    """
    T, nmk, x_prev, P_prev = _prep_gaussian(A, B, Q, R, C, d, y, x0, P0, symmetrize)

    # fmt: off
    err, out = kf_forward(
        T, nmk,
        A, B, C, d,
        Q, R,
        steady_state,
        y,
        x_prev, P_prev,
        symmetrize,
        joseph_cov,
        jitter,
        _store_history,
    )
    # fmt: on
    _raise_status(err, _raise_on_error)
    return _filter_result(out, err)


def smooth_linear(
    A: NDF,
    B: NDF,
    C: NDF,
    d: NDF,
    Q: NDF,
    R: NDF,
    steady_state: NDF,
    y: NDF,
    x0: NDF | None = None,
    P0: NDF | None = None,
    symmetrize: bool = True,
    joseph_cov: bool = False,
    jitter: float = 0.0,
    _raise_on_error: bool = True,
) -> SmootherResult[FilterResult]:
    """Linear Gaussian Kalman smoother.

    Parameters
    ----------
    A : NDF
        State transition matrix. (SolvedModel.policy.A)
    B : NDF
        Shock loading matrix. (SolvedModel.policy.B)
    C : NDF
        Linear measurement coefficient matrix.
    d : NDF
        Linear measurement constant vector.
    Q : NDF
        Shock covariance matrix.
    R : NDF
        Measurement-noise covariance matrix.
    steady_state : NDF
        Steady-state vector of the state variables.
    y : NDF
        Observations in declared order. Shape (T, m) where T is the number of periods and m is the number of observed variables.
    x0 : NDF | None
        State vector at t-1 to initialize the filter.
    P0 : NDF | None
        State covariance matrix at t-1 to initialize the filter.
    symmetrize : bool
        Whether to symmetrize the covariance matrices at each step via (M + M.T)/2.
    joseph_cov : bool
        Whether to use the Joseph form of the covariance update.
    jitter : float
        Jitter to add when a cholesky decomposition fails. Set to 0.0 to disable.
    _raise_on_error : bool
        Whether to raise an exception if the filter encounters an error. If False, the filter will return a FilterResult with a nonzero :attr:`status` code.

    Returns
    -------
    SmootherResult[FilterResult]
        The result container of linear and extended Kalman smoothers.
        Contains the forward pass filter result, smoothed states, state covariances, and shock estimates.
    """
    T, nmk, x_prev, P_prev = _prep_gaussian(A, B, Q, R, C, d, y, x0, P0, symmetrize)

    # fmt: off
    err, fwd, smooth = kf_smooth(
        T, nmk,
        A, B, C, d,
        Q, R,
        steady_state,
        y,
        x_prev, P_prev,
        symmetrize,
        joseph_cov,
        jitter,
    )
    # fmt: on
    _raise_status(err, _raise_on_error)
    return _smoother_result(fwd, smooth, err)


def run_unscented(
    meas_addr: int,
    hx: NDF,
    gx: NDF,
    bu: NDF,
    hxx: NDF,
    gxx: NDF,
    hxu: NDF,
    gxu: NDF,
    huu: NDF,
    guu: NDF,
    hss: NDF,
    gss: NDF,
    steady_state: NDF,
    calib_params: NDF,
    Q: NDF,
    R: NDF,
    y: NDF,
    z0: NDF,
    P0: NDF | None,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 1.0,
    symmetrize: bool = True,
    jitter: float = 0.0,
    _store_history: bool = True,
    _raise_on_error: bool = True,
) -> UnscentedFilterResult:
    """Unscented Kalman filter run against a second-order model solution.

    The policy tensors and ``steady_state`` come from an ``order=2`` solve. The
    measurement is propagated through the sigma points via ``meas_addr``, so no
    observation Jacobian is required.

    Parameters
    ----------
    meas_addr : int
        Address of the compiled measurement cfunc, applied to each sigma point.
    hx : NDF
        First-order state transition tensor.
    gx : NDF
        First-order control policy tensor.
    bu : NDF
        Shock loading tensor.
    hxx : NDF
        Second-order state transition tensor in the state.
    gxx : NDF
        Second-order control policy tensor in the state.
    hxu : NDF
        Second-order state transition cross term in state and shock.
    gxu : NDF
        Second-order control policy cross term in state and shock.
    huu : NDF
        Second-order state transition tensor in the shock.
    guu : NDF
        Second-order control policy tensor in the shock.
    hss : NDF
        Risk correction term of the state transition.
    gss : NDF
        Risk correction term of the control policy.
    steady_state : NDF
        Steady-state vector of the state variables.
    calib_params : NDF
        Calibration parameter vector passed to the measurement cfunc.
    Q : NDF
        Shock covariance matrix.
    R : NDF
        Measurement-noise covariance matrix.
    y : NDF
        Observations in declared order, shape ``(T, m)``.
    z0 : NDF
        Augmented initial state of length ``2 * n_state``, describing the state
        before the first observation.
    P0 : NDF | None
        Covariance of the augmented state, shape ``(2 * n_state, 2 * n_state)``.
    alpha : float
        Sigma-point spread parameter.
    beta : float
        Sigma-point prior-knowledge parameter.
    kappa : float
        Secondary sigma-point scaling parameter.
    symmetrize : bool
        Whether to symmetrize the covariance matrices at each step via ``(M + M.T)/2``.
    jitter : float
        Jitter to add when a cholesky decomposition fails. Set to 0.0 to disable.
    _store_history : bool
        Whether to store the full history of the filter. If False, the filter
        operates in likelihood-only mode.
    _raise_on_error : bool
        Whether to raise if the recursion fails. If False, a result carrying a
        nonzero ``status`` is returned instead.

    Returns
    -------
    UnscentedFilterResult
        Filter outputs, extending :class:`FilterResult` with the first- and
        second-order components of the state estimate.
    """
    P0, Q, R = _prep_unscented(
        meas_addr,
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
        Q,
        R,
        y,
        z0,
        P0,
        symmetrize,
    )

    # fmt: off
    err, out = ukf_forward(
        meas_addr,
        hx, gx, bu,
        hxx, gxx,
        hxu, gxu,
        huu, guu,
        hss, gss,
        steady_state,
        calib_params,
        Q, R,
        y,
        z0, P0,
        alpha, beta, kappa,
        jitter,
        symmetrize,
        _store_history,
    )
    # fmt: on
    _raise_status(err, _raise_on_error)
    return _unscented_result(out, err)


def smooth_unscented(
    meas_addr: int,
    hx: NDF,
    gx: NDF,
    bu: NDF,
    hxx: NDF,
    gxx: NDF,
    hxu: NDF,
    gxu: NDF,
    huu: NDF,
    guu: NDF,
    hss: NDF,
    gss: NDF,
    steady_state: NDF,
    calib_params: NDF,
    Q: NDF,
    R: NDF,
    y: NDF,
    z0: NDF,
    P0: NDF | None,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 1.0,
    symmetrize: bool = True,
    jitter: float = 0.0,
    _raise_on_error: bool = True,
) -> UnscentedSmootherResult:
    """Unscented Kalman smoother run against a second-order model solution.

    The policy tensors and ``steady_state`` come from an ``order=2`` solve. The
    measurement is propagated through the sigma points via ``meas_addr``, so no
    observation Jacobian is required.

    Parameters
    ----------
    meas_addr : int
        Address of the compiled measurement cfunc, applied to each sigma point.
    hx : NDF
        First-order state transition tensor.
    gx : NDF
        First-order control policy tensor.
    bu : NDF
        Shock loading tensor.
    hxx : NDF
        Second-order state transition tensor in the state.
    gxx : NDF
        Second-order control policy tensor in the state.
    hxu : NDF
        Second-order state transition cross term in state and shock.
    gxu : NDF
        Second-order control policy cross term in state and shock.
    huu : NDF
        Second-order state transition tensor in the shock.
    guu : NDF
        Second-order control policy tensor in the shock.
    hss : NDF
        Risk correction term of the state transition.
    gss : NDF
        Risk correction term of the control policy.
    steady_state : NDF
        Steady-state vector of the state variables.
    calib_params : NDF
        Calibration parameter vector passed to the measurement cfunc.
    Q : NDF
        Shock covariance matrix.
    R : NDF
        Measurement-noise covariance matrix.
    y : NDF
        Observations in declared order, shape ``(T, m)``.
    z0 : NDF
        Augmented initial state of length ``2 * n_state``, describing the state
        before the first observation.
    P0 : NDF | None
        Covariance of the augmented state, shape ``(2 * n_state, 2 * n_state)``.
    alpha : float
        Sigma-point spread parameter.
    beta : float
        Sigma-point prior-knowledge parameter.
    kappa : float
        Secondary sigma-point scaling parameter.
    symmetrize : bool
        Whether to symmetrize the covariance matrices at each step via ``(M + M.T)/2``.
    jitter : float
        Jitter to add when a cholesky decomposition fails. Set to 0.0 to disable.
    _raise_on_error : bool
        Whether to raise if the recursion fails. If False, a result carrying a
        nonzero ``status`` is returned instead.

    Returns
    -------
    UnscentedSmootherResult
        Smoother outputs, extending :class:`SmootherResult` with the first- and
        second-order components of the state estimate. Contains the forward pass
        filter result, smoothed states, state covariances, and shock estimates.
    """
    P0, Q, R = _prep_unscented(
        meas_addr,
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
        Q,
        R,
        y,
        z0,
        P0,
        symmetrize,
    )

    # fmt: off
    err, fwd, smooth = ukf_smooth(
        meas_addr,
        hx, gx, bu,
        hxx, gxx,
        hxu, gxu,
        huu, guu,
        hss, gss,
        steady_state,
        calib_params,
        Q, R,
        y,
        z0, P0,
        alpha, beta, kappa,
        jitter,
        symmetrize,
    )
    # fmt: on
    _raise_status(err, _raise_on_error)
    return _unscented_smoother_result(fwd, smooth, err)


def run_extended(
    meas_addr: int,
    jac_addr: int,
    A: NDF,
    B: NDF,
    calib_params: NDF,
    Q: NDF,
    R: NDF,
    steady_state: NDF,
    y: NDF,
    x0: NDF | None = None,
    P0: NDF | None = None,
    symmetrize: bool = True,
    joseph_cov: bool = False,
    jitter: float = 0.0,
    _store_history: bool = True,
    _raise_on_error: bool = True,
) -> FilterResult:
    """Extended Kalman Filter with a linear transition and nonlinear measurement.

    The state evolves linearly while the measurement does not::

        x_t = A x_{t-1} + B eps_t,     eps_t ~ N(0, Q)
        y_t = h(x_t, t) + v_t,         v_t   ~ N(0, R)

    The prediction step is the standard linear recursion. The update step
    linearizes the measurement around the predicted state, taking
    ``H_t = dh/dx`` evaluated at ``x_{t|t-1}``.

    Parameters
    ----------
    meas_addr : int
        Address of the compiled measurement cfunc. Evaluates ``h`` at a single
        state, returning a vector of length ``m``.
    jac_addr : int
        Address of the compiled measurement Jacobian cfunc. Returns ``H_t`` with
        shape ``(m, n)``.
    A : NDF
        State transition matrix, shape ``(n, n)``.
    B : NDF
        Shock loading matrix, shape ``(n, k)``.
    calib_params : NDF
        Calibration parameter vector passed to both cfuncs.
    Q : NDF
        Shock covariance matrix, shape ``(k, k)``. Process noise lives in shock
        space, so ``Q`` is ``(k, k)`` against a ``(n, k)`` loading.
    R : NDF
        Measurement-noise covariance matrix, shape ``(m, m)``.
    steady_state : NDF
        Steady-state vector of the state variables.
    y : NDF
        Observations in declared order, shape ``(T, m)``.
    x0 : NDF | None
        Initial state mean ``x_{0|0}``, shape ``(n,)``. None starts from zeros.
    P0 : NDF | None
        Initial state covariance ``P_{0|0}``, shape ``(n, n)``. None starts from
        ``1e2 * I_n``.
    symmetrize : bool
        Whether to symmetrize the covariance matrices at each step via ``(M + M.T)/2``.
    joseph_cov : bool
        Whether to use the Joseph form of the covariance update.
    jitter : float
        Diagonal jitter added to ``S_t``, and only when its Cholesky factorization
        fails. Set to 0.0 to disable.
    _store_history : bool
        Whether to store the full history of the filter. If False, the filter
        operates in likelihood-only mode.
    _raise_on_error : bool
        Whether to raise if the recursion fails. If False, a result carrying a
        nonzero ``status`` is returned instead.

    Returns
    -------
    FilterResult
        Predicted and filtered states, their covariances, measurements,
        innovations and the log likelihood.
    """
    _, _, x0, P0 = _prep_gaussian(A, B, Q, R, None, None, y, x0, P0, symmetrize)

    # fmt: off
    err, out = ekf_forward(
        meas_addr, jac_addr,
        A, B,
        calib_params,
        Q, R,
        steady_state,
        y,
        x0, P0,
        symmetrize,
        joseph_cov,
        jitter,
        _store_history,
    )
    # fmt: on
    _raise_status(err, _raise_on_error)
    return _filter_result(out, err)


def smooth_extended(
    meas_addr: int,
    jac_addr: int,
    A: NDF,
    B: NDF,
    calib_params: NDF,
    Q: NDF,
    R: NDF,
    steady_state: NDF,
    y: NDF,
    x0: NDF | None = None,
    P0: NDF | None = None,
    symmetrize: bool = True,
    joseph_cov: bool = False,
    jitter: float = 0.0,
    _raise_on_error: bool = True,
) -> SmootherResult[FilterResult]:
    """Extended Kalman smoother with a linear transition and nonlinear measurement.

    The state evolves linearly while the measurement does not:

        x_t = A x_{t-1} + B eps_t,     eps_t ~ N(0, Q)
        y_t = h(x_t, t) + v_t,         v_t   ~ N(0, R)

    The prediction step is the standard linear recursion. The update step
    linearizes the measurement around the predicted state, taking
    ``H_t = dh/dx`` evaluated at ``x_{t|t-1}``.

    Parameters
    ----------
    meas_addr : int
        Address of the compiled measurement cfunc. Evaluates ``h`` at a single
        state, returning a vector of length ``m``.
    jac_addr : int
        Address of the compiled measurement Jacobian cfunc. Returns ``H_t`` with
        shape ``(m, n)``.
    A : NDF
        State transition matrix, shape ``(n, n)``.
    B : NDF
        Shock loading matrix, shape ``(n, k)``.
    calib_params : NDF
        Calibration parameter vector passed to both cfuncs.
    Q : NDF
        Shock covariance matrix, shape ``(k, k)``. Process noise lives in shock
        space, so ``Q`` is ``(k, k)`` against a ``(n, k)`` loading.
    R : NDF
        Measurement-noise covariance matrix, shape ``(m, m)``.
    steady_state : NDF
        Steady-state vector of the state variables.
    y : NDF
        Observations in declared order, shape ``(T, m)``.
    x0 : NDF | None
        Initial state mean ``x_{0|0}``, shape ``(n,)``. None starts from zeros.
    P0 : NDF | None
        Initial state covariance ``P_{0|0}``, shape ``(n, n)``. None starts from
        ``1e2 * I_n``.
    symmetrize : bool
        Whether to symmetrize the covariance matrices at each step via ``(M + M.T)/2``.
    joseph_cov : bool
        Whether to use the Joseph form of the covariance update.
    jitter : float
        Diagonal jitter added to ``S_t``, and only when its Cholesky factorization
        fails. Set to 0.0 to disable.
    _raise_on_error : bool
        Whether to raise if the recursion fails. If False, a result carrying a
        nonzero ``status`` is returned instead.

    Returns
    -------
    SmootherResult[FilterResult]
        The result container of extended Kalman smoothers.
        Contains the forward pass filter result, smoothed states, state covariances, and shock estimates.
    """
    _, _, x0, P0 = _prep_gaussian(A, B, Q, R, None, None, y, x0, P0, symmetrize)

    # fmt: off
    err, fwd, smooth = ekf_smooth(
        meas_addr, jac_addr,
        A, B,
        calib_params,
        Q, R,
        steady_state,
        y,
        x0, P0,
        symmetrize,
        joseph_cov,
        jitter,
    )
    # fmt: on
    _raise_status(err, _raise_on_error)
    return _smoother_result(fwd, smooth, err)
