from .._ckernels.kalman import (
    stationary_covariance,
    kalman_hot_loop,
    ukf_hot_loop,
    ekf_hot_loop,
)
from .errors import (
    ErrorCode,
    ShapeMismatchError,
    get_error_constructor,
)
from dataclasses import dataclass
from numba import njit
import numpy as np
from numpy import (
    float64,
    zeros,
)
from numpy.typing import NDArray

from typing import Tuple

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
    eps_hat : NDF | None
        Conditional estimates of the structural shocks given the observed data.
        Present when the run was made with shock recovery requested, otherwise
        ``None``. Never available for the unscented filter.
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

    eps_hat: NDF | None = None
    status: int = 0


@dataclass(frozen=True, kw_only=True, slots=True)
class UnscentedFilterResult(FilterResult):
    x1_pred: NDF
    x2_pred: NDF
    x1_filt: NDF
    x2_filt: NDF


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


@njit(cache=True)
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


def _initalize_P0_unscented(
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


# Static & Parametrized Kalman Filter (written to act with SolvedModel object attributes)
class KalmanFilter:
    """Kalman filter recursions over a solved model's state space.

    A namespace of static methods; nothing is stored on an instance. :meth:`run`
    drives the linear and extended recursions, :meth:`run_unscented` the unscented
    one against a second-order solution.
    """

    _shape_validate = staticmethod(_shape_validate)

    @staticmethod
    def run(
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
        return_shocks: bool = False,
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
        return_shocks : bool
            Whether to compute shock estimates (:attr:`FilterResult.eps_hat`).
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
        T, m = y.shape  # T: time steps, m: obs dim
        n = A.shape[0]  # n: state dim
        k = B.shape[1]  # k: shock dim

        _shape_validate(
            A,
            B,
            Q,
            R,
            C,
            d,
            nmk=(n, m, k),
        )

        x_prev = x0.reshape(n) if x0 is not None else np.zeros((n,), dtype=float64)
        P_prev = _initialize_P0(P0, A, B, Q, n, symmetrize)

        err, out = kalman_hot_loop(
            T,
            (n, m, k),
            A,
            B,
            C,
            d,
            Q,
            R,
            steady_state,
            y,
            x_prev,
            P_prev,
            symmetrize,
            joseph_cov,
            jitter,
            return_shocks,
            _store_history,
        )
        if err != 0 and _raise_on_error:
            ErrorConstructor = get_error_constructor(ErrorCode(err))
            raise ErrorConstructor()
        (
            x_pred,
            x_filt,
            P_pred,
            P_filt,
            y_pred,
            y_filt,
            v,
            u,
            S,
            eps_hat,
            loglik,
        ) = out

        return FilterResult(
            x_pred=x_pred,
            x_filt=x_filt,
            P_pred=P_pred,
            P_filt=P_filt,
            y_pred=y_pred,
            y_filt=y_filt,
            innov=v,
            std_innov=u,
            S=S,
            eps_hat=eps_hat if (return_shocks and _store_history) else None,
            loglik=loglik,
            status=err,
        )

    @staticmethod
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
        observation Jacobian is required. Shock recovery is not supported.

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
        if meas_addr == 0:
            raise ValueError("meas_addr must be a nonzero measurement cfunc address.")

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
                "hxx",
                f"({n_state}, {n_state}, {n_state})",
                str(hxx.shape),
            )
        if gxx.shape != (n_ctrl, n_state, n_state):
            raise ShapeMismatchError(
                "gxx",
                f"({n_ctrl}, {n_state}, {n_state})",
                str(gxx.shape),
            )
        if hxu.shape != (n_state, n_state, n_exog):
            raise ShapeMismatchError(
                "hxu",
                f"({n_state}, {n_state}, {n_exog})",
                str(hxu.shape),
            )
        if gxu.shape != (n_ctrl, n_state, n_exog):
            raise ShapeMismatchError(
                "gxu",
                f"({n_ctrl}, {n_state}, {n_exog})",
                str(gxu.shape),
            )
        if huu.shape != (n_state, n_exog, n_exog):
            raise ShapeMismatchError(
                "huu",
                f"({n_state}, {n_exog}, {n_exog})",
                str(huu.shape),
            )
        if guu.shape != (n_ctrl, n_exog, n_exog):
            raise ShapeMismatchError(
                "guu",
                f"({n_ctrl}, {n_exog}, {n_exog})",
                str(guu.shape),
            )
        if hss.shape != (n_state,):
            raise ShapeMismatchError("hss", f"({n_state},)", str(hss.shape))
        if gss.shape != (n_ctrl,):
            raise ShapeMismatchError("gss", f"({n_ctrl},)", str(gss.shape))
        if steady_state.shape != (n_var,):
            raise ShapeMismatchError(
                "steady_state", f"({n_var},)", str(steady_state.shape)
            )
        if Q.shape != (n_exog, n_exog):
            raise ShapeMismatchError("Q", f"({n_exog}, {n_exog})", str(Q.shape))
        if y.ndim != 2:
            raise ShapeMismatchError("y", "(T, n_obs)", str(y.shape))
        n_obs = y.shape[1]
        if R.shape != (n_obs, n_obs):
            raise ShapeMismatchError("R", f"({n_obs}, {n_obs})", str(R.shape))
        if z0.shape != (n_z,):
            raise ShapeMismatchError("z0", f"({n_z},)", str(z0.shape))

        P0 = _initalize_P0_unscented(P0, hx, bu[:n_state, :], Q, n_state, symmetrize)
        if P0.shape != (n_z, n_z):
            raise ShapeMismatchError("P0", f"({n_z}, {n_z})", str(P0.shape))

        if symmetrize:
            Q = _sym(Q)  # pyright: ignore
            R = _sym(R)  # pyright: ignore

        err, out = ukf_hot_loop(
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
            calib_params,
            Q,
            R,
            y,
            z0,
            P0,
            alpha,
            beta,
            kappa,
            jitter,
            symmetrize,
            _store_history,
        )
        if err != 0 and _raise_on_error:
            ErrorConstructor = get_error_constructor(ErrorCode(err))
            raise ErrorConstructor()

        (
            x1_pred,
            x2_pred,
            x1_filt,
            x2_filt,
            x_pred,
            x_filt,
            P_pred,
            P_filt,
            y_pred,
            y_filt,
            v,
            u,
            S,
            loglik,
        ) = out
        return UnscentedFilterResult(
            x_pred=x_pred,
            x_filt=x_filt,
            P_pred=P_pred,
            P_filt=P_filt,
            y_pred=y_pred,
            y_filt=y_filt,
            innov=v,
            std_innov=u,
            S=S,
            x1_pred=x1_pred,
            x2_pred=x2_pred,
            x1_filt=x1_filt,
            x2_filt=x2_filt,
            loglik=loglik,
            status=err,
        )

    @staticmethod
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
        return_shocks: bool = False,
        symmetrize: bool = True,
        joseph_cov: bool = False,
        jitter: float = 0.0,
        compute_y_filt: bool = True,
        _store_history: bool = True,
        _raise_on_error: bool = True,
    ) -> FilterResult:
        """Extended Kalman Filter with a linear transition and nonlinear measurement.

            x_t = A x_{t-1} + B eps_t,     eps_t ~ N(0, Q)
            y_t = h(x_t, t) + v_t,         v_t   ~ N(0, R)

        The transition step is standard linear KF. The update step linearizes the
        nonlinear measurement mapping around the predicted state:

            H_t = ∂h/∂x evaluated at x_{t|t-1}

        Notes
        -----
            - `h(x, t)` must return shape (m,)
            - `H_jac(x, t)` must return shape (m, n)
            - Process noise is in "shock space": Q is (k, k), B is (n, k)

        :param A: State transition matrix with shape (n, n).
        :type A: NDF

        :param B: Shock loading matrix with shape (n, k).
        :type B: NDF

        :param h: Nonlinear measurement function. Accepts (x, t) and returns y_pred with shape (m,).
        :type h: Callable[[NDF, int], NDF]

        :param H_jac: Measurement Jacobian function. Accepts (x, t) and returns H_t = ∂h/∂x with shape (m, n).
        :type H_jac: Callable[[NDF, int], NDF]

        :param Q: Shock covariance matrix with shape (k, k).
        :type Q: NDF

        :param R: Measurement-noise covariance matrix with shape (m, m).
        :type R: NDF

        :param y: Observations array with shape (T, m).
        :type y: NDF

        :param x0: Optional initial state mean x_{0|0} with shape (n,). Defaults to zeros.
        :type x0: NDF | None

        :param P0: Optional initial state covariance P_{0|0} with shape (n, n). Defaults to 1e2 * I_n.
        :type P0: NDF | None

        :param return_shocks: If True, compute eps_hat (shock estimates) using the same formula as the linear KF.
                              Interpretable only if the innovation-to-shock mapping is meaningful under your measurement design.
        :type return_shocks: bool

        :param symmetrize: If True, symmetrize P and S matrices at each step via (M+M.T)/2.
        :type symmetrize: bool

        :param jitter: Diagonal jitter added to S_t only if Cholesky factorization fails. Set to 0.0 to disable.
        :type jitter: float

        :param compute_y_filt: If True, compute y_filt[t] = h(x_filt[t], t). If False, leave y_filt as zeros with shape (T, m).
        :type compute_y_filt: bool
        """
        _, m = y.shape
        n = A.shape[0]
        k = B.shape[1]

        # Shapes (reuse existing helper; C/d not used here)
        _shape_validate(
            A,
            B,
            Q,
            R,
            C=None,
            d=None,
            nmk=(n, m, k),
        )

        x0 = x0.reshape(n) if x0 is not None else zeros((n,), dtype=float64)
        P0 = _initialize_P0(P0, A, B, Q, n, symmetrize)
        err, out = ekf_hot_loop(
            meas_addr,
            jac_addr,
            A,
            B,
            calib_params,
            Q,
            R,
            steady_state,
            y,
            x0,
            P0,
            symmetrize,
            joseph_cov,
            jitter,
            compute_y_filt,
            return_shocks,
            _store_history,
        )

        if err != 0 and _raise_on_error:
            ErrorConstructor = get_error_constructor(ErrorCode(err))
            raise ErrorConstructor()
        (
            x_pred,
            x_filt,
            P_pred,
            P_filt,
            y_pred,
            y_filt,
            v,
            u,
            S,
            eps_hat,
            loglik,
        ) = out

        return FilterResult(
            x_pred=x_pred,
            x_filt=x_filt,
            P_pred=P_pred,
            P_filt=P_filt,
            y_pred=y_pred,
            y_filt=y_filt,
            innov=v,
            std_innov=u,
            S=S,
            eps_hat=eps_hat if (return_shocks and _store_history) else None,
            loglik=loglik,
            status=err,
        )
