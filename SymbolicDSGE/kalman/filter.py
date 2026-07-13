from .._ckernels.kalman import kalman_hot_loop, ukf_hot_loop, ekf_hot_loop
from .errors import (
    ErrorCode,
    ComplexMatrixError,
    MatrixConditionError,
    ShapeMismatchError,
    get_error_constructor,
)
from dataclasses import dataclass
from numba import njit
import numpy as np
from numpy import (
    float64,
    complex128,
    eye,
    zeros,
    linalg,
    real_if_close,
)
from numpy.typing import NDArray

from typing import Tuple, NamedTuple

NDF = NDArray[float64]
NDC = NDArray[complex128]


@dataclass(frozen=True, slots=True)
class FilterResult:
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


@dataclass(frozen=True, kw_only=True, slots=True)
class UnscentedFilterResult(FilterResult):
    x1_pred: NDF
    x2_pred: NDF
    x1_filt: NDF
    x2_filt: NDF

    loglik: float64


class FilterRawResult(NamedTuple):
    x_pred: NDF
    x_filt: NDF
    P_pred: NDF
    P_filt: NDF
    y_pred: NDF
    y_filt: NDF
    innov: NDF
    std_innov: NDF
    S: NDF
    eps_hat: NDF | None
    loglik: float64
    status: int


class UnscentedFilterRawResult(NamedTuple):
    x_pred: NDF
    x_filt: NDF
    P_pred: NDF
    P_filt: NDF
    y_pred: NDF
    y_filt: NDF
    innov: NDF
    std_innov: NDF
    S: NDF
    eps_hat: NDF | None
    loglik: float64
    x1_pred: NDF
    x2_pred: NDF
    x1_filt: NDF
    x2_filt: NDF
    status: int


def _filter_result_from_raw(raw: FilterRawResult) -> FilterResult:
    return FilterResult(
        x_pred=raw.x_pred,
        x_filt=raw.x_filt,
        P_pred=raw.P_pred,
        P_filt=raw.P_filt,
        y_pred=raw.y_pred,
        y_filt=raw.y_filt,
        innov=raw.innov,
        std_innov=raw.std_innov,
        S=raw.S,
        eps_hat=raw.eps_hat,
        loglik=raw.loglik,
    )


def _unscented_filter_result_from_raw(
    raw: UnscentedFilterRawResult,
) -> UnscentedFilterResult:
    return UnscentedFilterResult(
        x_pred=raw.x_pred,
        x_filt=raw.x_filt,
        x1_pred=raw.x1_pred,
        x2_pred=raw.x2_pred,
        x1_filt=raw.x1_filt,
        x2_filt=raw.x2_filt,
        P_pred=raw.P_pred,
        P_filt=raw.P_filt,
        y_pred=raw.y_pred,
        y_filt=raw.y_filt,
        innov=raw.innov,
        std_innov=raw.std_innov,
        S=raw.S,
        eps_hat=raw.eps_hat,
        loglik=raw.loglik,
    )


def _get_real(mat: NDC | NDF, name: str, tol: float = 1e8) -> NDF:
    """
    Convert a complex matrix to a real matrix if the imaginary parts are negligible.
    """
    res = real_if_close(mat, tol=tol)
    if np.iscomplexobj(res):
        raise ComplexMatrixError(name, np.max(np.abs(res.imag)))  # pyright: ignore
    return res


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


# Static & Parametrized Kalman Filter (written to act with SolvedModel object attributes)
class KalmanFilter:
    _get_real = staticmethod(_get_real)
    _shape_validate = staticmethod(_shape_validate)

    @staticmethod
    def run_raw(
        A: NDF | NDC,
        B: NDF | NDC,
        C: NDF | NDC,
        d: NDF | NDC,
        Q: NDF | NDC,
        R: NDF | NDC,
        y: NDF | NDC,
        x0: NDF | None = None,
        P0: NDF | None = None,
        return_shocks: bool = False,
        symmetrize: bool = True,
        jitter: float = 0.0,
        _store_history: bool = True,
        _raise_on_error: bool = True,
    ) -> FilterRawResult:

        # Get reals if needed
        A = _get_real(A, "A")
        B = _get_real(B, "B")
        C = _get_real(C, "C")

        d = _get_real(d, "d").reshape(-1)
        Q = _get_real(Q, "Q")
        R = _get_real(R, "R")

        y = _get_real(y, "y")

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

        x_prev = (
            _get_real(x0, "x0").reshape(n)
            if x0 is not None
            else np.zeros((n,), dtype=float64)
        )
        P_prev = (
            _get_real(P0, "P0").reshape(n, n)
            if P0 is not None
            else eye(n, dtype=float64) * 1e2
        )

        if symmetrize:
            P_prev = _sym(P_prev)

        err, out = kalman_hot_loop(
            T,
            (n, m, k),
            A,
            B,
            C,
            d,
            Q,
            R,
            y,
            x_prev,
            P_prev,
            symmetrize,
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

        return FilterRawResult(
            status=err,
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
        )

    @staticmethod
    def run(
        A: NDF | NDC,
        B: NDF | NDC,
        C: NDF | NDC,
        d: NDF | NDC,
        Q: NDF | NDC,
        R: NDF | NDC,
        y: NDF | NDC,
        x0: NDF | None = None,
        P0: NDF | None = None,
        return_shocks: bool = False,
        symmetrize: bool = True,
        jitter: float = 0.0,
        _store_history: bool = True,
    ) -> FilterResult:
        return _filter_result_from_raw(
            KalmanFilter.run_raw(
                A=A,
                B=B,
                C=C,
                d=d,
                Q=Q,
                R=R,
                y=y,
                x0=x0,
                P0=P0,
                return_shocks=return_shocks,
                symmetrize=symmetrize,
                jitter=jitter,
                _store_history=_store_history,
            )
        )

    @staticmethod
    def run_unscented_raw(
        meas_addr: int,
        hx: NDF | NDC,
        gx: NDF | NDC,
        bx: NDF | NDC,
        hxx: NDF | NDC,
        gxx: NDF | NDC,
        hss: NDF | NDC,
        gss: NDF | NDC,
        steady_state: NDF | NDC,
        calib_params: NDF | NDC,
        Q: NDF | NDC,
        R: NDF | NDC,
        y: NDF | NDC,
        z0: NDF | NDC,
        P0: NDF | NDC,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 1.0,
        symmetrize: bool = True,
        jitter: float = 0.0,
        _store_history: bool = True,
        _raise_on_error: bool = True,
    ) -> UnscentedFilterRawResult:
        if meas_addr == 0:
            raise ValueError("meas_addr must be a nonzero measurement cfunc address.")

        hx = _get_real(hx, "hx")
        bx = _get_real(bx, "bx")
        hxx = _get_real(hxx, "hxx")
        hss = _get_real(hss, "hss").reshape(-1)
        steady_state = _get_real(steady_state, "steady_state").reshape(-1)
        calib_params = _get_real(calib_params, "calib_params").reshape(-1)
        Q_real: NDF = _get_real(Q, "Q")  # narrow for mypy
        R_real: NDF = _get_real(R, "R")  # narrow for mypy
        y = _get_real(y, "y")
        z0 = _get_real(z0, "z0").reshape(-1)
        P0_real: NDF = _get_real(P0, "P0")  # narrow for mypy

        gx = _get_real(gx, "gx")
        gxx = _get_real(gxx, "gxx")
        gss = _get_real(gss, "gss").reshape(-1)

        if hx.ndim != 2 or hx.shape[0] != hx.shape[1]:
            raise ShapeMismatchError("hx", "(n_state, n_state)", str(hx.shape))
        n_state = hx.shape[0]
        n_z = 2 * n_state

        if bx.ndim != 2 or bx.shape[0] != n_state:
            raise ShapeMismatchError("bx", f"({n_state}, n_exog)", str(bx.shape))
        n_exog = bx.shape[1]

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
        if P0.shape != (n_z, n_z):
            raise ShapeMismatchError("P0", f"({n_z}, {n_z})", str(P0.shape))

        if symmetrize:
            Q_real = _sym(Q)  # pyright: ignore
            R_real = _sym(R)  # pyright: ignore
            P0_real = _sym(P0)  # pyright: ignore

        err, out = ukf_hot_loop(
            meas_addr,
            hx,
            gx,
            bx,
            hxx,
            gxx,
            hss,
            gss,
            steady_state,
            calib_params,
            Q_real,
            R_real,
            y,
            z0,
            P0_real,
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

        return UnscentedFilterRawResult(
            status=err,
            x_pred=x_pred,
            x_filt=x_filt,
            P_pred=P_pred,
            P_filt=P_filt,
            y_pred=y_pred,
            y_filt=y_filt,
            innov=v,
            std_innov=u,
            S=S,
            eps_hat=None,
            loglik=loglik,
            x1_pred=x1_pred,
            x2_pred=x2_pred,
            x1_filt=x1_filt,
            x2_filt=x2_filt,
        )

    @staticmethod
    def run_unscented(
        meas_addr: int,
        hx: NDF | NDC,
        gx: NDF | NDC,
        bx: NDF | NDC,
        hxx: NDF | NDC,
        gxx: NDF | NDC,
        hss: NDF | NDC,
        gss: NDF | NDC,
        steady_state: NDF | NDC,
        calib_params: NDF | NDC,
        Q: NDF | NDC,
        R: NDF | NDC,
        y: NDF | NDC,
        z0: NDF | NDC,
        P0: NDF | NDC,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 1.0,
        symmetrize: bool = True,
        jitter: float = 0.0,
        _store_history: bool = True,
    ) -> UnscentedFilterResult:
        return _unscented_filter_result_from_raw(
            KalmanFilter.run_unscented_raw(
                meas_addr=meas_addr,
                hx=hx,
                gx=gx,
                bx=bx,
                hxx=hxx,
                gxx=gxx,
                hss=hss,
                gss=gss,
                steady_state=steady_state,
                calib_params=calib_params,
                Q=Q,
                R=R,
                y=y,
                z0=z0,
                P0=P0,
                alpha=alpha,
                beta=beta,
                kappa=kappa,
                symmetrize=symmetrize,
                jitter=jitter,
                _store_history=_store_history,
            )
        )

    @staticmethod
    def run_extended_raw(
        meas_addr: int,
        jac_addr: int,
        A: NDF | NDC,
        B: NDF | NDC,
        calib_params: NDF,
        Q: NDF | NDC,
        R: NDF | NDC,
        y: NDF | NDC,
        x0: NDF | None = None,
        P0: NDF | None = None,
        return_shocks: bool = False,
        symmetrize: bool = True,
        jitter: float = 0.0,
        compute_y_filt: bool = True,
        _store_history: bool = True,
        _raise_on_error: bool = True,
    ) -> FilterRawResult:
        """
        Extended Kalman Filter with a linear transition and nonlinear measurement:

            x_t = A x_{t-1} + B eps_t,     eps_t ~ N(0, Q)
            y_t = h(x_t, t) + v_t,         v_t   ~ N(0, R)

        The transition step is standard linear KF. The update step linearizes the
        nonlinear measurement mapping around the predicted state:

            H_t = ∂h/∂x evaluated at x_{t|t-1}

        Notes:
            - `h(x, t)` must return shape (m,)
            - `H_jac(x, t)` must return shape (m, n)
            - Process noise is in "shock space": Q is (k, k), B is (n, k)

        :param A: State transition matrix with shape (n, n).
        :type A: NDF | NDC

        :param B: Shock loading matrix with shape (n, k).
        :type B: NDF | NDC

        :param h: Nonlinear measurement function. Accepts (x, t) and returns y_pred with shape (m,).
        :type h: Callable[[NDF, int], NDF]

        :param H_jac: Measurement Jacobian function. Accepts (x, t) and returns H_t = ∂h/∂x with shape (m, n).
        :type H_jac: Callable[[NDF, int], NDF]

        :param Q: Shock covariance matrix with shape (k, k).
        :type Q: NDF | NDC

        :param R: Measurement-noise covariance matrix with shape (m, m).
        :type R: NDF | NDC

        :param y: Observations array with shape (T, m).
        :type y: NDF | NDC

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

        # Real-ify numeric inputs
        A = _get_real(A, "A")
        B = _get_real(B, "B")
        Q = _get_real(Q, "Q")
        R = _get_real(R, "R")
        y = _get_real(y, "y")

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

        x0 = (
            _get_real(x0, "x0").reshape(n)
            if x0 is not None
            else zeros((n,), dtype=float64)
        )
        P0 = (
            _get_real(P0, "P0").reshape(n, n)
            if P0 is not None
            else eye(n, dtype=float64) * 1e2
        )
        if symmetrize:
            P0 = _sym(P0)

        err, out = ekf_hot_loop(
            meas_addr,
            jac_addr,
            A,
            B,
            calib_params,
            Q,
            R,
            y,
            x0,
            P0,
            symmetrize,
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

        return FilterRawResult(
            status=err,
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
        )

    @staticmethod
    def run_extended(
        meas_addr: int,
        jac_addr: int,
        A: NDF | NDC,
        B: NDF | NDC,
        calib_params: NDF,
        Q: NDF | NDC,
        R: NDF | NDC,
        y: NDF | NDC,
        x0: NDF | None = None,
        P0: NDF | None = None,
        return_shocks: bool = False,
        symmetrize: bool = True,
        jitter: float = 0.0,
        compute_y_filt: bool = True,
        _store_history: bool = True,
    ) -> FilterResult:
        return _filter_result_from_raw(
            KalmanFilter.run_extended_raw(
                meas_addr=meas_addr,
                jac_addr=jac_addr,
                A=A,
                B=B,
                calib_params=calib_params,
                Q=Q,
                R=R,
                y=y,
                x0=x0,
                P0=P0,
                return_shocks=return_shocks,
                symmetrize=symmetrize,
                jitter=jitter,
                compute_y_filt=compute_y_filt,
                _store_history=_store_history,
            )
        )
