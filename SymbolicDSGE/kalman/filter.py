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
    asarray,
    float64,
    complex128,
    eye,
    zeros,
    linalg,
    real_if_close,
)
from numpy.typing import NDArray

from typing import Tuple, Callable

NDF = NDArray[float64]
NDC = NDArray[complex128]


@dataclass(frozen=True)
class FilterResult:
    x_pred: NDF
    x_filt: NDF

    P_pred: NDF
    P_filt: NDF

    y_pred: NDF  # y_{t|t-1} = C x_pred + d
    y_filt: NDF  # y_{t|t}   = C x_filt + d

    innov: NDF  # pre-update
    S: NDF

    loglik: float64
    eps_hat: NDF | None = None


def _get_real(mat: NDC | NDF, name: str, tol: float = 1e8) -> NDF:
    """
    Convert a complex matrix to a real matrix if the imaginary parts are negligible.
    """
    res = real_if_close(mat, tol=tol)
    if np.iscomplexobj(res):
        max_i = np.max(np.abs(res.imag))  # pyright: ignore
        raise ComplexMatrixError(name, max_i)
    return np.ascontiguousarray(res, dtype=float64)


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


OK = 0
ERR_COND = -3
ERR_LINALG = -4


@njit
def _sym(P: NDF) -> NDF:
    return 0.5 * (P + P.T)


@njit
def _chol_shifted(S: NDF, jit: float = 0.0) -> NDF:
    n = S.shape[0]
    if jit > 0.0:
        return linalg.cholesky(S + jit * eye(n, dtype=float64)).astype(float64)
    return linalg.cholesky(S).astype(float64)


@njit
def _forward_subst_vec(L: NDF, b: NDF) -> NDF:
    """
    Solve L x = b for x, where L is lower triangular (n,n), b is (n,).
    """
    n = L.shape[0]
    x = np.empty(n, dtype=float64)

    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * x[j]
        x[i] = (b[i] - s) / L[i, i]

    return x.astype(float64)


@njit
def _backward_subst_vec(U: NDF, b: NDF) -> NDF:
    """
    Solve U x = b for x, where U is upper triangular (n,n), b is (n,).
    """
    n = U.shape[0]
    x = np.empty(n, dtype=float64)

    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += U[i, j] * x[j]
        x[i] = (b[i] - s) / U[i, i]

    return x.astype(float64)


@njit
def _chol_solve_vec(L: NDF, b: NDF) -> NDF:
    """
    Solve (L L.T) x = b for x, where b is (n,).
    """
    y: NDF = _forward_subst_vec(L, b)
    LT: NDF = np.ascontiguousarray(L.T)
    x: NDF = _backward_subst_vec(LT, y)
    return x.astype(float64)


@njit
def _chol_solve_mat(L: NDF, B: NDF) -> NDF:
    """
    Solve (L L.T) X = B for X, where B is (n, k).
    """
    n, k = B.shape
    X = np.empty((n, k), dtype=float64)

    for col in range(k):
        y = _forward_subst_vec(L, B[:, col])
        x = _backward_subst_vec(L.T, y)
        X[:, col] = x

    return X


@njit
def _logdet_from_chol(L: NDF) -> float64:
    s = float64(0.0)
    n = L.shape[0]
    for i in range(n):
        s += np.log(L[i, i])
    return float64(2.0) * s


@njit
def _kalman_hot_loop(
    T: int,
    nmk: Tuple[int, int, int],
    A: NDF,
    B: NDF,
    C: NDF,
    d: NDF,
    Q: NDF,
    R: NDF,
    y: NDF,
    x_0: NDF,
    P_0: NDF,
    symmetrize: bool,
    jitter: float,
) -> tuple[
    int,
    tuple[float64, float64, float64],
    tuple[NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, float64],
]:
    n, m, k = nmk

    x_prev = x_0
    P_prev = P_0

    x_pred = zeros((T, n), dtype=float64)
    x_filt = zeros((T, n), dtype=float64)

    P_pred = zeros((T, n, n), dtype=float64)
    P_filt = zeros((T, n, n), dtype=float64)

    y_pred = zeros((T, m), dtype=float64)
    y_filt = zeros((T, m), dtype=float64)

    v = zeros((T, m), dtype=float64)
    S_hist = zeros((T, m, m), dtype=float64)

    eps_hat = zeros((T, k), dtype=float64)

    loglik = float64(0.0)
    const = m * np.log(2.0 * np.pi)

    BT = np.ascontiguousarray(B.T)
    CT = np.ascontiguousarray(C.T)
    AT = np.ascontiguousarray(A.T)

    BQBT = _sym(B @ Q @ BT)
    In = eye(n, dtype=float64)
    M = Q @ (BT @ CT)

    for t in range(T):
        x_t_pred = A @ x_prev
        P_t_pred = A @ P_prev @ AT + BQBT

        if symmetrize:
            P_t_pred = _sym(P_t_pred)

        y_t_pred = C @ x_t_pred + d
        v_t = y[t] - y_t_pred
        S_t = C @ P_t_pred @ CT + R

        if symmetrize:
            S_t = _sym(S_t)

        L = _chol_shifted(S_t, jitter)

        S_inv_v = _chol_solve_vec(L, v_t)

        PCt = P_t_pred @ CT  # (n, m)
        PCtT = np.ascontiguousarray(PCt.T)
        K_t = _chol_solve_mat(L, PCtT).T
        K_tT = np.ascontiguousarray(K_t.T)

        x_t_filt = x_t_pred + K_t @ v_t
        y_t_filt = C @ x_t_filt + d

        KC = K_t @ C
        P_t_filt = (In - KC) @ P_t_pred @ (In - KC).T + K_t @ R @ K_tT

        if symmetrize:
            P_t_filt = _sym(P_t_filt)

        ldS = _logdet_from_chol(L)
        quad = float64(v_t @ S_inv_v)
        loglik += -0.5 * (const + ldS + quad)

        eps_hat[t] = M @ S_inv_v

        x_pred[t] = x_t_pred
        x_filt[t] = x_t_filt

        P_pred[t] = P_t_pred
        P_filt[t] = P_t_filt

        y_pred[t] = y_t_pred
        y_filt[t] = y_t_filt

        v[t] = v_t
        S_hist[t] = S_t

        x_prev = x_t_filt
        P_prev = P_t_filt

    return (
        OK,
        (float64(0.0), float64(0.0), float64(0.0)),
        (
            x_pred,
            x_filt,
            P_pred,
            P_filt,
            y_pred,
            y_filt,
            v,
            S_hist,
            eps_hat,
            loglik,
        ),
    )


def _ekf_hot_loop(
    T: int,
    nmk: Tuple[int, int, int],
    A: NDF,
    B: NDF,
    h: Callable[[NDF], NDF],
    H_jac: Callable[[NDF], NDF],
    calib_params: NDF,
    Q: NDF,
    R: NDF,
    y: NDF,
    x_0: NDF,
    P_0: NDF,
    symmetrize: bool,
    jitter: float,
    compute_y_filt: bool,
) -> tuple[
    int,
    tuple[float64, float64, float64],
    tuple[NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, float64],
]:

    n, m, k = nmk
    x_prev = x_0
    P_prev = P_0

    # Outputs
    x_pred = zeros((T, n), dtype=float64)
    x_filt = zeros((T, n), dtype=float64)

    P_pred = zeros((T, n, n), dtype=float64)
    P_filt = zeros((T, n, n), dtype=float64)

    y_pred = zeros((T, m), dtype=float64)
    y_filt = zeros((T, m), dtype=float64)

    v = zeros((T, m), dtype=float64)
    S = zeros((T, m, m), dtype=float64)

    eps_hat = zeros((T, k), dtype=float64)

    loglik = float64(0.0)
    const = m * np.log(2 * np.pi)

    BT = np.ascontiguousarray(B.T)
    AT = np.ascontiguousarray(A.T)

    In = eye(n, dtype=float64)
    BQBT = _sym(B @ Q @ BT)

    for t in range(T):
        # --- Linear predict ---
        x_t_pred = A @ x_prev
        P_t_pred = A @ P_prev @ AT + BQBT

        if symmetrize:
            P_t_pred = _sym(P_t_pred)

        # --- Nonlinear measurement predict + Jacobian ---
        y_t_pred = asarray(h(*x_t_pred, *calib_params), dtype=float64).reshape(m)
        H_t = asarray(H_jac(*x_t_pred, *calib_params), dtype=float64).reshape(m, n)
        H_tT = np.ascontiguousarray(H_t.T)

        v_t = y[t] - y_t_pred
        S_t = H_t @ P_t_pred @ H_tT + R

        if symmetrize:
            S_t = _sym(S_t)

        # --- Gain/update (swap C -> H_t) ---
        L = _chol_shifted(S_t, jitter)

        v_col = v_t.reshape(m, 1)
        S_inv_v = _chol_solve_mat(L, v_col).reshape(m)

        PHt = P_t_pred @ H_tT  # (n, m)
        PHtT = np.ascontiguousarray(PHt.T)
        K_t = _chol_solve_mat(L, PHtT).T  # (n, m)
        K_tT = np.ascontiguousarray(K_t.T)

        x_t_filt = x_t_pred + K_t @ v_t

        KH = K_t @ H_t
        P_t_filt = (In - KH) @ P_t_pred @ (In - KH).T + K_t @ R @ K_tT
        if symmetrize:
            P_t_filt = _sym(P_t_filt)

        # Log-likelihood
        ldS = _logdet_from_chol(L)
        quad = float64(v_t @ S_inv_v)
        loglik += -0.5 * (const + ldS + quad)

        # Optional y_filt
        if compute_y_filt:
            y_t_filt = asarray(h(*x_t_filt, *calib_params), dtype=float64).reshape(m)
        else:
            y_t_filt = y_t_pred

        # Optional "shock estimate" (same form as linear KF)
        M = Q @ (B.T @ H_t.T)  # mirrors linear case with C -> H_t
        eps_hat[t] = M @ S_inv_v

        # Store
        x_pred[t] = x_t_pred
        x_filt[t] = x_t_filt
        P_pred[t] = P_t_pred
        P_filt[t] = P_t_filt
        y_pred[t] = y_t_pred
        y_filt[t] = y_t_filt
        v[t] = v_t
        S[t] = S_t

        x_prev = x_t_filt
        P_prev = P_t_filt

    return (
        OK,
        (float64(0.0), float64(0.0), float64(0.0)),
        (
            x_pred,
            x_filt,
            P_pred,
            P_filt,
            y_pred,
            y_filt,
            v,
            S,
            eps_hat,
            loglik,
        ),
    )


# Static & Parametrized Kalman Filter (written to act with SolvedModel object attributes)
class KalmanFilter:
    _get_real = staticmethod(_get_real)
    _shape_validate = staticmethod(_shape_validate)
    _sym = staticmethod(_sym)

    @staticmethod
    def _chol(S: NDF, jit: float = 0.0) -> NDF | None:
        try:
            out: NDF = _chol_shifted(np.ascontiguousarray(S, dtype=float64), jit)
            return out
        except linalg.LinAlgError:
            return None

    @staticmethod
    def _chol_solve(L: NDF | None, S: NDF, B: NDF) -> NDF:
        if L is not None:
            B_arr = np.ascontiguousarray(B, dtype=float64)
            if B_arr.ndim == 1:
                out: NDF = _chol_solve_vec(L, B_arr)
                return out
            out = _chol_solve_mat(L, B_arr)
            return out
        c = linalg.cond(S)
        if c > 1e12:
            raise MatrixConditionError(c)

        return linalg.solve(S, B).astype(float64)

    @staticmethod
    def _logdet(L: NDF | None, S: NDF) -> float64:
        if L is not None:
            out: float64 = _logdet_from_chol(L)
            return out

        sign, ldS = linalg.slogdet(S)
        if sign <= 0:
            raise linalg.LinAlgError(
                "Innovation covariance S is not positive definite."
            )
        return float64(ldS)

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
    ) -> FilterResult:

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

        try:
            err, err_info, out = _kalman_hot_loop(
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
            )
        except linalg.LinAlgError as exc:
            raise MatrixConditionError(float("inf")) from exc
        if err != 0:
            ErrorConstructor = get_error_constructor(ErrorCode(err))
            raise ErrorConstructor(*err_info)
        (
            x_pred,
            x_filt,
            P_pred,
            P_filt,
            y_pred,
            y_filt,
            v,
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
            S=S,
            eps_hat=eps_hat if return_shocks else None,
            loglik=loglik,
        )

    @staticmethod
    def run_extended(
        A: NDF | NDC,
        B: NDF | NDC,
        h: Callable[[NDF], NDF],
        H_jac: Callable[[NDF], NDF],
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
    ) -> "FilterResult":
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

        :param compute_y_filt: If True, compute y_filt[t] = h(x_filt[t], t). If False, set y_filt[t] = y_pred[t].
        :type compute_y_filt: bool
        """

        # Real-ify numeric inputs
        A = _get_real(A, "A")
        B = _get_real(B, "B")
        Q = _get_real(Q, "Q")
        R = _get_real(R, "R")
        y = _get_real(y, "y")

        T, m = y.shape
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

        x_prev = (
            _get_real(x0, "x0").reshape(n)
            if x0 is not None
            else zeros((n,), dtype=float64)
        )
        P_prev = (
            _get_real(P0, "P0").reshape(n, n)
            if P0 is not None
            else eye(n, dtype=float64) * 1e2
        )
        if symmetrize:
            P_prev = _sym(P_prev)

        try:
            err, err_info, out = _ekf_hot_loop(
                T,
                (n, m, k),
                A,
                B,
                h,
                H_jac,
                calib_params,
                Q,
                R,
                y,
                x_prev,
                P_prev,
                symmetrize,
                jitter,
                compute_y_filt,
            )
        except linalg.LinAlgError as exc:
            raise MatrixConditionError(float("inf")) from exc
        if err != 0:
            ErrorConstructor = get_error_constructor(ErrorCode(err))
            raise ErrorConstructor(*err_info)
        (
            x_pred,
            x_filt,
            P_pred,
            P_filt,
            y_pred,
            y_filt,
            v,
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
            S=S,
            eps_hat=eps_hat if return_shocks else None,
            loglik=loglik,
        )
