from dataclasses import dataclass

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


class ComplexMatrixError(Exception):
    def __init__(self, name: str, max_imag: float64) -> None:
        message = f"Matrix '{name}' has significant imaginary parts (max abs imag: {max_imag})."
        super().__init__(message)


class ShapeMismatchError(Exception):
    def __init__(self, name: str, exp_shape: str, cur_shape: str) -> None:
        message = f"Matrix '{name}' has incompatible shape. Expected: {exp_shape}, got: {cur_shape}."
        super().__init__(message)


class MatrixConditionError(Exception):
    def __init__(self, cond: float64) -> None:
        message = f"Matrix is ill-conditioned with condition number: {float(cond)}."
        super().__init__(message)


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


# Static & Parametrized Kalman Filter (written to act with SolvedModel object attributes)
class KalmanFilter:

    @staticmethod
    def _get_real(mat: NDC | NDF, name: str, tol: float = 1e8) -> NDF:
        """
        Convert a complex matrix to a real matrix if the imaginary parts are negligible.
        """
        res = real_if_close(mat, tol=tol)
        if np.iscomplexobj(res):
            max_i = np.max(np.abs(res.imag))  # pyright: ignore
            raise ComplexMatrixError(name, max_i)
        return asarray(res, dtype=float64)

    @staticmethod
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

    @staticmethod
    def _sym(P: NDF) -> NDF:
        return (P + P.T) / 2

    @staticmethod
    def _chol(S: NDF, jit: float = 0.0) -> NDF | None:
        """Attempt Cholesky, add jitter if fails."""
        try:
            return linalg.cholesky(S).astype(float64)

        except linalg.LinAlgError:
            try:
                if jit == 0.0:
                    raise  # Skip adding jitter if not specified

                # Add jitter and try again
                jitter = jit * np.eye(S.shape[0], dtype=float64)
                return linalg.cholesky(S + jitter).astype(float64)

            except linalg.LinAlgError:
                return None

    @staticmethod
    def _chol_solve(L: NDArray[float64] | None, S: NDF, B: NDF) -> NDF:
        """Solve Sx = B using Cholesky if possible, else standard solve."""
        if L is not None:
            # Use Cholesky factors to solve
            y = linalg.solve(L, B)
            return linalg.solve(L.T, y).astype(float64)
        else:
            # Fall back to standard solve
            c = linalg.cond(S)
            if c > 1e12:
                raise MatrixConditionError(c)

            return linalg.solve(S, B).astype(float64)

    @staticmethod
    def _logdet(L: NDF | None, S: NDF) -> float64:
        """Attempt Log Determinant via Cholesky, else use slogdet."""
        if L is not None:
            ldS = 2.0 * np.sum(np.log(np.diag(L)))
        else:
            sign, ldS = linalg.slogdet(S)
            if sign <= 0:
                raise linalg.LinAlgError(
                    "Innovation covariance S is not positive definite."
                )  # Only S uses slogdet
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
        A = KalmanFilter._get_real(A, "A")
        B = KalmanFilter._get_real(B, "B")
        C = KalmanFilter._get_real(C, "C")

        d = KalmanFilter._get_real(d, "d").reshape(-1)
        Q = KalmanFilter._get_real(Q, "Q")
        R = KalmanFilter._get_real(R, "R")

        y = KalmanFilter._get_real(y, "y")

        T, m = y.shape  # T: time steps, m: obs dim
        n = A.shape[0]  # n: state dim
        k = B.shape[1]  # k: shock dim

        KalmanFilter._shape_validate(
            A,
            B,
            Q,
            R,
            C,
            d,
            nmk=(n, m, k),
        )

        x_prev = (
            KalmanFilter._get_real(x0, "x0").reshape(n)
            if x0 is not None
            else np.zeros((n,), dtype=float64)
        )
        P_prev = (
            KalmanFilter._get_real(P0, "P0").reshape(n, n)
            if P0 is not None
            else eye(n, dtype=float64) * 1e2
        )

        if symmetrize:
            P_prev = KalmanFilter._sym(P_prev)

        # Out Arrays
        x_pred = zeros((T, n), dtype=float64)
        x_filt = zeros((T, n), dtype=float64)

        P_pred = zeros((T, n, n), dtype=float64)
        P_filt = zeros((T, n, n), dtype=float64)

        y_pred = zeros((T, m), dtype=float64)
        y_filt = zeros((T, m), dtype=float64)

        v = zeros((T, m), dtype=float64)  # innovations
        S = zeros((T, m, m), dtype=float64)  # innovation cov

        eps_hat = zeros((T, k), dtype=float64) if return_shocks else None

        loglik = float64(0.0)
        const = m * np.log(2 * np.pi)

        BQBT = KalmanFilter._sym(B @ Q @ B.T)  # (n, n)
        In = eye(n, dtype=float64)

        for t in range(T):
            x_t_pred = A @ x_prev
            P_t_pred = A @ P_prev @ A.T + BQBT

            if symmetrize:
                P_t_pred = KalmanFilter._sym(P_t_pred)
            y_t_pred = C @ x_t_pred + d
            v_t = y[t] - y_t_pred
            S_t = C @ P_t_pred @ C.T + R

            if symmetrize:
                S_t = KalmanFilter._sym(S_t)

            # GAIN: K = P * C' * S^-1

            L = KalmanFilter._chol(S_t, jitter)
            v_col = v_t.reshape(m, 1)
            S_inv_v = KalmanFilter._chol_solve(L, S_t, v_col).reshape(m)

            PCt = P_t_pred @ C.T
            K_t = KalmanFilter._chol_solve(L, S_t, PCt.T).T  # (n, m)

            # Update outputs
            x_t_filt = x_t_pred + K_t @ v_t
            y_t_filt = C @ x_t_filt + d

            KC = K_t @ C
            P_t_filt = (In - KC) @ P_t_pred @ (In - KC).T + K_t @ R @ K_t.T
            if symmetrize:
                P_t_filt = KalmanFilter._sym(P_t_filt)

            ldS = KalmanFilter._logdet(L, S_t)
            quad = float64(v_t @ S_inv_v)
            loglik += -0.5 * (const + ldS + quad)

            if return_shocks and eps_hat is not None:
                M = Q @ (B.T @ C.T)
                eps_hat[t] = M @ S_inv_v

            # Store results
            x_pred[t] = x_t_pred
            x_filt[t] = x_t_filt

            P_pred[t] = P_t_pred
            P_filt[t] = P_t_filt

            y_pred[t] = y_t_pred
            y_filt[t] = y_t_filt

            v[t] = v_t
            S[t] = S_t

            # Prepare next iteration
            x_prev = x_t_filt
            P_prev = P_t_filt

        return FilterResult(
            x_pred=x_pred,
            x_filt=x_filt,
            P_pred=P_pred,
            P_filt=P_filt,
            y_pred=y_pred,
            y_filt=y_filt,
            innov=v,
            S=S,
            eps_hat=eps_hat,
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
        A = KalmanFilter._get_real(A, "A")
        B = KalmanFilter._get_real(B, "B")
        Q = KalmanFilter._get_real(Q, "Q")
        R = KalmanFilter._get_real(R, "R")
        y = KalmanFilter._get_real(y, "y")

        T, m = y.shape
        n = A.shape[0]
        k = B.shape[1]

        # Shapes (reuse existing helper; C/d not used here)
        KalmanFilter._shape_validate(
            A,
            B,
            Q,
            R,
            C=None,
            d=None,
            nmk=(n, m, k),
        )

        x_prev = (
            KalmanFilter._get_real(x0, "x0").reshape(n)
            if x0 is not None
            else zeros((n,), dtype=float64)
        )
        P_prev = (
            KalmanFilter._get_real(P0, "P0").reshape(n, n)
            if P0 is not None
            else eye(n, dtype=float64) * 1e2
        )
        if symmetrize:
            P_prev = KalmanFilter._sym(P_prev)

        # Outputs
        x_pred = zeros((T, n), dtype=float64)
        x_filt = zeros((T, n), dtype=float64)

        P_pred = zeros((T, n, n), dtype=float64)
        P_filt = zeros((T, n, n), dtype=float64)

        y_pred = zeros((T, m), dtype=float64)
        y_filt = zeros((T, m), dtype=float64)

        v = zeros((T, m), dtype=float64)
        S = zeros((T, m, m), dtype=float64)

        eps_hat = zeros((T, k), dtype=float64) if return_shocks else None

        loglik = float64(0.0)
        const = m * np.log(2 * np.pi)

        In = eye(n, dtype=float64)
        BQBT = KalmanFilter._sym(B @ Q @ B.T)

        for t in range(T):
            # --- Linear predict ---
            x_t_pred = A @ x_prev
            P_t_pred = A @ P_prev @ A.T + BQBT

            if symmetrize:
                P_t_pred = KalmanFilter._sym(P_t_pred)

            # --- Nonlinear measurement predict + Jacobian ---
            y_t_pred = asarray(h(*x_t_pred, *calib_params), dtype=float64).reshape(m)
            H_t = asarray(H_jac(*x_t_pred, *calib_params), dtype=float64).reshape(m, n)

            v_t = y[t] - y_t_pred
            S_t = H_t @ P_t_pred @ H_t.T + R

            if symmetrize:
                S_t = KalmanFilter._sym(S_t)

            # --- Gain/update (swap C -> H_t) ---
            L = KalmanFilter._chol(S_t, jitter)

            v_col = v_t.reshape(m, 1)
            S_inv_v = KalmanFilter._chol_solve(L, S_t, v_col).reshape(m)

            PHt = P_t_pred @ H_t.T  # (n, m)
            K_t = KalmanFilter._chol_solve(L, S_t, PHt.T).T  # (n, m)

            x_t_filt = x_t_pred + K_t @ v_t

            KH = K_t @ H_t
            P_t_filt = (In - KH) @ P_t_pred @ (In - KH).T + K_t @ R @ K_t.T
            if symmetrize:
                P_t_filt = KalmanFilter._sym(P_t_filt)

            # Log-likelihood
            ldS = KalmanFilter._logdet(L, S_t)
            quad = float64(v_t @ S_inv_v)
            loglik += -0.5 * (const + ldS + quad)

            # Optional y_filt
            if compute_y_filt:
                y_t_filt = asarray(h(*x_t_filt, *calib_params), dtype=float64).reshape(
                    m
                )
            else:
                y_t_filt = y_t_pred

            # Optional "shock estimate" (same form as linear KF)
            if return_shocks and eps_hat is not None:
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

        return FilterResult(
            x_pred=x_pred,
            x_filt=x_filt,
            P_pred=P_pred,
            P_filt=P_filt,
            y_pred=y_pred,
            y_filt=y_filt,
            innov=v,
            S=S,
            eps_hat=eps_hat,
            loglik=loglik,
        )
