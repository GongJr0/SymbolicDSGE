from dataclasses import dataclass

import numpy as np
from numpy import (
    ndarray,
    asarray,
    float64,
    complex128,
    eye,
    zeros,
    linalg,
    real_if_close,
)
from numpy.typing import NDArray

from typing import Tuple


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


@dataclass(frozen=True)
class FilterResult:
    x_pred: NDF
    x_filt: NDF

    P_pred: NDF
    P_filt: NDF

    y_pred: NDF
    innov: NDF
    S: NDF

    eps_hat: NDF | None = None
    loglik: float64 | None = None


# Static & Parametrized Kalman Filter (written to act with SolvedModel object attributes)
class KalmanFilter:

    @staticmethod
    def _get_real(mat: NDC | NDF, name: str, tol: float = 1e8) -> NDF:
        """
        Convert a complex matrix to a real matrix if the imaginary parts are negligible.
        """
        res = real_if_close(mat, tol=tol)
        if np.iscomplexobj(res):
            max_i = np.max(np.abs(res.imag))
            raise ComplexMatrixError(name, max_i)
        return asarray(res, dtype=float64)

    @staticmethod
    def _shape_validate(
        A: NDF, B: NDF, C: NDF, d: NDF, Q: NDF, R: NDF, nmk: Tuple[int, int, int]
    ) -> None:
        n, m, k = nmk
        if A.shape != (n, n):
            raise ShapeMismatchError("A", f"({n}, {n})", str(A.shape))
        if B.shape != (n, k):
            raise ShapeMismatchError("B", f"({n}, {k})", str(B.shape))
        if C.shape != (m, n):
            raise ShapeMismatchError("C", f"({m}, {n})", str(C.shape))
        if d.shape != (m,):
            raise ShapeMismatchError("d", f"({m},)", str(d.shape))
        if Q.shape != (k, k):
            raise ShapeMismatchError("Q", f"({k}, {k})", str(Q.shape))
        if R.shape != (m, m):
            raise ShapeMismatchError("R", f"({m}, {m})", str(R.shape))

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
            C,
            d,
            Q,
            R,
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
            innov=v,
            S=S,
            eps_hat=eps_hat,
            loglik=loglik,
        )
