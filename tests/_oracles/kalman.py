"""Reference (oracle) implementations of the Kalman hot loops.

Pure-numpy transcriptions of the native kernels in
``SymbolicDSGE/_ckernels/kalman/kalman.c``, used to validate the compiled
extension. These favor readability over speed and take Python callables where
the native kernel takes ``@cfunc`` addresses, so the parity tests can feed the
same measurement to both sides.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy import float64
from numpy.linalg import LinAlgError, cholesky
from numpy.typing import NDArray
from scipy.linalg import solve_triangular

from SymbolicDSGE.kalman.filter import FilterRawResult

NDF = NDArray[float64]


def _sym(M: NDF) -> NDF:
    return 0.5 * (M + M.T)


def ekf_reference(
    A: NDF,
    B: NDF,
    h: Callable[[NDF, NDF], NDF],
    H_jac: Callable[[NDF, NDF], NDF],
    calib_params: NDF,
    Q: NDF,
    R: NDF,
    y: NDF,
    x0: NDF | None = None,
    P0: NDF | None = None,
    *,
    symmetrize: bool = True,
    jitter: float = 0.0,
    compute_y_filt: bool = True,
    return_shocks: bool = False,
    store_history: bool = True,
) -> FilterRawResult:
    """Extended Kalman filter reference: linear transition, nonlinear measurement.

    ``h(x, params) -> (m,)`` and ``H_jac(x, params) -> (m, n)`` are Python
    callables (the native kernel takes their ``@cfunc`` addresses instead). The
    recursion, Joseph-form update, Cholesky-based whitening, and log-likelihood
    match ``ekf_hot_loop`` in ``kalman.c`` term for term.
    """
    A = np.asarray(A, dtype=float64)
    B = np.asarray(B, dtype=float64)
    Q = np.asarray(Q, dtype=float64)
    R = np.asarray(R, dtype=float64)
    y = np.asarray(y, dtype=float64)
    params = np.asarray(calib_params, dtype=float64)

    T, m = y.shape
    n = A.shape[0]
    k = B.shape[1]

    x_prev = (
        np.zeros((n,), dtype=float64)
        if x0 is None
        else np.asarray(x0, dtype=float64).reshape(n)
    )
    P_prev = (
        np.eye(n, dtype=float64) * 1e2
        if P0 is None
        else np.asarray(P0, dtype=float64).reshape(n, n)
    )
    if symmetrize:
        P_prev = _sym(P_prev)

    BQBt = _sym(B @ Q @ B.T)

    hist_T = T if store_history else 0
    shock_T = T if (return_shocks and store_history) else 0

    x_pred_h = np.zeros((hist_T, n), dtype=float64)
    x_filt_h = np.zeros((hist_T, n), dtype=float64)
    P_pred_h = np.zeros((hist_T, n, n), dtype=float64)
    P_filt_h = np.zeros((hist_T, n, n), dtype=float64)
    y_pred_h = np.zeros((hist_T, m), dtype=float64)
    y_filt_h = np.zeros((hist_T, m), dtype=float64)
    innov_h = np.zeros((hist_T, m), dtype=float64)
    std_innov_h = np.zeros((hist_T, m), dtype=float64)
    S_h = np.zeros((hist_T, m, m), dtype=float64)
    eps_hat_h = np.zeros((shock_T, k), dtype=float64)

    loglik = 0.0
    const = m * np.log(2.0 * np.pi)
    eye_n = np.eye(n, dtype=float64)

    for t in range(T):
        x_pred = A @ x_prev
        P_pred = A @ P_prev @ A.T + BQBt
        if symmetrize:
            P_pred = _sym(P_pred)

        y_pred = np.asarray(h(x_pred, params), dtype=float64).reshape(m)
        H = np.asarray(H_jac(x_pred, params), dtype=float64).reshape(m, n)

        v = y[t] - y_pred
        S = H @ P_pred @ H.T + R
        if symmetrize:
            S = _sym(S)

        try:
            L = cholesky(S)
        except LinAlgError:
            L = cholesky(S + jitter * np.eye(m, dtype=float64))

        # u = L^{-1} v (whitened innovation); S_inv_v = S^{-1} v via the factor.
        u = solve_triangular(L, v, lower=True)
        S_inv_v = solve_triangular(L.T, u, lower=False)

        # K = P_pred H^T S^{-1}
        PHt = P_pred @ H.T
        gain_u = solve_triangular(L, PHt.T, lower=True)
        K = solve_triangular(L.T, gain_u, lower=False).T

        x_filt = x_pred + K @ v
        I_minus_KH = eye_n - K @ H
        P_filt = I_minus_KH @ P_pred @ I_minus_KH.T + K @ R @ K.T
        if symmetrize:
            P_filt = _sym(P_filt)

        logdet_S = 2.0 * np.sum(np.log(np.diag(L)))
        loglik += -0.5 * (const + logdet_S + v @ S_inv_v)

        if return_shocks and store_history:
            M = Q @ B.T @ H.T
            eps_hat_h[t] = M @ S_inv_v

        if store_history:
            x_pred_h[t] = x_pred
            x_filt_h[t] = x_filt
            P_pred_h[t] = P_pred
            P_filt_h[t] = P_filt
            y_pred_h[t] = y_pred
            if compute_y_filt:
                y_filt_h[t] = np.asarray(h(x_filt, params), dtype=float64).reshape(m)
            innov_h[t] = v
            std_innov_h[t] = u
            S_h[t] = S

        x_prev = x_filt
        P_prev = P_filt

    return FilterRawResult(
        status=0,
        x_pred=x_pred_h,
        x_filt=x_filt_h,
        P_pred=P_pred_h,
        P_filt=P_filt_h,
        y_pred=y_pred_h,
        y_filt=y_filt_h,
        innov=innov_h,
        std_innov=std_innov_h,
        S=S_h,
        eps_hat=eps_hat_h if (return_shocks and store_history) else None,
        loglik=float64(loglik),
    )


def kf_reference(
    T: int,
    nmk: tuple[int, int, int],
    A: NDF,
    B: NDF,
    C: NDF,
    d: NDF,
    Q: NDF,
    R: NDF,
    y: NDF,
    x0: NDF,
    P0: NDF,
    symmetrize: bool,
    jitter: float,
    return_shocks: bool = False,
    store_history: bool = True,
) -> tuple:
    """Linear Kalman filter reference (fixed affine measurement ``y = C x + d``).

    Pure-numpy transcription of ``kf_hot_loop`` in ``kalman.c``. The positional
    signature mirrors the native ``kalman_hot_loop`` shim, and it returns the same
    11-element history tuple ``(x_pred, x_filt, P_pred, P_filt, y_pred, y_filt,
    innov, std_innov, S, eps_hat, loglik)``.
    """
    n, m, k = nmk
    A = np.asarray(A, dtype=float64)
    B = np.asarray(B, dtype=float64)
    C = np.asarray(C, dtype=float64)
    d = np.asarray(d, dtype=float64).reshape(m)
    Q = np.asarray(Q, dtype=float64)
    R = np.asarray(R, dtype=float64)
    y = np.asarray(y, dtype=float64)

    x_prev = np.asarray(x0, dtype=float64).reshape(n)
    P_prev = np.asarray(P0, dtype=float64).reshape(n, n)
    if symmetrize:
        P_prev = _sym(P_prev)

    BQBt = _sym(B @ Q @ B.T)

    hist_T = T if store_history else 0
    shock_T = T if (return_shocks and store_history) else 0

    x_pred_h = np.zeros((hist_T, n), dtype=float64)
    x_filt_h = np.zeros((hist_T, n), dtype=float64)
    P_pred_h = np.zeros((hist_T, n, n), dtype=float64)
    P_filt_h = np.zeros((hist_T, n, n), dtype=float64)
    y_pred_h = np.zeros((hist_T, m), dtype=float64)
    y_filt_h = np.zeros((hist_T, m), dtype=float64)
    innov_h = np.zeros((hist_T, m), dtype=float64)
    std_innov_h = np.zeros((hist_T, m), dtype=float64)
    S_h = np.zeros((hist_T, m, m), dtype=float64)
    eps_hat_h = np.zeros((shock_T, k), dtype=float64)

    loglik = 0.0
    const = m * np.log(2.0 * np.pi)
    eye_n = np.eye(n, dtype=float64)

    for t in range(T):
        x_pred = A @ x_prev
        P_pred = A @ P_prev @ A.T + BQBt
        if symmetrize:
            P_pred = _sym(P_pred)

        y_pred = C @ x_pred + d
        v = y[t] - y_pred
        S = C @ P_pred @ C.T + R
        if symmetrize:
            S = _sym(S)

        try:
            L = cholesky(S)
        except LinAlgError:
            L = cholesky(S + jitter * np.eye(m, dtype=float64))

        u = solve_triangular(L, v, lower=True)
        S_inv_v = solve_triangular(L.T, u, lower=False)

        PCt = P_pred @ C.T
        gain_u = solve_triangular(L, PCt.T, lower=True)
        K = solve_triangular(L.T, gain_u, lower=False).T

        x_filt = x_pred + K @ v
        I_minus_KC = eye_n - K @ C
        P_filt = I_minus_KC @ P_pred @ I_minus_KC.T + K @ R @ K.T
        if symmetrize:
            P_filt = _sym(P_filt)

        logdet_S = 2.0 * np.sum(np.log(np.diag(L)))
        loglik += -0.5 * (const + logdet_S + v @ S_inv_v)

        if return_shocks and store_history:
            M = Q @ B.T @ C.T
            eps_hat_h[t] = M @ S_inv_v

        if store_history:
            x_pred_h[t] = x_pred
            x_filt_h[t] = x_filt
            P_pred_h[t] = P_pred
            P_filt_h[t] = P_filt
            y_pred_h[t] = y_pred
            y_filt_h[t] = C @ x_filt + d
            innov_h[t] = v
            std_innov_h[t] = u
            S_h[t] = S

        x_prev = x_filt
        P_prev = P_filt

    return (
        x_pred_h,
        x_filt_h,
        P_pred_h,
        P_filt_h,
        y_pred_h,
        y_filt_h,
        innov_h,
        std_innov_h,
        S_h,
        eps_hat_h,
        float64(loglik),
    )
