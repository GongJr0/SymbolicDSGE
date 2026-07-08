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

# Prefer the compiled native linear hot loop; fall back to the numba kernel
# below when the extension is not built. The numba version stays as the fallback
# and the parity oracle. ALWAYS_USE_NUMBA / NEVER_USE_NUMBA override this default
# (see _native_dispatch).
from .._native_dispatch import FORCE_NUMBA, REQUIRE_NATIVE

# Declared explicitly so the FORCE_NUMBA branch (which binds None first) doesn't
# pin the inferred type to None; the native handle matches the _kalman.pyi stub
# and the numba _kalman_hot_loop return shape.
_KalmanReturn = Tuple[
    int,
    Tuple[float64, float64, float64],
    Tuple[NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, float64],
]
_kalman_hot_loop_native: Callable[..., _KalmanReturn] | None
_UKFReturn = Tuple[
    int,
    Tuple[float64, float64, float64],
    Tuple[NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, float64],
]
_ukf_hot_loop_native: Callable[..., _UKFReturn] | None

if FORCE_NUMBA:
    _kalman_hot_loop_native = None
    _ukf_hot_loop_native = None
else:
    try:
        from .._ckernels.kalman import kalman_hot_loop as _kalman_hot_loop_native
    except ImportError:  # pragma: no cover - exercised only without the extension
        if REQUIRE_NATIVE:
            raise
        _kalman_hot_loop_native = None
    try:
        from .._ckernels.kalman import ukf_hot_loop as _ukf_hot_loop_native
    except ImportError:  # pragma: no cover - exercised only without the extension
        if REQUIRE_NATIVE:
            raise
        _ukf_hot_loop_native = None


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class UnscentedFilterResult:
    x1_pred: NDF
    x2_pred: NDF
    x1_filt: NDF
    x2_filt: NDF

    P_pred: NDF
    P_filt: NDF

    y_pred: NDF
    y_filt: NDF

    innov: NDF
    std_innov: NDF
    S: NDF

    loglik: float64


def _get_real(mat: NDC | NDF, name: str, tol: float = 1e8) -> NDF:
    """
    Convert a complex matrix to a real matrix if the imaginary parts are negligible.
    """
    res = real_if_close(mat, tol=tol)
    if np.iscomplexobj(res):
        if res.size == 0:
            return np.ascontiguousarray(res.real, dtype=float64)
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


@njit(cache=True)
def _sym(P: NDF) -> NDF:
    return 0.5 * (P + P.T)


@njit(cache=True)
def _chol_shifted(S: NDF, jit: float = 0.0) -> NDF:
    n = S.shape[0]
    if jit > 0.0:
        return linalg.cholesky(S + jit * eye(n, dtype=float64)).astype(float64)
    return linalg.cholesky(S).astype(float64)


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def _logdet_from_chol(L: NDF) -> float64:
    s = float64(0.0)
    n = L.shape[0]
    for i in range(n):
        s += np.log(L[i, i])
    return float64(2.0) * s


@njit(cache=True)
def _zero_mat_into(out: NDF) -> None:
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = 0.0


@njit(cache=True)
def _sym_inplace(P: NDF) -> None:
    n = P.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            avg = 0.5 * (P[i, j] + P[j, i])
            P[i, j] = avg
            P[j, i] = avg


@njit(cache=True)
def _matmul_into(A: NDF, B: NDF, out: NDF) -> None:
    n, p = A.shape
    m = B.shape[1]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += A[i, k] * B[k, j]
            out[i, j] = s


@njit(cache=True)
def _matmul_abt_plus_c_into(A: NDF, B: NDF, C: NDF, out: NDF) -> None:
    n, p = A.shape
    m = B.shape[0]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += A[i, k] * B[j, k]
            out[i, j] = s + C[i, j]


@njit(cache=True)
def _matmul_abt_into(A: NDF, B: NDF, out: NDF) -> None:
    n, p = A.shape
    m = B.shape[0]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += A[i, k] * B[j, k]
            out[i, j] = s


@njit(cache=True)
def _matvec_into(A: NDF, x: NDF, out: NDF) -> None:
    n, m = A.shape
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += A[i, j] * x[j]
        out[i] = s


@njit(cache=True)
def _matvec_plus_vec_into(A: NDF, x: NDF, b: NDF, out: NDF) -> None:
    n, m = A.shape
    for i in range(n):
        s = b[i]
        for j in range(m):
            s += A[i, j] * x[j]
        out[i] = s


@njit(cache=True)
def _row_minus_vec_into(A: NDF, row: int, x: NDF, out: NDF) -> None:
    for j in range(x.shape[0]):
        out[j] = A[row, j] - x[j]


@njit(cache=True)
def _dot_vec(a: NDF, b: NDF) -> float64:
    s = float64(0.0)
    for i in range(a.shape[0]):
        s += a[i] * b[i]
    return s


@njit(cache=True)
def _chol_shifted_into(S: NDF, jit: float, L: NDF) -> None:
    n = S.shape[0]
    _zero_mat_into(L)
    for i in range(n):
        for j in range(i + 1):
            s = S[i, j]
            if i == j and jit > 0.0:
                s += jit
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 0.0:
                    raise linalg.LinAlgError("Matrix is not positive definite.")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]


@njit(cache=True)
def _forward_subst_vec_into(L: NDF, b: NDF, out: NDF) -> None:
    n = L.shape[0]
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * out[j]
        out[i] = (b[i] - s) / L[i, i]


@njit(cache=True)
def _backward_subst_chol_t_vec_into(L: NDF, b: NDF, out: NDF) -> None:
    n = L.shape[0]
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += L[j, i] * out[j]
        out[i] = (b[i] - s) / L[i, i]


@njit(cache=True)
def _chol_solve_row_into(
    L: NDF,
    B: NDF,
    row: int,
    forward_buf: NDF,
    backward_buf: NDF,
    out: NDF,
) -> None:
    n = L.shape[0]
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * forward_buf[j]
        forward_buf[i] = (B[row, i] - s) / L[i, i]

    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += L[j, i] * backward_buf[j]
        backward_buf[i] = (forward_buf[i] - s) / L[i, i]

    for i in range(n):
        out[row, i] = backward_buf[i]


@njit(cache=True)
def _predict_cov_into(
    A: NDF,
    P_prev: NDF,
    BQBT: NDF,
    temp_nn: NDF,
    out: NDF,
) -> None:
    _matmul_into(A, P_prev, temp_nn)
    _matmul_abt_plus_c_into(temp_nn, A, BQBT, out)


@njit(cache=True)
def _measurement_cov_into(
    C: NDF,
    P_pred: NDF,
    R: NDF,
    temp_mn: NDF,
    out: NDF,
) -> None:
    _matmul_into(C, P_pred, temp_mn)
    _matmul_abt_plus_c_into(temp_mn, C, R, out)


@njit(cache=True)
def _pc_t_into(P_pred: NDF, C: NDF, out: NDF) -> None:
    n = P_pred.shape[0]
    m = C.shape[0]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(n):
                s += P_pred[i, k] * C[j, k]
            out[i, j] = s


@njit(cache=True)
def _gain_from_pc_t_into(
    L: NDF,
    PCt: NDF,
    forward_buf: NDF,
    backward_buf: NDF,
    out: NDF,
) -> None:
    for row in range(PCt.shape[0]):
        _chol_solve_row_into(L, PCt, row, forward_buf, backward_buf, out)


@njit(cache=True)
def _state_update_into(x_pred: NDF, K: NDF, v: NDF, out: NDF) -> None:
    n, m = K.shape
    for i in range(n):
        s = x_pred[i]
        for j in range(m):
            s += K[i, j] * v[j]
        out[i] = s


@njit(cache=True)
def _identity_minus_into(A: NDF, out: NDF) -> None:
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            out[i, j] = -A[i, j]
        out[i, i] += 1.0


@njit(cache=True)
def _joseph_cov_into(
    K: NDF,
    C: NDF,
    P_pred: NDF,
    R: NDF,
    KC: NDF,
    I_minus_KC: NDF,
    temp_nn: NDF,
    temp_nm: NDF,
    out: NDF,
) -> None:
    _matmul_into(K, C, KC)
    _identity_minus_into(KC, I_minus_KC)
    _matmul_into(I_minus_KC, P_pred, temp_nn)
    _matmul_abt_into(temp_nn, I_minus_KC, out)
    _matmul_into(K, R, temp_nm)
    n = K.shape[0]
    m = K.shape[1]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(m):
                s += temp_nm[i, k] * K[j, k]
            out[i, j] += s


@njit(cache=True)
def _build_bqbt_into(B: NDF, Q: NDF, temp_nk: NDF, out: NDF) -> None:
    _matmul_into(B, Q, temp_nk)
    n = B.shape[0]
    k_dim = B.shape[1]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(k_dim):
                s += temp_nk[i, k] * B[j, k]
            out[i, j] = s
    _sym_inplace(out)


@njit(cache=True)
def _build_shock_projection_into(
    B: NDF, C: NDF, Q: NDF, temp_km: NDF, out: NDF
) -> None:
    n = B.shape[0]
    k_dim = B.shape[1]
    m = C.shape[0]
    for i in range(k_dim):
        for j in range(m):
            s = 0.0
            for l in range(n):
                s += B[l, i] * C[j, l]
            temp_km[i, j] = s
    _matmul_into(Q, temp_km, out)


@njit(cache=True)
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
    return_shocks: bool = False,
    _store_history: bool = True,
) -> tuple[
    int,
    tuple[float64, float64, float64],
    tuple[NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, float64],
]:
    n, m, k = nmk

    x_prev = x_0
    P_prev = P_0

    history_T = T if _store_history else 0
    shock_history_T = T if (return_shocks and _store_history) else 0

    x_pred = zeros((history_T, n), dtype=float64)
    x_filt = zeros((history_T, n), dtype=float64)

    P_pred = zeros((history_T, n, n), dtype=float64)
    P_filt = zeros((history_T, n, n), dtype=float64)

    y_pred = zeros((history_T, m), dtype=float64)
    y_filt = zeros((history_T, m), dtype=float64)

    v = zeros((history_T, m), dtype=float64)
    u = zeros((history_T, m), dtype=float64)
    S_hist = zeros((history_T, m, m), dtype=float64)

    eps_hat = zeros((shock_history_T, k), dtype=float64)

    x_pred_buf = zeros((n,), dtype=float64)
    x_filt_buf = zeros((n,), dtype=float64)
    y_pred_buf = zeros((m,), dtype=float64)
    y_filt_buf = zeros((m,), dtype=float64)
    v_buf = zeros((m,), dtype=float64)
    u_buf = zeros((m,), dtype=float64)
    S_inv_v = zeros((m,), dtype=float64)

    P_pred_buf = zeros((n, n), dtype=float64)
    P_filt_buf = zeros((n, n), dtype=float64)
    S_buf = zeros((m, m), dtype=float64)
    L = zeros((m, m), dtype=float64)
    PCt = zeros((n, m), dtype=float64)
    K = zeros((n, m), dtype=float64)
    KC = zeros((n, n), dtype=float64)
    I_minus_KC = zeros((n, n), dtype=float64)

    temp_nn = zeros((n, n), dtype=float64)
    temp_nm = zeros((n, m), dtype=float64)
    temp_mn = zeros((m, n), dtype=float64)
    solve_forward = zeros((m,), dtype=float64)
    solve_backward = zeros((m,), dtype=float64)

    BQBT = zeros((n, n), dtype=float64)
    temp_nk = zeros((n, k), dtype=float64)
    _build_bqbt_into(B, Q, temp_nk, BQBT)

    M = zeros((k, m), dtype=float64)
    if return_shocks and _store_history:
        temp_km = zeros((k, m), dtype=float64)
        _build_shock_projection_into(B, C, Q, temp_km, M)

    loglik = float64(0.0)
    const = m * np.log(2.0 * np.pi)

    for t in range(T):
        _matvec_into(A, x_prev, x_pred_buf)
        _predict_cov_into(A, P_prev, BQBT, temp_nn, P_pred_buf)

        if symmetrize:
            _sym_inplace(P_pred_buf)

        _matvec_plus_vec_into(C, x_pred_buf, d, y_pred_buf)
        _row_minus_vec_into(y, t, y_pred_buf, v_buf)
        _measurement_cov_into(C, P_pred_buf, R, temp_mn, S_buf)

        if symmetrize:
            _sym_inplace(S_buf)

        _chol_shifted_into(S_buf, jitter, L)
        _forward_subst_vec_into(L, v_buf, u_buf)
        _backward_subst_chol_t_vec_into(L, u_buf, S_inv_v)

        _pc_t_into(P_pred_buf, C, PCt)
        _gain_from_pc_t_into(L, PCt, solve_forward, solve_backward, K)

        _state_update_into(x_pred_buf, K, v_buf, x_filt_buf)

        _joseph_cov_into(
            K,
            C,
            P_pred_buf,
            R,
            KC,
            I_minus_KC,
            temp_nn,
            temp_nm,
            P_filt_buf,
        )

        if symmetrize:
            _sym_inplace(P_filt_buf)

        ldS = _logdet_from_chol(L)
        quad = _dot_vec(v_buf, S_inv_v)
        loglik += -0.5 * (const + ldS + quad)

        if return_shocks and _store_history:
            _matvec_into(M, S_inv_v, eps_hat[t])

        if _store_history:
            _matvec_plus_vec_into(C, x_filt_buf, d, y_filt_buf)

            for i in range(n):
                x_pred[t, i] = x_pred_buf[i]
                x_filt[t, i] = x_filt_buf[i]
            for i in range(n):
                for j in range(n):
                    P_pred[t, i, j] = P_pred_buf[i, j]
                    P_filt[t, i, j] = P_filt_buf[i, j]
            for i in range(m):
                y_pred[t, i] = y_pred_buf[i]
                y_filt[t, i] = y_filt_buf[i]
                v[t, i] = v_buf[i]
                u[t, i] = u_buf[i]
            for i in range(m):
                for j in range(m):
                    S_hist[t, i, j] = S_buf[i, j]

        x_prev = x_filt_buf
        P_prev = P_filt_buf

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
            u,
            S_hist,
            eps_hat,
            loglik,
        ),
    )


def _is_numba_array_dispatch(func: Callable[..., object]) -> bool:
    return bool(getattr(func, "_symbolicdsge_array_dispatch", False))


def _call_extended_measurement(
    func: Callable[..., NDF],
    state: NDF,
    calib_params: NDF,
) -> NDF:
    if _is_numba_array_dispatch(func):
        return asarray(func(state, calib_params), dtype=float64)
    return asarray(func(*state, *calib_params), dtype=float64)


def _call_extended_jacobian(
    func: Callable[..., NDF],
    state: NDF,
    calib_params: NDF,
) -> NDF:
    if _is_numba_array_dispatch(func):
        return asarray(func(state, calib_params), dtype=float64)
    return asarray(func(*state, *calib_params), dtype=float64)


def _ekf_hot_loop_python(
    T: int,
    nmk: Tuple[int, int, int],
    A: NDF,
    B: NDF,
    h: Callable[..., NDF],
    H_jac: Callable[..., NDF],
    calib_params: NDF,
    Q: NDF,
    R: NDF,
    y: NDF,
    x_0: NDF,
    P_0: NDF,
    symmetrize: bool,
    jitter: float,
    compute_y_filt: bool,
    return_shocks: bool = False,
    _store_history: bool = True,
) -> tuple[
    int,
    tuple[float64, float64, float64],
    tuple[NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, float64],
]:

    n, m, k = nmk
    x_prev = x_0
    P_prev = P_0

    # Outputs
    history_T = T if _store_history else 0
    shock_history_T = T if (return_shocks and _store_history) else 0

    x_pred = zeros((history_T, n), dtype=float64)
    x_filt = zeros((history_T, n), dtype=float64)

    P_pred = zeros((history_T, n, n), dtype=float64)
    P_filt = zeros((history_T, n, n), dtype=float64)

    y_pred = zeros((history_T, m), dtype=float64)
    y_filt = zeros((history_T, m), dtype=float64)

    v = zeros((history_T, m), dtype=float64)
    u = zeros((history_T, m), dtype=float64)
    S = zeros((history_T, m, m), dtype=float64)

    eps_hat = zeros((shock_history_T, k), dtype=float64)

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
        y_t_pred = _call_extended_measurement(h, x_t_pred, calib_params).reshape(m)
        H_t = _call_extended_jacobian(H_jac, x_t_pred, calib_params).reshape(m, n)
        H_tT = np.ascontiguousarray(H_t.T)

        v_t = y[t] - y_t_pred
        S_t = H_t @ P_t_pred @ H_tT + R

        if symmetrize:
            S_t = _sym(S_t)

        # --- Gain/update (swap C -> H_t) ---
        L = _chol_shifted(S_t, jitter)
        u_t = _forward_subst_vec(L, v_t)
        S_inv_v = _backward_subst_vec(L.T, u_t)

        PHt = P_t_pred @ H_tT  # (n, m)
        K_t = _chol_solve_mat(L, PHt.T).T  # (n, m)

        x_t_filt = x_t_pred + K_t @ v_t

        KH = K_t @ H_t
        P_t_filt = (In - KH) @ P_t_pred @ (In - KH).T + K_t @ R @ K_t.T
        if symmetrize:
            P_t_filt = _sym(P_t_filt)

        # Log-likelihood
        ldS = _logdet_from_chol(L)
        quad = float64(v_t @ S_inv_v)
        loglik += -0.5 * (const + ldS + quad)

        if compute_y_filt and _store_history:
            y_filt[t] = _call_extended_measurement(h, x_t_filt, calib_params).reshape(m)

        # Optional "shock estimate" (same form as linear KF)
        if return_shocks and _store_history:
            M = Q @ (BT @ H_tT)  # mirrors linear case with C -> H_t
            eps_hat[t] = M @ S_inv_v

        # Store
        if _store_history:
            x_pred[t] = x_t_pred
            x_filt[t] = x_t_filt
            P_pred[t] = P_t_pred
            P_filt[t] = P_t_filt
            y_pred[t] = y_t_pred
            v[t] = v_t
            u[t] = u_t
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
            u,
            S,
            eps_hat,
            loglik,
        ),
    )


@njit
def _ekf_hot_loop_numba(
    T: int,
    nmk: Tuple[int, int, int],
    A: NDF,
    B: NDF,
    h: Callable[[NDF, NDF], NDF],
    H_jac: Callable[[NDF, NDF], NDF],
    calib_params: NDF,
    Q: NDF,
    R: NDF,
    y: NDF,
    x_0: NDF,
    P_0: NDF,
    symmetrize: bool,
    jitter: float,
    compute_y_filt: bool,
    return_shocks: bool = False,
    _store_history: bool = True,
) -> tuple[
    int,
    tuple[float64, float64, float64],
    tuple[NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, NDF, float64],
]:

    n, m, k = nmk
    x_prev = x_0
    P_prev = P_0

    history_T = T if _store_history else 0
    shock_history_T = T if (return_shocks and _store_history) else 0

    x_pred = zeros((history_T, n), dtype=float64)
    x_filt = zeros((history_T, n), dtype=float64)

    P_pred = zeros((history_T, n, n), dtype=float64)
    P_filt = zeros((history_T, n, n), dtype=float64)

    y_pred = zeros((history_T, m), dtype=float64)
    y_filt = zeros((history_T, m), dtype=float64)

    v = zeros((history_T, m), dtype=float64)
    u = zeros((history_T, m), dtype=float64)
    S = zeros((history_T, m, m), dtype=float64)

    eps_hat = zeros((shock_history_T, k), dtype=float64)

    x_pred_buf = zeros((n,), dtype=float64)
    x_filt_buf = zeros((n,), dtype=float64)
    y_pred_buf = zeros((m,), dtype=float64)
    v_buf = zeros((m,), dtype=float64)
    u_buf = zeros((m,), dtype=float64)
    S_inv_v = zeros((m,), dtype=float64)

    P_pred_buf = zeros((n, n), dtype=float64)
    P_filt_buf = zeros((n, n), dtype=float64)
    H_buf = zeros((m, n), dtype=float64)
    S_buf = zeros((m, m), dtype=float64)
    L = zeros((m, m), dtype=float64)
    PHt = zeros((n, m), dtype=float64)
    K = zeros((n, m), dtype=float64)
    KH = zeros((n, n), dtype=float64)
    I_minus_KH = zeros((n, n), dtype=float64)

    temp_nn = zeros((n, n), dtype=float64)
    temp_nm = zeros((n, m), dtype=float64)
    temp_mn = zeros((m, n), dtype=float64)
    solve_forward = zeros((m,), dtype=float64)
    solve_backward = zeros((m,), dtype=float64)

    BQBT = zeros((n, n), dtype=float64)
    temp_nk = zeros((n, k), dtype=float64)
    _build_bqbt_into(B, Q, temp_nk, BQBT)

    M = zeros((k, m), dtype=float64)
    temp_km = zeros((k, m), dtype=float64)

    loglik = float64(0.0)
    const = m * np.log(2 * np.pi)

    for t in range(T):
        _matvec_into(A, x_prev, x_pred_buf)
        _predict_cov_into(A, P_prev, BQBT, temp_nn, P_pred_buf)

        if symmetrize:
            _sym_inplace(P_pred_buf)

        y_t_pred = h(x_pred_buf, calib_params).reshape(m)
        H_t = H_jac(x_pred_buf, calib_params).reshape(m, n)
        for i in range(m):
            y_pred_buf[i] = y_t_pred[i]
            for j in range(n):
                H_buf[i, j] = H_t[i, j]

        _row_minus_vec_into(y, t, y_pred_buf, v_buf)
        _measurement_cov_into(H_buf, P_pred_buf, R, temp_mn, S_buf)

        if symmetrize:
            _sym_inplace(S_buf)

        _chol_shifted_into(S_buf, jitter, L)
        _forward_subst_vec_into(L, v_buf, u_buf)
        _backward_subst_chol_t_vec_into(L, u_buf, S_inv_v)

        _pc_t_into(P_pred_buf, H_buf, PHt)
        _gain_from_pc_t_into(L, PHt, solve_forward, solve_backward, K)

        _state_update_into(x_pred_buf, K, v_buf, x_filt_buf)

        _joseph_cov_into(
            K,
            H_buf,
            P_pred_buf,
            R,
            KH,
            I_minus_KH,
            temp_nn,
            temp_nm,
            P_filt_buf,
        )
        if symmetrize:
            _sym_inplace(P_filt_buf)

        ldS = _logdet_from_chol(L)
        quad = _dot_vec(v_buf, S_inv_v)
        loglik += -0.5 * (const + ldS + quad)

        if compute_y_filt and _store_history:
            y_t_filt = h(x_filt_buf, calib_params).reshape(m)
            for i in range(m):
                y_filt[t, i] = y_t_filt[i]
        if return_shocks and _store_history:
            _build_shock_projection_into(B, H_buf, Q, temp_km, M)
            _matvec_into(M, S_inv_v, eps_hat[t])

        if _store_history:
            for i in range(n):
                x_pred[t, i] = x_pred_buf[i]
                x_filt[t, i] = x_filt_buf[i]
            for i in range(n):
                for j in range(n):
                    P_pred[t, i, j] = P_pred_buf[i, j]
                    P_filt[t, i, j] = P_filt_buf[i, j]
            for i in range(m):
                y_pred[t, i] = y_pred_buf[i]
                v[t, i] = v_buf[i]
                u[t, i] = u_buf[i]
            for i in range(m):
                for j in range(m):
                    S[t, i, j] = S_buf[i, j]

        x_prev = x_filt_buf
        P_prev = P_filt_buf

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
            u,
            S,
            eps_hat,
            loglik,
        ),
    )


# Static & Parametrized Kalman Filter (written to act with SolvedModel object attributes)
class KalmanFilter:
    _get_real = staticmethod(_get_real)
    _shape_validate = staticmethod(_shape_validate)

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

        hot_loop = (
            _kalman_hot_loop_native
            if _kalman_hot_loop_native is not None
            else _kalman_hot_loop
        )
        try:
            err, err_info, out = hot_loop(
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
        if _ukf_hot_loop_native is None:
            raise RuntimeError("Native unscented Kalman filter is not available.")
        if meas_addr == 0:
            raise ValueError("meas_addr must be a nonzero measurement cfunc address.")

        hx = _get_real(hx, "hx")
        bx = _get_real(bx, "bx")
        hxx = _get_real(hxx, "hxx")
        hss = _get_real(hss, "hss").reshape(-1)
        steady_state = _get_real(steady_state, "steady_state").reshape(-1)
        calib_params = _get_real(calib_params, "calib_params").reshape(-1)
        Q = _get_real(Q, "Q")
        R = _get_real(R, "R")
        y = _get_real(y, "y")
        z0 = _get_real(z0, "z0").reshape(-1)
        P0 = _get_real(P0, "P0")

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
            Q = _sym(Q)
            R = _sym(R)
            P0 = _sym(P0)

        try:
            err, err_info, out = _ukf_hot_loop_native(
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
        except linalg.LinAlgError as exc:
            raise MatrixConditionError(float("inf")) from exc
        if err != 0:
            ErrorConstructor = get_error_constructor(ErrorCode(err))
            raise ErrorConstructor(*err_info)

        (
            x1_pred,
            x2_pred,
            x1_filt,
            x2_filt,
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
            x1_pred=x1_pred,
            x2_pred=x2_pred,
            x1_filt=x1_filt,
            x2_filt=x2_filt,
            P_pred=P_pred,
            P_filt=P_filt,
            y_pred=y_pred,
            y_filt=y_filt,
            innov=v,
            std_innov=u,
            S=S,
            loglik=loglik,
        )

    @staticmethod
    def run_extended(
        A: NDF | NDC,
        B: NDF | NDC,
        h: Callable[..., NDF],
        H_jac: Callable[..., NDF],
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

        :param compute_y_filt: If True, compute y_filt[t] = h(x_filt[t], t). If False, leave y_filt as zeros with shape (T, m).
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

        use_numba_path = _is_numba_array_dispatch(h) and _is_numba_array_dispatch(H_jac)

        try:
            if use_numba_path:
                err, err_info, out = _ekf_hot_loop_numba(
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
                    return_shocks,
                    _store_history,
                )
            else:
                err, err_info, out = _ekf_hot_loop_python(
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
                    return_shocks,
                    _store_history,
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
        )
