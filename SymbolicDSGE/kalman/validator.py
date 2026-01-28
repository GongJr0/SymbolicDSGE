import numpy as np
from numpy.typing import NDArray
from typing import Optional
from dataclasses import dataclass

NDF = NDArray[np.float64]


@dataclass(frozen=True)
class _KalmanDebugInfo:
    A: NDF
    B: NDF
    C: NDF
    d: NDF
    Q: NDF
    R: NDF
    y: NDF
    x0: Optional[NDF]
    P0: Optional[NDF]


@dataclass(frozen=True)
class KFValidationContext:
    """Small container for clearer error messages."""

    n_state: int
    n_obs: int
    n_shock: int
    T: int


def validate_kf_inputs(
    *,
    A: NDF,
    B: NDF,
    C: NDF,
    d: NDF,
    Q: NDF,
    R: NDF,
    y: NDF,
    x0: Optional[NDF] = None,
    P0: Optional[NDF] = None,
    check_symmetry: bool = True,
    check_nonneg_diag: bool = True,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> KFValidationContext:
    """
    Validate shapes and basic covariance sanity for Kalman filter inputs.

    Uses explicit raises (not asserts) so it is safe under Python -O.

    Intended to be called right before KalmanFilter.run(...).

    Notes:
    - Does NOT check y for NaNs/ndim mismatch if you already validate elsewhere,
      but it *does* re-check the key dimensional compatibility with C/R.
    - Does NOT check PSD via eig/Cholesky (you may rely on jitter/symmetrize downstream).
    """
    # ---------- Basic dtype/ndim checks ----------
    for name, M, nd in [
        ("A", A, 2),
        ("B", B, 2),
        ("C", C, 2),
        ("Q", Q, 2),
        ("R", R, 2),
        ("y", y, 2),
    ]:
        if not isinstance(M, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray, got {type(M).__name__}.")
        if M.ndim != nd:
            raise ValueError(f"{name} must be {nd}D, got shape {M.shape}.")

    if not isinstance(d, np.ndarray):
        raise TypeError(f"d must be a numpy ndarray, got {type(d).__name__}.")
    if d.ndim not in (1, 2):
        raise ValueError(f"d must be 1D or 2D, got shape {d.shape}.")

    # ---------- Infer core dimensions ----------
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError(f"A must be square (n x n). Got shape {A.shape}.")

    if B.shape[0] != n:
        raise ValueError(
            f"B must have n_state={n} rows to match A. Got shape {B.shape}."
        )
    k = B.shape[1]

    m = C.shape[0]
    if C.shape[1] != n:
        raise ValueError(
            f"C must be (n_obs x n_state)=({m} x {n}). Got shape {C.shape}."
        )

    # y compatibility
    T, m_y = y.shape
    if m_y != m:
        raise ValueError(
            f"y must have n_obs={m} columns to match C. Got y.shape={y.shape}."
        )

    # d shape compatibility
    if d.ndim == 1:
        if d.shape != (m,):
            raise ValueError(f"d must have shape ({m},), got {d.shape}.")
    else:  # 2D
        if d.shape not in ((m, 1), (1, m)):
            raise ValueError(
                f"d must have shape ({m},1) or (1,{m}) if 2D. Got {d.shape}."
            )

    # Q compatibility
    if Q.shape != (k, k):
        raise ValueError(
            f"Q must be (n_shock x n_shock)=({k} x {k}) to match B. Got {Q.shape}."
        )

    # R compatibility
    if R.shape != (m, m):
        raise ValueError(
            f"R must be (n_obs x n_obs)=({m} x {m}) to match C/y. Got {R.shape}."
        )

    # x0/P0 checks (optional)
    if x0 is not None:
        if not isinstance(x0, np.ndarray):
            raise TypeError(f"x0 must be a numpy ndarray, got {type(x0).__name__}.")
        if x0.ndim != 1 or x0.shape != (n,):
            raise ValueError(
                f"x0 must be 1D with shape (n_state,) = ({n},). Got {x0.shape}."
            )

    if P0 is not None:
        if not isinstance(P0, np.ndarray):
            raise TypeError(f"P0 must be a numpy ndarray, got {type(P0).__name__}.")
        if P0.ndim != 2 or P0.shape != (n, n):
            raise ValueError(
                f"P0 must be 2D with shape (n_state,n_state)=({n},{n}). Got {P0.shape}."
            )

    # ---------- Covariance sanity (optional) ----------
    if check_symmetry:
        if not np.allclose(Q, Q.T, rtol=rtol, atol=atol):
            raise ValueError("Q must be symmetric (within tolerance).")
        if not np.allclose(R, R.T, rtol=rtol, atol=atol):
            raise ValueError("R must be symmetric (within tolerance).")
        if P0 is not None and not np.allclose(P0, P0.T, rtol=rtol, atol=atol):
            raise ValueError("P0 must be symmetric (within tolerance).")

    if check_nonneg_diag:
        if np.any(np.diag(Q) < -atol):
            raise ValueError("Q must have non-negative diagonal entries (variances).")
        if np.any(np.diag(R) < -atol):
            raise ValueError("R must have non-negative diagonal entries (variances).")
        if P0 is not None and np.any(np.diag(P0) < -atol):
            raise ValueError("P0 must have non-negative diagonal entries (variances).")

    return KFValidationContext(n_state=n, n_obs=m, n_shock=k, T=T)
