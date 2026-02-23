import numpy as np
from numpy.typing import NDArray
from typing import Optional, Callable, Any, Literal
from dataclasses import dataclass
from enum import StrEnum

NDF = NDArray[np.float64]


class FilterMode(StrEnum):
    LINEAR = "linear"
    EXTENDED = "extended"


@dataclass(frozen=True)
class _KalmanDebugInfo:
    A: NDF
    B: NDF
    C: NDF | None
    d: NDF | None
    h_func: Callable[..., NDF] | None
    H_jac: Callable[..., NDF] | None
    Q: NDF
    R: NDF
    y: NDF | None
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
    filter_mode: "FilterMode",
    # --- shared / transition ---
    A: NDF,
    B: NDF,
    Q: NDF,
    R: NDF,
    y: NDF,
    x0: Optional[NDF] = None,
    P0: Optional[NDF] = None,
    # --- linear-only measurement ---
    C: Optional[NDF] = None,
    d: Optional[NDF] = None,
    # --- extended-only measurement ---
    h: Optional[Callable[..., Any]] = None,
    H_jac: Optional[Callable[..., Any]] = None,
    calib_params: Optional[NDF] = None,
    # --- general checks ---
    check_symmetry: bool = True,
    check_nonneg_diag: bool = True,
    check_finite: bool = True,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    # --- EKF probing (extended only) ---
    probe_measurement: bool = True,
    probe_state: Literal["x0", "zeros"] = "x0",
) -> KFValidationContext:
    """
    Validate shapes and basic covariance sanity for Kalman filter inputs.

    Supports both:
      - Linear KF:    x_t = A x_{t-1} + B eps_t,   y_t = C x_t + d + v_t
      - Extended KF:  x_t = A x_{t-1} + B eps_t,   y_t = h(x_t) + v_t

    Uses explicit raises (not asserts) so it is safe under Python -O.

    Notes:
    - Does NOT check PSD via eig/Cholesky (you may rely on jitter/symmetrize downstream).
    - If `filter_mode=EXTENDED` and `probe_measurement=True`, the function calls
      `h_func` and `H_jac` once at a probe state to validate output shapes and (optionally)
      finiteness.

    :param filter_mode: Which filter measurement model to validate (linear vs extended).
    :type filter_mode: FilterMode
    """
    # ---------- Basic dtype/ndim checks ----------
    for name, M, nd in [
        ("A", A, 2),
        ("B", B, 2),
        ("Q", Q, 2),
        ("R", R, 2),
        ("y", y, 2),
    ]:
        if not isinstance(M, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray, got {type(M).__name__}.")
        if M.ndim != nd:
            raise ValueError(f"{name} must be {nd}D, got shape {M.shape}.")

    # ---------- Infer core dimensions ----------
    n = int(A.shape[0])
    if A.shape[1] != n:
        raise ValueError(f"A must be square (n x n). Got shape {A.shape}.")

    if B.shape[0] != n:
        raise ValueError(
            f"B must have n_state={n} rows to match A. Got shape {B.shape}."
        )
    k = int(B.shape[1])

    T, m_y = y.shape

    # ---------- Linear vs Extended measurement checks ----------
    if filter_mode == FilterMode.LINEAR:
        if C is None or d is None:
            raise ValueError("Linear mode requires C and d.")

        if not isinstance(C, np.ndarray):
            raise TypeError(f"C must be a numpy ndarray, got {type(C).__name__}.")
        if C.ndim != 2:
            raise ValueError(f"C must be 2D, got shape {C.shape}.")

        if not isinstance(d, np.ndarray):
            raise TypeError(f"d must be a numpy ndarray, got {type(d).__name__}.")
        if d.ndim not in (1, 2):
            raise ValueError(f"d must be 1D or 2D, got shape {d.shape}.")

        m = int(C.shape[0])
        if C.shape[1] != n:
            raise ValueError(
                f"C must be (n_obs x n_state)=({m} x {n}). Got shape {C.shape}."
            )

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

    elif filter_mode == FilterMode.EXTENDED:
        if h is None or H_jac is None:
            raise ValueError("Extended mode requires h and H.")
        if not callable(h):
            raise TypeError(f"h must be callable, got {type(h).__name__}.")
        if not callable(H_jac):
            raise TypeError(f"H_jac must be callable, got {type(H_jac).__name__}.")

        # In extended mode, m is inferred from y (and must match R)
        m = int(m_y)

        if probe_measurement:
            if probe_state == "x0" and x0 is not None:
                x_probe = x0
            else:
                x_probe = np.zeros((n,), dtype=np.float64)

            if calib_params is None:
                raise ValueError(
                    "calib_params must be provided for probing in extended mode."
                )

            y_hat = h(*x_probe, *calib_params)
            H_hat = H_jac(*x_probe, *calib_params)

            y_hat_arr = np.asarray(y_hat)
            H_hat_arr = np.asarray(H_hat)

            # Accept scalar only if m==1
            if y_hat_arr.ndim == 0:
                if m != 1:
                    raise ValueError(
                        f"h_func returned a scalar, but y has m={m} columns. "
                        f"Expected shape ({m},)."
                    )
            elif y_hat_arr.ndim == 1:
                if y_hat_arr.shape != (m,):
                    raise ValueError(
                        f"h_func must return shape ({m},), got {y_hat_arr.shape}."
                    )
            elif y_hat_arr.ndim == 2:
                if y_hat_arr.shape not in ((m, 1), (1, m)):
                    raise ValueError(
                        f"h_func must return shape ({m},) or ({m},1)/(1,{m}); got {y_hat_arr.shape}."
                    )
            else:
                raise ValueError(
                    f"h_func must return a vector-like output; got array with shape {y_hat_arr.shape}."
                )

            if H_hat_arr.ndim != 2 or H_hat_arr.shape != (m, n):
                raise ValueError(
                    f"H_jac must return shape (m,n)=({m},{n}), got {H_hat_arr.shape}."
                )

            if check_finite:
                if not np.isfinite(y_hat_arr.astype(np.float64, copy=False)).all():
                    raise ValueError(
                        "h_func produced non-finite values at probe state."
                    )
                if not np.isfinite(H_hat_arr.astype(np.float64, copy=False)).all():
                    raise ValueError("H_jac produced non-finite values at probe state.")

    else:
        raise ValueError(f"Unknown filter_mode: {filter_mode!r}")

    # ---------- Shared Q/R compatibility ----------
    if Q.shape != (k, k):
        raise ValueError(
            f"Q must be (n_shock x n_shock)=({k} x {k}) to match B. Got {Q.shape}."
        )

    # R compatibility (m depends on mode; set above in both branches)
    if filter_mode == FilterMode.LINEAR:
        m = int(C.shape[0])  # type: ignore[union-attr]
    else:
        m = int(m_y)

    if R.shape != (m, m):
        raise ValueError(
            f"R must be (n_obs x n_obs)=({m} x {m}) to match measurement dimension. Got {R.shape}."
        )

    # ---------- x0/P0 checks (optional) ----------
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
