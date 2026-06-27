import numpy as np
from numpy import float64
from numpy.typing import NDArray

from numba import njit

from typing import Literal, cast

NDF = NDArray[float64]


# Kernel IDs (integer). The HAC estimator selects a lag window by id rather than
# by a passed-in callable, so the jitted path can be ``cache=True`` and the whole
# branch maps directly onto a plain C ``switch`` for the native port.
_BARTLETT: int = 0
_PARZEN: int = 1
_QS: int = 2

_KERNEL_STR_TO_ID: dict[str, int] = {
    "bartlett": _BARTLETT,
    "parzen": _PARZEN,
    "qs": _QS,
}

# Andrews (1991) bandwidth constants for different kernels
_C_BARTLETT = 1.1447
_C_PARZEN = 2.6614
_C_QUADRATIC_SPECTRAL = 1.3221

_ANDREWS_C_Q_GETTER: dict[int, tuple[float, float]] = {
    _BARTLETT: (_C_BARTLETT, 1.0),
    _PARZEN: (_C_PARZEN, 2.0),
    _QS: (_C_QUADRATIC_SPECTRAL, 2.0),
}


# Wooldridge Textbook bandwoidth selection rule
def wooldridge_bandwidth(x: NDF) -> int:
    n = x.shape[0]
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


# Andrews (1991) bandwidth selection rule for HAC covariance estimation
def andrews_bandwidth(y: NDF, kernel_id: int = _BARTLETT) -> int:
    n = y.shape[0]
    if n < 2 or np.var(y) <= 1e-14:
        return 1

    y_lag = y[:-1]
    y_cur = y[1:]

    denom = np.dot(y_lag, y_lag)
    if denom <= 1e-14 or not np.isfinite(denom):
        return 1

    beta = np.dot(y_lag, y_cur) / denom
    if not np.isfinite(beta):
        return 1

    beta = np.clip(beta, -0.999, 0.999)  # Avoid blowup of Rhat

    Rhat = 2 * beta * (1 + beta) / (1 - beta) ** 2
    if Rhat <= 0.0 or not np.isfinite(Rhat):
        return 1

    c, q = _ANDREWS_C_Q_GETTER[kernel_id]
    b = c * (Rhat ** (1.0 / (2 * q + 1))) * (n ** (1.0 / (2 * q + 1)))
    return max(1, int(np.floor(b)))


def andrews_bandwidth_matrix(
    r: NDF,
    kernel_id: int = _BARTLETT,
) -> int:
    r = np.asarray(r, dtype=np.float64)

    if r.ndim == 1:
        return andrews_bandwidth(r, kernel_id=kernel_id)

    if r.ndim != 2:
        raise ValueError(f"r must be 1D or 2D, got shape {r.shape}.")

    Ls = []
    for j in range(r.shape[1]):
        col = r[:, j]
        if np.var(col) > 1e-14:
            Ls.append(andrews_bandwidth(col, kernel_id=kernel_id))

    if not Ls:
        return 1

    return int(np.median(np.asarray(Ls)))


@njit(cache=True)
def _kernel_weight(j: int, L: int, kernel_id: int) -> float64:
    """Lag-window weight w(j; L) for the selected kernel.

    A single branch over the integer kernel id (rather than a passed-in callable)
    so the jitted HAC estimator can be ``cache=True`` and the whole thing maps
    onto a plain C ``switch`` for the native port. All three closed forms are
    math.h-expressible.
    """
    x: float64 = float64(j) / (float64(L) + 1.0)

    if kernel_id == _BARTLETT:
        if j > L:
            return float64(0.0)
        return float64(1.0 - x)

    if kernel_id == _PARZEN:
        if x > 1.0:
            return float64(0.0)
        if x <= 0.5:
            return float64(1.0 - 6.0 * x**2 + 6.0 * x**3)
        return float64(2.0 * (1.0 - x) ** 3)

    # _QS
    if abs(x) <= 1e-8:
        return float64(1.0)
    outer = 25.0 / (12.0 * np.pi**2 * x**2)
    arg = 6.0 * np.pi * x / 5.0
    inner = np.sin(arg) / arg - np.cos(arg)
    return float64(outer * inner)


# HAC covariance estimation using the selected kernel and bandwidth. Now that the
# kernel is chosen by integer id (no jitted-callable argument), this can be
# ``cache=True``: the on-disk cache keys on the stable (array, int, int)
# signature instead of a dispatcher weakref that dies with the process.
@njit(cache=True)
def jit_hac_estimator_matmul(
    r: NDF,
    kernel_id: int,
    L: int,  # Bandwidth
) -> NDF:
    n = r.shape[0]
    p = r.shape[1]
    S = np.zeros((p, p), dtype=float64)
    L = min(L, n - 1)

    S += r.T @ r / n  # Gamma 0

    for j in range(1, L + 1):
        w_j = _kernel_weight(j, L, kernel_id)
        if w_j == 0.0:
            continue
        gamma_j = r[j:].T @ r[:-j] / n
        S += w_j * (gamma_j + gamma_j.T)  # Add symmetric contribution
    return S


# Only used when nopython is disabled on kernel dispatch.
# For testing and maintaining functionality in cases where numba isn't supported.
def py_hac_estimator(
    r: NDF,
    kernel_id: int,
    L: int,  # Bandwidth
) -> NDF:
    n, p = r.shape
    S = np.zeros((p, p), dtype=float64)
    L = min(L, n - 1)

    S += r.T @ r / n  # Gamma 0

    for j in range(1, L + 1):
        w_j = _kernel_weight.py_func(j, L, kernel_id)
        if w_j == 0.0:
            continue
        gamma_j = r[j:].T @ r[:-j] / n
        S += w_j * (gamma_j + gamma_j.T)  # Add symmetric contribution
    return S


def hac_covariance(
    r: NDF,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["wooldridge", "andrews", "auto"] | None = None,
    center: bool = False,
    nopython: bool = True,
) -> NDF:
    r = np.ascontiguousarray(r, dtype=np.float64)

    if r.ndim != 2:
        raise ValueError(f"r must be 2D with shape (n, p), got {r.shape}.")

    n = r.shape[0]
    if n < 2:
        raise ValueError(f"r must have at least 2 observations, got {n}.")

    kernel_id = _KERNEL_STR_TO_ID.get(kernel)
    if kernel_id is None:
        raise ValueError(
            f"Unsupported kernel: {kernel}. Supported kernels are: "
            f"{list(_KERNEL_STR_TO_ID)}"
        )

    if center:
        r = r - r.mean(axis=0)

    # Bandwidth resolution. An explicit non-negative int is used as L directly; a
    # string names a data-driven selection rule; None/"auto" picks the rule
    # conventionally paired with the kernel (Wooldridge for Bartlett, Andrews
    # otherwise). There is no integer id for the selection rules -- a bare int
    # always means a manual bandwidth.
    if bandwidth is None or bandwidth == "auto":
        if kernel == "bartlett":
            L = wooldridge_bandwidth(r)
        else:
            L = andrews_bandwidth_matrix(r, kernel_id=kernel_id)
    elif isinstance(bandwidth, str):
        if bandwidth == "wooldridge":
            L = wooldridge_bandwidth(r)
        elif bandwidth == "andrews":
            L = andrews_bandwidth_matrix(r, kernel_id=kernel_id)
        else:
            raise ValueError(f"Unsupported bandwidth selection method: {bandwidth}")
    elif isinstance(bandwidth, int):
        if bandwidth < 0:
            raise ValueError(f"Bandwidth must be non-negative, got {bandwidth}.")
        L = bandwidth
    else:
        raise ValueError(f"Invalid bandwidth specification: {bandwidth}")

    L = min(L, n - 1)

    if nopython:
        return cast(NDF, jit_hac_estimator_matmul(r, kernel_id, L))

    return py_hac_estimator(r, kernel_id, L)
