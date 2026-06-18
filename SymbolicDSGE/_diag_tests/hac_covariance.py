import numpy as np
from numpy import float64
from numpy.typing import NDArray

from numba import njit

from typing import Literal, Callable, cast

NDF = NDArray[float64]

# Andrews (1991) bandwidth constants for different kernels
_C_BARTLETT = 1.1447
_C_PARZEN = 2.6614
_C_QUADRATIC_SPECTRAL = 1.3221

_ANDREWS_C_Q_GETTER: dict[str, tuple[float, float]] = {
    "bartlett": (_C_BARTLETT, 1.0),
    "parzen": (_C_PARZEN, 2.0),
    "qs": (_C_QUADRATIC_SPECTRAL, 2.0),
}


# Wooldridge Textbook bandwoidth selection rule
def wooldridge_bandwidth(x: NDF) -> int:
    n = x.shape[0]
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


# Andrews (1991) bandwidth selection rule for HAC covariance estimation
def andrews_bandwidth(
    y: NDF, kernel: Literal["bartlett", "parzen", "qs"] = "bartlett"
) -> int:
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

    c, q = _ANDREWS_C_Q_GETTER[kernel]
    b = c * (Rhat ** (1.0 / (2 * q + 1))) * (n ** (1.0 / (2 * q + 1)))
    return max(1, int(np.floor(b)))


def andrews_bandwidth_matrix(
    r: NDF,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
) -> int:
    r = np.asarray(r, dtype=np.float64)

    if r.ndim == 1:
        return andrews_bandwidth(r, kernel=kernel)

    if r.ndim != 2:
        raise ValueError(f"r must be 1D or 2D, got shape {r.shape}.")

    Ls = []
    for j in range(r.shape[1]):
        col = r[:, j]
        if np.var(col) > 1e-14:
            Ls.append(andrews_bandwidth(col, kernel=kernel))

    if not Ls:
        return 1

    return int(np.median(np.asarray(Ls)))


_BW_SELECTION_DISPATCHER: dict[str, Callable] = {
    "wooldridge": wooldridge_bandwidth,
    "andrews": andrews_bandwidth_matrix,
}


def bartlett_kernel(j: int, L: int) -> float64:
    out: float64 = float64(0.0)
    if j > L:
        return out
    else:
        out += 1.0 - float64(j) / (float64(L) + 1.0)
        return out


def parzen_kernel(j: int, L: int) -> float64:
    out: float64 = float64(0.0)
    x: float64 = float64(j) / (float64(L) + 1.0)

    if x > 1.0:
        return out
    elif x <= 0.5:
        out += 1.0 - 6.0 * x**2 + 6.0 * x**3
        return out
    else:
        out += 2.0 * (1.0 - x) ** 3
        return out


def quadratic_spectral_kernel(j: int, L: int) -> float64:
    out: float64 = float64(0.0)
    x: float64 = float64(j) / (float64(L) + 1.0)

    if np.isclose(x, 0.0, atol=1e-8):
        out += 1.0
        return out
    else:
        outer = 25.0 / (12.0 * np.pi**2 * x**2)
        inner = np.sin(6.0 * np.pi * x / 5.0) / (6.0 * np.pi * x / 5.0) - np.cos(
            6.0 * np.pi * x / 5.0
        )
        out += outer * inner
        return out


_KERNEL_GETTER: dict[str, Callable[[int, int], float64]] = {
    "bartlett": bartlett_kernel,
    "parzen": parzen_kernel,
    "qs": quadratic_spectral_kernel,
}

_JIT_CACHE: dict[str, Callable[[int, int], float64]] = {}


def kernel_dispatcher(
    kernel: Literal["bartlett", "parzen", "qs"], nopython: bool = True
) -> Callable[[int, int], float64]:
    if nopython:
        if kernel not in _JIT_CACHE:
            _JIT_CACHE[kernel] = njit(_KERNEL_GETTER[kernel], cache=True)
        return _JIT_CACHE[kernel]
    else:
        return _KERNEL_GETTER[kernel]


# HAC covariance estimation using the specified kernel and bandwidth.
# NOTE: deliberately *not* cache=True. `k` is a jitted dispatcher argument, and
# numba keys an on-disk cache entry on the dispatcher's identity (a weakref), not
# a stable signature. Each session's kernel is a new object, so the cache never
# hits cross-session — it only accumulates stale keys whose weakrefs die with the
# process that made them; the next save re-pickles the index and raises
# "ReferenceError: underlying object has vanished". In-process JIT reuse is
# unaffected (the compiled overload is still memoized in memory).
@njit
def jit_hac_estimator_matmul(
    r: NDF,
    k: Callable[[int, int], float64],  # Kernel function
    L: int,  # Bandwidth
) -> NDF:
    n = r.shape[0]
    S = np.zeros((r.shape[1], r.shape[1]), dtype=float64)
    L = min(L, n - 1)

    S += r.T @ r / n  # Gamma 0

    for j in range(1, L + 1):
        w_j = k(j, L)
        if w_j == 0.0:
            continue
        gamma_j = r[j:].T @ r[:-j] / n
        S += w_j * (gamma_j + gamma_j.T)  # Add symmetric contribution
    return S


# Only used when nopython is disabled on kernel dispatch.
# For testing and maintaining functionality in cases where numba isn't supported.
def py_hac_estimator(
    r: NDF,
    k: Callable[[int, int], float64],  # Kernel function
    L: int,  # Bandwidth
) -> NDF:

    n, p = r.shape
    S = np.zeros((p, p), dtype=float64)
    L = min(L, n - 1)

    S += r.T @ r / n  # Gamma 0

    for j in range(1, L + 1):
        w_j = k(j, L)
        if w_j == 0.0:
            continue
        gamma_j = r[j:].T @ r[:-j] / n
        S += w_j * (gamma_j + gamma_j.T)  # Add symmetric contribution
    return S


def hac_covariance(
    r: NDF,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["wooldridge", "andrews", "auto"] | None = "auto",
    center: bool = False,
    nopython: bool = True,
) -> NDF:
    r = np.ascontiguousarray(r, dtype=np.float64)

    if r.ndim != 2:
        raise ValueError(f"r must be 2D with shape (n, p), got {r.shape}.")

    n = r.shape[0]
    if n < 2:
        raise ValueError(f"r must have at least 2 observations, got {n}.")

    if center:
        r = r - r.mean(axis=0)

    L = -1
    if bandwidth is None or bandwidth == "auto":
        if kernel == "bartlett":
            L = wooldridge_bandwidth(r)
        elif kernel in ("parzen", "qs"):
            L = andrews_bandwidth_matrix(r, kernel=kernel)
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")

    elif isinstance(bandwidth, str):
        if bandwidth not in _BW_SELECTION_DISPATCHER:
            raise ValueError(f"Unsupported bandwidth selection method: {bandwidth}")
        L = _BW_SELECTION_DISPATCHER[bandwidth](r)

    elif isinstance(bandwidth, int):
        if bandwidth < 0:
            raise ValueError(f"Bandwidth must be non-negative, got {bandwidth}.")
        L = bandwidth
    else:
        raise ValueError(f"Invalid bandwidth specification: {bandwidth}")

    L = min(L, n - 1)
    k = kernel_dispatcher(kernel, nopython=nopython)

    if nopython:
        return cast(NDF, jit_hac_estimator_matmul(r, k, L))

    return py_hac_estimator(r, k, L)
