"""Independent reference implementations for the native
``_ckernels.monte_carlo`` sample transforms, retained as parity oracles.

These recreate the numpy the pure-Python ops in
``monte_carlo/operations/transforms/ops.py`` carried before they were rewired to
call the C kernels. The native C is the production path; these exist only so the
parity tests can pin it, and must never import the library ops (that would
compare the kernels against themselves).

The rolling functions recompute every window from a ``sliding_window_view``,
where the kernels slide a Welford state by removing the leaving observation, so
the two agree to a tolerance rather than to the bit. That difference is the
point: it makes the oracle an independent check rather than a mirror.

Each function takes a plain ``(n, p)`` array, dropping the executor's
``context`` / ``reference`` / ``dgp`` / ``rep_idx`` keyword contract.
"""

from __future__ import annotations

import numpy as np
from numpy import float64
from numpy.typing import NDArray

NDF = NDArray[float64]


def standardize(x: NDF, ddof: int = 0) -> NDF:
    """Per-column z-score. Zero-variance columns come back as zeros."""
    arr = np.asarray(x, dtype=float64)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, ddof=ddof, keepdims=True)
    safe_std = np.where(std == 0.0, 1.0, std)
    out = (arr - mean) / safe_std
    out = np.where(std == 0.0, 0.0, out)
    return np.ascontiguousarray(out, dtype=float64)


def log(x: NDF, offset: float = 0.0) -> NDF:
    """``log(x + offset)`` elementwise."""
    return np.log(np.asarray(x, dtype=float64) + offset)


def log_diff(x: NDF, offset: float = 0.0) -> NDF:
    """One-period log differences down the time axis; (n - 1, p)."""
    logged = np.log(np.asarray(x, dtype=float64) + offset)
    return np.diff(logged, axis=0)


def diff(x: NDF, order: int = 1) -> NDF:
    """``order``-th difference down the time axis; (n - order, p)."""
    if order < 1:
        raise ValueError("diff order must be at least 1.")
    return np.diff(np.asarray(x, dtype=float64), n=order, axis=0)


def _windows(x: NDF, window: int) -> NDF:
    """Every trailing window as its own slice: (n - window + 1, p, window)."""
    if window < 1:
        raise ValueError("rolling window must be at least 1.")
    arr = np.asarray(x, dtype=float64)
    if window > arr.shape[0]:
        raise ValueError(
            f"rolling window ({window}) exceeds input length ({arr.shape[0]})."
        )
    return np.lib.stride_tricks.sliding_window_view(arr, window, axis=0)


def rolling_mean(x: NDF, window: int = 10) -> NDF:
    """Trailing rolling mean; (n - window + 1, p)."""
    out: NDF = _windows(x, window).mean(axis=-1)
    return out


def rolling_std(x: NDF, window: int = 10, ddof: int = 0) -> NDF:
    """Trailing rolling standard deviation; (n - window + 1, p)."""
    out: NDF = _windows(x, window).std(axis=-1, ddof=ddof)
    return out


def rolling_var(x: NDF, window: int = 10, ddof: int = 0) -> NDF:
    """Trailing rolling variance; (n - window + 1, p)."""
    out: NDF = _windows(x, window).var(axis=-1, ddof=ddof)
    return out
