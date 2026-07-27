"""Independent NumPy references for native Monte Carlo sample transforms.

These functions deliberately do not import the production MC transform wrappers:
the wrappers dispatch to the native extension, so doing so would compare the
native kernels with themselves.
"""

from __future__ import annotations

import numpy as np
from numpy import float64
from numpy.typing import NDArray

NDF = NDArray[float64]


def standardize(x: NDF, ddof: int = 0) -> NDF:
    """Per-column z-score, with zero-variance columns set to zero."""
    arr = np.asarray(x, dtype=np.float64)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, ddof=ddof, keepdims=True)
    out = (arr - mean) / np.where(std == 0.0, 1.0, std)
    return np.ascontiguousarray(np.where(std == 0.0, 0.0, out), dtype=np.float64)


def log(x: NDF, offset: float = 0.0) -> NDF:
    """Elementwise ``log(x + offset)``."""
    return np.log(np.asarray(x, dtype=np.float64) + offset)


def log_diff(x: NDF, offset: float = 0.0) -> NDF:
    """One-period log differences along the first axis."""
    return np.diff(log(x, offset), axis=0)


def diff(x: NDF, order: int = 1) -> NDF:
    """Repeated first-axis differences."""
    return np.diff(np.asarray(x, dtype=np.float64), n=order, axis=0)


def _windows(x: NDF, window: int) -> NDF:
    """Trailing windows with shape ``(n - window + 1, p, window)``."""
    return np.lib.stride_tricks.sliding_window_view(
        np.asarray(x, dtype=np.float64), window, axis=0
    )


def rolling_mean(x: NDF, window: int) -> NDF:
    out: NDF = np.asarray(_windows(x, window).mean(axis=-1), dtype=np.float64)
    return out


def rolling_var(x: NDF, window: int, ddof: int = 0) -> NDF:
    out: NDF = np.asarray(_windows(x, window).var(axis=-1, ddof=ddof), dtype=np.float64)
    return out


def rolling_std(x: NDF, window: int, ddof: int = 0) -> NDF:
    out: NDF = np.asarray(_windows(x, window).std(axis=-1, ddof=ddof), dtype=np.float64)
    return out
