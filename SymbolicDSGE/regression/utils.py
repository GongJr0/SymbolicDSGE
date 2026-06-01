from __future__ import annotations
import numpy as np
from numpy import float64
from numpy.typing import NDArray
from numba import njit

NDF = NDArray[float64]


@njit(cache=True)
def log_grid(start: float64, stop: float64, num: int) -> NDF:
    if num == 1:
        return np.array([start], dtype=float64)
    elif num == 2:
        return np.array([start, stop], dtype=float64)
    else:
        return np.exp(np.linspace(np.log(start), np.log(stop), num=num))


def process_args(
    x: NDF, y: NDF, variables: list[str] | None
) -> tuple[NDF, NDF, list[str]]:
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError("design matrix must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("response must be a 1D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("response and design matrix row counts differ.")
    if variables is None:
        variables = [f"x{i}" for i in range(x.shape[1])]
    elif len(variables) != x.shape[1]:
        raise ValueError("variables must match the number of design columns.")

    return x, y, variables
