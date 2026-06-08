import numpy as np
from numpy import float64
from numpy.typing import NDArray
from numba import njit

NDF = NDArray[float64]


@njit(cache=True)
def jit_fill_mean_ax0(x: NDF, mean: NDF) -> None:
    n, q = x.shape

    for j in range(q):
        mean[j] = 0.0

    for t in range(n):
        for j in range(q):
            mean[j] += x[t, j]

    for j in range(q):
        mean[j] /= n


@njit(cache=True)
def jit_fill_centered(x: NDF, mean: NDF, centered: NDF) -> None:
    n, q = x.shape

    for t in range(n):
        for j in range(q):
            centered[t, j] = x[t, j] - mean[j]


@njit(cache=True)
def jit_nth_central_moment_into(x: NDF, mean: float64, n: int, out: NDF) -> None:
    """Calculate the n-th central moment of x and store it in out."""
    for i in range(x.size):
        out[0] += (x[i] - mean) ** n
    out[0] /= x.size
