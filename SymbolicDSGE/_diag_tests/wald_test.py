from .hac_covariance import jit_hac_estimator_loop_into

import numpy as np
from numpy import float64, int64
from numpy.typing import NDArray

from numba import njit

from typing import Callable

NDF = NDArray[float64]

OK = int64(0)
ERR_BAD_SHAPE = int64(-1)
ERR_LINALG = int64(-2)

F64_NAN = float64(np.nan)


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
def jit_wald_stat_from_mean_and_cov(
    mean: NDF,
    target: NDF,
    omega: NDF,
    n: int,
) -> tuple[int64, float64, int64]:
    q = mean.shape[0]

    if target.shape[0] != q:
        return ERR_BAD_SHAPE, F64_NAN, int64(q)

    dev = np.empty(q, dtype=float64)

    for i in range(q):
        dev[i] = mean[i] - target[i]

    try:
        solved = np.linalg.solve(omega, dev)
    except Exception:
        return ERR_LINALG, F64_NAN, int64(q)

    stat = 0.0
    for i in range(q):
        stat += dev[i] * solved[i]
    stat *= n

    if stat < 0.0 and stat > -1e-12:
        stat = 0.0

    return OK, float64(stat), int64(q)


@njit(cache=True)
def jit_wald_hac_stat(
    g: NDF,
    target: NDF,
    k: Callable[[int, int], float64],
    L: int,
    mean_buffer: NDF,
    centered_buffer: NDF,
    omega_buffer: NDF,
) -> tuple[int64, float64, int64]:
    """
    Generic HAC-robust Wald statistic for H0: E[g_t] = target.

    Parameters
    ----------
    g:
        Moment array with shape (n, q).
    target:
        Target vector with shape (q,).
    k:
        Numba-compatible HAC kernel.
    L:
        HAC bandwidth.
    mean_buf:
        Work buffer with shape (q,).
    centered_buf:
        Work buffer with shape (n, q).
    omega_buf:
        Work buffer with shape (q, q).

    Returns
    -------
    err:
        0 if successful, negative error code otherwise.
    statistic:
        Wald statistic.
    df:
        Chi-square degrees of freedom.
    """
    n, q = g.shape

    if target.shape[0] != q:
        return ERR_BAD_SHAPE, F64_NAN, q

    jit_fill_mean_ax0(g, mean_buffer)
    jit_fill_centered(g, mean_buffer, centered_buffer)

    jit_hac_estimator_loop_into(centered_buffer, k, L, omega_buffer)

    out: tuple[int64, float64, int64] = jit_wald_stat_from_mean_and_cov(
        mean_buffer,
        target,
        omega_buffer,
        n,
    )
    return out
