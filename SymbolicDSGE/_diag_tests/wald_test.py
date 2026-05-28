from .hac_covariance import jit_hac_estimator_loop_into, hac_covariance
from .moment_calculation_utils import jit_fill_centered, jit_fill_mean_ax0
from .result import TestResult
from .distributions import PvalMethod, ReferenceDistribution
import numpy as np
from numpy import float64, int64
from numpy.typing import NDArray

from numba import njit

from typing import Callable, Literal

NDF = NDArray[float64]

OK = int64(0)
ERR_BAD_SHAPE = int64(-1)
ERR_LINALG = int64(-2)

F64_NAN = float64(np.nan)


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


# ---- Preconfigured Mean and Covariance Tests ---


def wald_mean_hac(
    g: NDF,
    target: NDF,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["andrews", "wooldridge", "auto"] | None = "auto",
    alpha: float = 0.05,
) -> TestResult:
    """
    HAC-robust Wald test for H0: E[g_t] = target.

    Parameters
    ----------
    g:
        Moment array with shape (n, q).
    target:
        Target vector with shape (q,).
    kernel:
        HAC kernel to use. One of "bartlett", "parzen", or "qs".
    bandwidth:
        HAC bandwidth. If an integer is provided, it is used directly. If "andrews", "wooldridge", or "auto" is provided, the bandwidth is selected using the corresponding method from Andrews (1991) or Wooldridge (2006). If None, defaults to "auto".
    alpha:
        Significance level for the test, used for p-value calculation. Default is 0.05.

    Returns
    -------
    TestResult
         Wald test result containing the test statistic, degrees of freedom, and p-value method.
    """

    n, q = g.shape

    mean_buffer = np.empty(q, dtype=float64)
    jit_fill_mean_ax0(g, mean_buffer)

    omega = hac_covariance(
        g,
        kernel=kernel,
        bandwidth=bandwidth,
        center=True,
        nopython=True,
    )
    out: tuple[int64, float64, int64] = jit_wald_stat_from_mean_and_cov(
        mean_buffer,
        target,
        omega,
        n,
    )
    err, stat, df = out
    if err != OK:
        raise ValueError(f"Wald test failed with error code {err}")

    res = TestResult(
        test_name="wald_mean_hac",
        dist=ReferenceDistribution.CHI2,
        df=float64(df),
        statistic=stat,
        pval_method=PvalMethod.SF,
        alpha=float64(alpha),
    )
    return res


def wald_covariance_hac(
    g: NDF,
    target: NDF,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["andrews", "wooldridge", "auto"] | None = "auto",
    alpha: float = 0.05,
) -> TestResult:
    """
    HAC-robust Wald test for H0: Cov[g_t] = target.

    Parameters
    ----------
    g:
        Moment array with shape (n, q).
    target:
        Target covariance matrix with shape (q, q).
    kernel:
        HAC kernel to use. One of "bartlett", "parzen", or "qs".
    bandwidth:
        HAC bandwidth. If an integer is provided, it is used directly. If "andrews", "wooldridge", or "auto" is provided, the bandwidth is selected using the corresponding method from Andrews (1991) or Wooldridge (2006). If None, defaults to "auto".
    alpha:
        Significance level for the test, used for p-value calculation. Default is 0.05.
    """

    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("Target covariance matrix must be square")
    if target.shape[0] != g.shape[1]:
        raise ValueError("Target covariance matrix dimension must match g columns")
    if not np.allclose(target, target.T, atol=1e-8):
        raise ValueError("Target covariance matrix must be symmetric")
    vech_idx = np.triu_indices(target.shape[0])
    target_vec = target[vech_idx]  # p * (p + 1) // 2

    n, _ = g.shape
    q = target_vec.shape[0]

    centered = np.empty_like(g)
    mean = np.empty(g.shape[1], dtype=float64)
    jit_fill_mean_ax0(g, mean)
    jit_fill_centered(g, mean, centered)

    g_cov = np.empty((n, q), dtype=float64)
    for i in range(n):
        g_cov[i] = np.outer(centered[i], centered[i])[vech_idx]

    g_cov_mean = np.empty(q, dtype=float64)
    jit_fill_mean_ax0(g_cov, g_cov_mean)

    omega = hac_covariance(
        g_cov,
        kernel=kernel,
        bandwidth=bandwidth,
        center=True,
        nopython=True,
    )

    out: tuple[int64, float64, int64] = jit_wald_stat_from_mean_and_cov(
        g_cov_mean,
        target_vec,
        omega,
        g.shape[0],
    )
    err, stat, df = out
    if err != OK:
        raise ValueError(f"Wald test failed with error code {err}")

    res = TestResult(
        test_name="wald_covariance_hac",
        dist=ReferenceDistribution.CHI2,
        df=float64(df),
        statistic=stat,
        pval_method=PvalMethod.SF,
        alpha=float64(alpha),
    )
    return res


def wald_second_moment_hac(
    g: NDF,
    target: NDF,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["andrews", "wooldridge", "auto"] | None = "auto",
    alpha: float = 0.05,
) -> TestResult:
    """
    HAC-robust Wald test for H0: E[g_t g_t'] = target.

    Parameters
    ----------
    g:
        Moment array with shape (n, q).
    target:
        Target second moment matrix with shape (q, q).
    kernel:
        HAC kernel to use. One of "bartlett", "parzen", or "qs".
    bandwidth:
        HAC bandwidth. If an integer is provided, it is used directly. If "andrews", "wooldridge", or "auto" is provided, the bandwidth is selected using the corresponding method from Andrews (1991) or Wooldridge (2006). If None, defaults to "auto".
    alpha:
        Significance level for the test, used for p-value calculation. Default is 0.05.
    """

    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("Target second moment matrix must be square")
    if target.shape[0] != g.shape[1]:
        raise ValueError("Target second moment matrix dimension must match g columns")
    if not np.allclose(target, target.T, atol=1e-8):
        raise ValueError("Target second moment matrix must be symmetric")
    vech_idx = np.triu_indices(target.shape[0])
    target_vec = target[vech_idx]  # p * (p + 1) // 2

    n, _ = g.shape
    q = target_vec.shape[0]

    g_second_moment = np.empty((n, q), dtype=float64)
    for i in range(n):
        g_second_moment[i] = np.outer(g[i], g[i])[vech_idx]

    g_second_moment_mean = np.empty(q, dtype=float64)
    jit_fill_mean_ax0(g_second_moment, g_second_moment_mean)

    omega = hac_covariance(
        g_second_moment,
        kernel=kernel,
        bandwidth=bandwidth,
        center=True,
        nopython=True,
    )

    out: tuple[int64, float64, int64] = jit_wald_stat_from_mean_and_cov(
        g_second_moment_mean,
        target_vec,
        omega,
        g.shape[0],
    )
    err, stat, df = out
    if err != OK:
        raise ValueError(f"Wald test failed with error code {err}")

    res = TestResult(
        test_name="wald_second_moment_hac",
        dist=ReferenceDistribution.CHI2,
        df=float64(df),
        statistic=stat,
        pval_method=PvalMethod.SF,
        alpha=float64(alpha),
    )
    return res
