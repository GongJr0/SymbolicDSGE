from .hac_covariance import hac_covariance
from .moment_calculation_utils import jit_fill_centered, jit_fill_mean_ax0
from .result import TestResult
from .distributions import PvalMethod, ReferenceDistribution
from .status import TestStatus
import numpy as np
from numpy import float64, int64
from numpy.typing import NDArray

from numba import njit

from typing import Literal

NDF = NDArray[float64]

OK = int64(TestStatus.OK)
ERR_BAD_SHAPE = int64(TestStatus.BAD_SHAPE)
ERR_LINALG = int64(TestStatus.LINALG)

F64_NAN = float64(np.nan)
SYMMETRY_ATOL = float64(1e-8)
SYMMETRY_RTOL = float64(1e-5)


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
def jit_symmetric_outer_prod_2dim(x: NDF, out: NDF) -> int64:
    n = x.shape[0]
    p = x.shape[1]
    q = p * (p + 1) // 2

    if out.shape[0] != n or out.shape[1] != q:
        return ERR_BAD_SHAPE

    for t in range(n):
        k = 0
        for i in range(p):
            x_i = x[t, i]
            for j in range(i, p):
                out[t, k] = x_i * x[t, j]
                k += 1
    return OK


@njit(cache=True)
def jit_fill_symmetric_target_vec(
    target: NDF,
    out: NDF,
    atol: float64 = SYMMETRY_ATOL,
    rtol: float64 = SYMMETRY_RTOL,
) -> int64:
    p = target.shape[0]
    q = p * (p + 1) // 2

    if target.shape[1] != p or out.shape[0] != q:
        return ERR_BAD_SHAPE

    k = 0
    for i in range(p):
        for j in range(i, p):
            a = target[i, j]
            b = target[j, i]
            if a != b:
                diff = abs(a - b)
                if not np.isfinite(diff) or diff > atol + rtol * abs(b):
                    return ERR_BAD_SHAPE
            out[k] = a
            k += 1

    return OK


# ---- Preconfigured Mean and Covariance Tests ---


def wald_mean_hac(
    g: NDF,
    target: NDF,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["andrews", "wooldridge", "auto"] | None = "auto",
    alpha: float = 0.05,
    _auto_pval: bool = True,
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

    g_arr = np.ascontiguousarray(g, dtype=float64)
    target_arr = np.ascontiguousarray(target, dtype=float64)
    n, q = g_arr.shape

    mean_buffer = np.empty(q, dtype=float64)
    centered_buffer = np.empty_like(g_arr)
    jit_fill_mean_ax0(g_arr, mean_buffer)
    jit_fill_centered(g_arr, mean_buffer, centered_buffer)

    omega = hac_covariance(
        centered_buffer,
        kernel=kernel,
        bandwidth=bandwidth,
        center=False,
        nopython=True,
    )
    out: tuple[int64, float64, int64] = jit_wald_stat_from_mean_and_cov(
        mean_buffer,
        target_arr,
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
        status=TestStatus.OK,
        _auto_pval=_auto_pval,
    )
    return res


def wald_covariance_hac(
    g: NDF,
    target: NDF,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["andrews", "wooldridge", "auto"] | None = "auto",
    alpha: float = 0.05,
    _auto_pval: bool = True,
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

    g_arr = np.ascontiguousarray(g, dtype=float64)
    target_arr = np.ascontiguousarray(target, dtype=float64)

    if target_arr.ndim != 2 or target_arr.shape[0] != target_arr.shape[1]:
        raise ValueError("Target covariance matrix must be square")
    if target_arr.shape[0] != g_arr.shape[1]:
        raise ValueError("Target covariance matrix dimension must match g columns")

    n, p = g_arr.shape
    q = p * (p + 1) // 2
    target_vec = np.empty(q, dtype=float64)
    err = jit_fill_symmetric_target_vec(target_arr, target_vec)
    if err != OK:
        raise ValueError("Target covariance matrix must be symmetric")

    centered = np.empty_like(g_arr)
    mean = np.empty(p, dtype=float64)
    jit_fill_mean_ax0(g_arr, mean)
    jit_fill_centered(g_arr, mean, centered)

    g_cov = np.empty((n, q), dtype=float64)
    err = jit_symmetric_outer_prod_2dim(centered, g_cov)
    if err != OK:
        raise ValueError(
            f"Failed to compute symmetric outer product with error code {err}"
        )

    g_cov_mean = np.empty(q, dtype=float64)
    g_cov_centered = np.empty_like(g_cov)
    jit_fill_mean_ax0(g_cov, g_cov_mean)
    jit_fill_centered(g_cov, g_cov_mean, g_cov_centered)

    omega = hac_covariance(
        g_cov_centered,
        kernel=kernel,
        bandwidth=bandwidth,
        center=False,
        nopython=True,
    )

    out: tuple[int64, float64, int64] = jit_wald_stat_from_mean_and_cov(
        g_cov_mean,
        target_vec,
        omega,
        g_arr.shape[0],
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
        status=TestStatus.OK,
        _auto_pval=_auto_pval,
    )
    return res


def wald_second_moment_hac(
    g: NDF,
    target: NDF,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["andrews", "wooldridge", "auto"] | None = "auto",
    alpha: float = 0.05,
    _auto_pval: bool = True,
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

    g_arr = np.ascontiguousarray(g, dtype=float64)
    target_arr = np.ascontiguousarray(target, dtype=float64)

    if target_arr.ndim != 2 or target_arr.shape[0] != target_arr.shape[1]:
        raise ValueError("Target second moment matrix must be square")
    if target_arr.shape[0] != g_arr.shape[1]:
        raise ValueError("Target second moment matrix dimension must match g columns")

    n, p = g_arr.shape
    q = p * (p + 1) // 2
    target_vec = np.empty(q, dtype=float64)
    err = jit_fill_symmetric_target_vec(target_arr, target_vec)
    if err != OK:
        raise ValueError("Target second moment matrix must be symmetric")

    g_second_moment = np.empty((n, q), dtype=float64)
    err = jit_symmetric_outer_prod_2dim(g_arr, g_second_moment)
    if err != OK:
        raise ValueError(
            f"Failed to compute symmetric outer product with error code {err}"
        )

    g_second_moment_mean = np.empty(q, dtype=float64)
    g_second_moment_centered = np.empty_like(g_second_moment)
    jit_fill_mean_ax0(g_second_moment, g_second_moment_mean)
    jit_fill_centered(
        g_second_moment,
        g_second_moment_mean,
        g_second_moment_centered,
    )

    omega = hac_covariance(
        g_second_moment_centered,
        kernel=kernel,
        bandwidth=bandwidth,
        center=False,
        nopython=True,
    )

    out: tuple[int64, float64, int64] = jit_wald_stat_from_mean_and_cov(
        g_second_moment_mean,
        target_vec,
        omega,
        g_arr.shape[0],
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
        status=TestStatus.OK,
        _auto_pval=_auto_pval,
    )
    return res
