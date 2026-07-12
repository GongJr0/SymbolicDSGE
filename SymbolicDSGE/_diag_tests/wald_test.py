from .hac_covariance import hac_covariance
from .result import TestResult
from .distributions import PvalMethod, ReferenceDistribution
from .status import TestStatus
from .._ckernels.diag import (
    FALLBACK as DIAG_FALLBACK,
    fill_centered_ax0,
    fill_mean_ax0,
    fill_symmetric_target_vec,
    symmetric_outer_prod_2dim,
    wald_stat_from_mean_and_cov,
)
from ..regression.solvers import _gram_is_pd
import numpy as np
from numpy import float64, int64
from numpy.typing import NDArray
from numpy.linalg import cholesky, solve

from numba import njit

from typing import Literal, cast

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

    # Cholesky on the PD fast path -- same algorithm as the native
    # sdsge_chol_solve, so native parity is same-algorithm rather than
    # Cholesky-vs-LU. LU (np.linalg.solve) is kept only as the fallback for an
    # indefinite-but-nonsingular omega. The PD gate is the deterministic
    # relative-threshold check shared with the native sdsge_chol; relying on
    # np.linalg.cholesky to raise is not reliably catchable in nopython mode.
    if _gram_is_pd(omega):
        L = cholesky(omega)
        z = solve(L, dev)
        solved = solve(L.T, z)
    else:
        try:
            solved = solve(omega, dev)
        except Exception:
            return ERR_LINALG, F64_NAN, int64(q)

    stat = 0.0
    for i in range(q):
        stat += dev[i] * solved[i]
    stat *= n

    if stat < 0.0 and stat > -1e-12:
        stat = 0.0

    return OK, float64(stat), int64(q)


def _wald_stat(
    mean: NDF, target: NDF, omega: NDF, n: int
) -> tuple[int64, float64, int64]:
    """Wald statistic; native Cholesky fast path, numba fallback.

    The native kernel returns ``DIAG_FALLBACK`` when omega is not positive
    definite, in which case the numba kernel recomputes via its LU fallback
    (which handles an indefinite-but-nonsingular omega). Shapes are guaranteed by
    the callers, so the native path is only gated on the extension being present.
    """
    q = mean.shape[0]
    status, stat = wald_stat_from_mean_and_cov(mean, target, omega, n)
    if status != DIAG_FALLBACK:
        return int64(status), float64(stat), int64(q)
    nb_err, nb_stat, nb_df = jit_wald_stat_from_mean_and_cov(mean, target, omega, n)
    return int64(nb_err), float64(nb_stat), int64(nb_df)


def _fill_symmetric_target_vec(target: NDF) -> tuple[int64, NDF]:
    """Pack a symmetric target into its vech vector.

    Pure copy that returns BAD_SHAPE only on a genuinely asymmetric target (a hard
    error the caller raises on).
    """
    status, vec = fill_symmetric_target_vec(
        np.ascontiguousarray(target, dtype=float64),
        SYMMETRY_ATOL,
        SYMMETRY_RTOL,
    )
    return int64(status), vec


def _symmetric_outer_prod(x: NDF) -> tuple[int64, NDF]:
    """Per-row vech of the outer product x_t x_t'."""
    status, out = symmetric_outer_prod_2dim(x)
    return int64(status), out


def _fill_mean_ax0(x: NDF) -> NDF:
    """Column means of x over axis 0."""
    return fill_mean_ax0(x)


def _fill_centered_ax0(x: NDF, mean: NDF) -> NDF:
    """x with its column means subtracted."""
    return fill_centered_ax0(
        np.ascontiguousarray(x, dtype=float64),
        np.ascontiguousarray(mean, dtype=float64),
    )


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
    n = g_arr.shape[0]

    mean_buffer = _fill_mean_ax0(g_arr)
    centered_buffer = _fill_centered_ax0(g_arr, mean_buffer)

    omega = hac_covariance(
        centered_buffer,
        kernel=kernel,
        bandwidth=bandwidth,
        center=False,
    )
    out: tuple[int64, float64, int64] = _wald_stat(
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

    err, target_vec = _fill_symmetric_target_vec(target_arr)
    if err != OK:
        raise ValueError("Target covariance matrix must be symmetric")

    mean = _fill_mean_ax0(g_arr)
    centered = _fill_centered_ax0(g_arr, mean)

    err, g_cov = _symmetric_outer_prod(centered)
    if err != OK:
        raise ValueError(
            f"Failed to compute symmetric outer product with error code {err}"
        )

    g_cov_mean = _fill_mean_ax0(g_cov)
    g_cov_centered = _fill_centered_ax0(g_cov, g_cov_mean)

    omega = hac_covariance(
        g_cov_centered,
        kernel=kernel,
        bandwidth=bandwidth,
        center=False,
    )

    out: tuple[int64, float64, int64] = _wald_stat(
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

    err, target_vec = _fill_symmetric_target_vec(target_arr)
    if err != OK:
        raise ValueError("Target second moment matrix must be symmetric")

    err, g_second_moment = _symmetric_outer_prod(g_arr)
    if err != OK:
        raise ValueError(
            f"Failed to compute symmetric outer product with error code {err}"
        )

    g_second_moment_mean = _fill_mean_ax0(g_second_moment)
    g_second_moment_centered = _fill_centered_ax0(g_second_moment, g_second_moment_mean)

    omega = hac_covariance(
        g_second_moment_centered,
        kernel=kernel,
        bandwidth=bandwidth,
        center=False,
    )

    out: tuple[int64, float64, int64] = _wald_stat(
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
