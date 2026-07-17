"""Numba reference kernels for the diagnostic tests.

Relocated from ``SymbolicDSGE/_diag_tests/*`` when those modules went
native-only. Each function is a pure-numba reimplementation of a native
``_ckernels.diag`` kernel; the parity tests check the native kernels against
these oracles.

Only the pure-duplicate kernels live here. The rank/PD-deficient numba paths
(the ``DIAG_FALLBACK`` retries) stay in the library: they have no native
counterpart and are a live code path, not a test reference.
"""

from __future__ import annotations

from math import erf, erfc, exp, sqrt
from typing import TypeAlias

import numpy as np
from numpy import float64, int64
from numpy.typing import NDArray

from numba import njit

from SymbolicDSGE._diag_tests.status import TestStatus

OK = int(TestStatus.OK)
UDEF_VARIANCE = int(TestStatus.UDEF_VARIANCE)
BAD_SHAPE = int(TestStatus.BAD_SHAPE)
INSUFFICIENT_SAMPLES = int(TestStatus.INSUFFICIENT_SAMPLES)
BAD_LAG = int(TestStatus.BAD_LAG)

LOOP_LIMIT_N = 1e6

NDF: TypeAlias = NDArray[float64]


@njit(cache=True)
def jb_stat(x: NDF) -> tuple[int, float64]:
    """Calculate the Jarque-Bera test statistic."""
    if x.ndim != 1:
        return BAD_SHAPE, float64(np.nan)

    n = x.size
    if n == 0:
        return INSUFFICIENT_SAMPLES, float64(np.nan)

    mean = 0.0
    for i in range(n):
        mean += x[i]
    mean /= n

    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    for i in range(n):
        centered = x[i] - mean
        centered2 = centered * centered
        m2 += centered2
        m3 += centered2 * centered
        m4 += centered2 * centered2
    m2 /= n
    m3 /= n
    m4 /= n

    if not np.isfinite(m2) or m2 <= 0.0:
        return UDEF_VARIANCE, float64(np.nan)

    skew = m3 / m2**1.5
    kurt = m4 / m2**2.0
    kurt_minus_3 = kurt - 3.0

    inner = (skew * skew) / 6.0 + (kurt_minus_3 * kurt_minus_3) / 24.0
    if n >= 10:
        return OK, n * inner
    else:
        return INSUFFICIENT_SAMPLES, n * inner


@njit(cache=True)
def acorr(x: NDF, L: int) -> tuple[int, NDF]:
    """Autocorrelation of x up to lag L."""
    n = x.size
    out = np.empty(L + 1, dtype=float64)

    mu = 0.0
    denom = 0.0
    z = np.empty_like(x)

    if n <= LOOP_LIMIT_N:
        for i in range(n):
            mu += x[i]
        mu /= n

        for i in range(n):
            z[i] = x[i] - mu
            denom += z[i] * z[i]

    else:
        mu = x.mean()
        z = x - mu
        denom = np.dot(z, z)

    if denom <= 0.0 or not np.isfinite(denom):
        for ell in range(L + 1):
            out[ell] = np.nan
        return UDEF_VARIANCE, out

    out[0] = 1.0  # L0

    for ell in range(1, L + 1):
        num = 0.0
        for t in range(ell, n):
            num += z[t] * z[t - ell]
        out[ell] = num / denom
    return OK, out


@njit(cache=True)
def lb_stat(x: NDF, L: int) -> tuple[int, float64]:
    """Ljung-Box test statistic for x up to lag L."""
    if x.ndim != 1:
        return BAD_SHAPE, float64(np.nan)

    n = x.size
    if n <= 1:
        return INSUFFICIENT_SAMPLES, float64(np.nan)

    if L >= n:
        L = n - 1
    if L <= 0:
        return BAD_LAG, float64(np.nan)

    err, rho = acorr(x, L)
    if err != OK:
        return err, float64(np.nan)  # type(np.nan) == float

    n = x.size
    stat = 0.0
    for ell in range(1, L + 1):
        stat += (rho[ell] * rho[ell]) / (n - ell)
    stat *= n * (n + 2)
    return OK, float64(stat)


# HAC kernel ids (mirror SymbolicDSGE._diag_tests.hac_covariance).
_BARTLETT: int = 0
_PARZEN: int = 1
_QS: int = 2


@njit(cache=True)
def _kernel_weight(j: int, L: int, kernel_id: int) -> float64:
    """Lag-window weight w(j; L) for the selected kernel."""
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


@njit(cache=True)
def jit_hac_estimator_matmul(r: NDF, kernel_id: int, L: int) -> NDF:
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


def jit_fill_mean_ax0(x: NDF, mean: NDF) -> None:
    # Plain-Python oracle helper: exact copy/mean over integer indices (no float
    # summation-order risk), single test callsite -- @njit warmup would be pure
    # cost here.
    n, q = x.shape

    for j in range(q):
        mean[j] = 0.0

    for t in range(n):
        for j in range(q):
            mean[j] += x[t, j]

    for j in range(q):
        mean[j] /= n


def jit_fill_centered(x: NDF, mean: NDF, centered: NDF) -> None:
    n, q = x.shape

    for t in range(n):
        for j in range(q):
            centered[t, j] = x[t, j] - mean[j]


# Wald vech helpers (mirror SymbolicDSGE._diag_tests.wald_test).
_OK_I64 = int64(TestStatus.OK)
_BAD_SHAPE_I64 = int64(TestStatus.BAD_SHAPE)
SYMMETRY_ATOL = float64(1e-8)
SYMMETRY_RTOL = float64(1e-5)


def jit_symmetric_outer_prod_2dim(x: NDF, out: NDF) -> int64:
    n = x.shape[0]
    p = x.shape[1]
    q = p * (p + 1) // 2

    if out.shape[0] != n or out.shape[1] != q:
        return _BAD_SHAPE_I64

    for t in range(n):
        k = 0
        for i in range(p):
            x_i = x[t, i]
            for j in range(i, p):
                out[t, k] = x_i * x[t, j]
                k += 1
    return _OK_I64


def jit_fill_symmetric_target_vec(
    target: NDF,
    out: NDF,
    atol: float64 = SYMMETRY_ATOL,
    rtol: float64 = SYMMETRY_RTOL,
) -> int64:
    p = target.shape[0]
    q = p * (p + 1) // 2

    if target.shape[1] != p or out.shape[0] != q:
        return _BAD_SHAPE_I64

    k = 0
    for i in range(p):
        for j in range(i, p):
            a = target[i, j]
            b = target[j, i]
            if a != b:
                diff = abs(a - b)
                if not np.isfinite(diff) or diff > atol + rtol * abs(b):
                    return _BAD_SHAPE_I64
            out[k] = a
            k += 1

    return _OK_I64


# --- cusum: Durbin reference distribution, survival-function chain ------------
# Independent reimplementation of the CusumDist sf kernels in
# SymbolicDSGE/_diag_tests/cusum.py; the parity tests pin the native
# diag_cusum.c against these.
_CUSUM_SQRT_2 = sqrt(2.0)


@njit(cache=True)
def _cusum_cdf(x: float64) -> float64:
    return float64(0.5 * (1.0 + erf(x / _CUSUM_SQRT_2)))


@njit(cache=True)
def _cusum_sf(x: float64) -> float64:
    return float64(0.5 * erfc(x / _CUSUM_SQRT_2))


@njit(cache=True)
def cusum_alpha_from_a(a: float64) -> float64:
    return float64(
        2.0 * (_cusum_sf(float64(2.0 * a)) + exp(-4.0 * a * a) * _cusum_cdf(a))
    )


@njit(cache=True)
def cusum_alpha_from_a_array(a: NDF) -> NDF:
    out = np.empty_like(a)
    for i in range(a.size):
        out[i] = cusum_alpha_from_a(a[i])
    return out
