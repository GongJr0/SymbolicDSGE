from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

import numpy as np
from numpy import float64
from numba import njit

# Normal distribution with Numba JIT
from math import sqrt, erf, erfc, exp, log

from scipy.stats import norm
from scipy.stats._distn_infrastructure import rv_frozen

from .status import TestStatus
from .result import TestResult
from .distributions import PvalMethod, ReferenceDistribution
from .cusum_utils import OK, NDF, recursive_residuals

from ..regression.solvers import chol_solve, lstsq_solve
from ._native import native as _native, DIAG_FALLBACK

if TYPE_CHECKING:
    import optype.numpy as onp

DistributionOutput: TypeAlias = float | float64 | NDF

# Newton Status
CONVERGED = 0
ITER_LIM = 1
OOB = 2

# Distribution Code

SQRT_2 = sqrt(2.0)
SQRT_2PI = sqrt(2.0 * np.pi)
INV_SQRT_2PI = 1.0 / SQRT_2PI


@njit(cache=True)
def _pdf(x: float64) -> float64:
    return float64(INV_SQRT_2PI * exp(-0.5 * x * x))


@njit(cache=True)
def _cdf(x: float64) -> float64:
    return float64(0.5 * (1.0 + erf(x / SQRT_2)))


@njit(cache=True)
def _sf(x: float64) -> float64:
    return float64(0.5 * erfc(x / SQRT_2))


# Durbin's approximation
@njit(cache=True)
def _alpha_from_a(a: float64) -> float64:
    return float64(2.0 * (_sf(float64(2.0 * a)) + exp(-4.0 * a * a) * _cdf(a)))


@njit(cache=True)
def _d_da(a: float64) -> float64:
    e = exp(-4.0 * a * a)
    return float64(
        2.0 * (-2.0 * _pdf(float64(2.0 * a)) + e * _pdf(a) - 8.0 * a * e * _cdf(a))
    )


# Inverse for a = f(alpha) using Newton's method
@njit(cache=True)
def _a_from_alpha(
    alpha: float64, tol: float = 1e-13, max_iter: int = 50
) -> tuple[int, float64]:
    if alpha <= 0.0 or alpha >= 1.0:
        return OOB, float64(np.nan)

    lo = 0.0

    # approximate a:
    hi = float64(max(1.0, -0.5 * sqrt(-0.5 * log(alpha / 2.0))))

    while _alpha_from_a(hi) > alpha:
        hi *= 2.0

    a: float64 = hi

    for _ in range(max_iter):
        fa = _alpha_from_a(a) - alpha
        if abs(fa) < tol:
            return CONVERGED, a

        if fa > 0.0:
            lo = a
        else:
            hi = a

        da = _d_da(a)
        if da == 0.0:
            cand: float64 = float64(0.5 * (lo + hi))
        else:
            cand = a - fa / da
            # Reject Newton steps that leave the bracket; bisect instead so
            # the iterate can never escape to a region where da underflows.
            if cand <= lo or cand >= hi:
                cand = float64(0.5 * (lo + hi))

        if abs(cand - a) < tol * max(1.0, abs(a)):
            return CONVERGED, cand
        a = cand
    return ITER_LIM, a


# Reference distribution
@njit(cache=True)
def _alpha_from_a_array(a: NDF) -> NDF:
    out = np.empty_like(a)
    for i in range(a.size):
        out[i] = _alpha_from_a(a[i])
    return out


@njit(cache=True)
def _a_from_alpha_array(alpha: NDF) -> NDF:
    out = np.empty_like(alpha)
    for i in range(alpha.size):
        out[i] = _a_from_alpha(alpha[i])[1]
    return out


def _isf_scalar(alpha: float64) -> float64:
    return float64(_a_from_alpha(alpha)[1])


def _dispatch(
    x: Any,
    scalar_fn: Callable[[float64], float64],
    array_fn: Callable[[NDF], NDF],
) -> DistributionOutput:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 0:
        return float64(scalar_fn(float64(values.item())))
    flat = np.ascontiguousarray(values.reshape(-1))
    return cast(NDF, array_fn(flat).reshape(values.shape))


class CusumDist(rv_frozen):
    """Durbin's (1969) approximation to the distribution of the recursive-
    residual CUSUM statistic.

    The test is parameter-free: the boundary-crossing probability is a closed
    form of the statistic itself, so the survival function *is* ``_alpha_from_a``
    and the inverse survival function *is* ``_a_from_alpha``. Exposing it as a
    ``FrozenDistribution`` lets CUSUM reuse the standard ``TestResult`` p-value
    machinery (``dist.freeze().sf(stat)`` via ``PvalMethod.SF``) instead of a
    bespoke code path. It takes no distribution parameters; any ``df`` passed
    through ``freeze`` is ignored.
    """

    def __init__(self) -> None:
        super().__init__(norm)  # placeholder base; all moments overridden

    @overload
    def sf(self, x: onp.ToFloat, /) -> DistributionOutput: ...
    @overload
    def sf(self, x: onp.ToFloatND, /) -> NDF: ...
    def sf(self, x: Any, /) -> DistributionOutput:  # pyright: ignore
        # Durbin's series can exceed 1 for small statistics; clamp so the
        # survival function always returns a valid probability.
        return np.minimum(1.0, _dispatch(x, _alpha_from_a, _alpha_from_a_array))

    @overload
    def cdf(self, x: onp.ToFloat, /) -> DistributionOutput: ...
    @overload
    def cdf(self, x: onp.ToFloatND, /) -> NDF: ...
    def cdf(self, x: Any, /) -> DistributionOutput:  # pyright: ignore
        return cast(DistributionOutput, 1.0 - self.sf(x))

    @overload
    def isf(self, q: onp.ToFloat, /) -> DistributionOutput: ...
    @overload
    def isf(self, q: onp.ToFloatND, /) -> NDF: ...
    def isf(self, q: Any, /) -> DistributionOutput:  # pyright: ignore
        return _dispatch(q, _isf_scalar, _a_from_alpha_array)

    @overload
    def ppf(self, q: onp.ToFloat, /) -> DistributionOutput: ...
    @overload
    def ppf(self, q: onp.ToFloatND, /) -> NDF: ...
    def ppf(self, q: Any, /) -> DistributionOutput:  # pyright: ignore
        return self.isf(1.0 - np.asarray(q, dtype=np.float64))


# Test Code
def cusum_series(y: NDF, X: NDF) -> tuple[int, NDF]:
    """Standardized CUSUM series; native fast path, numba fallback."""
    if _native is not None and y.shape[0] == X.shape[0]:
        status, series = _native.cusum_series(
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(X, dtype=np.float64),
        )
        if status != DIAG_FALLBACK:
            return status, series
    nb_status, nb_series = _cusum_series_numba(y, X)
    return int(nb_status), nb_series


@njit(cache=True)
def _cusum_series_numba(y: NDF, X: NDF) -> tuple[int, NDF]:
    T, p = X.shape
    status, rec_resid = recursive_residuals(y, X)
    if status != OK:
        return status, rec_resid

    try:
        bhat, _, _ = chol_solve(X, y)
    except Exception:
        bhat, _, _ = lstsq_solve(X, y)

    resid = y - X @ bhat
    sigma_hat = sqrt(np.sum(resid**2) / (T - p))  # DoF correction

    std_cusum = np.cumsum(rec_resid) / sigma_hat
    return OK, std_cusum


def cusum_stat(y: NDF, X: NDF) -> tuple[int, float64]:
    """CUSUM statistic; native fast path, numba fallback."""
    if _native is not None and y.shape[0] == X.shape[0]:
        status, stat = _native.cusum_stat(
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(X, dtype=np.float64),
        )
        if status != DIAG_FALLBACK:
            return status, float64(stat)
    nb_status, nb_stat = _cusum_stat_numba(y, X)
    return int(nb_status), float64(nb_stat)


@njit(cache=True)
def _cusum_stat_numba(y: NDF, X: NDF) -> tuple[int, float64]:
    T, p = X.shape
    status, cusum = _cusum_series_numba(y, X)
    if status != OK:
        return status, float64(np.nan)

    scaled = np.empty(T - p, dtype=float64)
    sqrt_Tp = sqrt(T - p)
    abs_cusum = np.abs(cusum)
    for i, t in enumerate(range(p, T)):
        denom = sqrt_Tp + (2 * (t - p) / sqrt_Tp)
        scaled[i] = abs_cusum[i] / denom
    return status, np.max(scaled)


def cusum(y: NDF, X: NDF, alpha: float = 0.05, _auto_pval: bool = True) -> TestResult:
    test_status, stat = cusum_stat(y, X)
    # _a_from_alpha is only used to validate alpha (in-bounds + convergent);
    # the p-value itself comes from sf(stat) inside TestResult.
    newton_status, _ = _a_from_alpha(float64(alpha))

    test_ok = test_status == OK
    nwt_ok = newton_status == CONVERGED

    if test_ok and nwt_ok:
        status = TestStatus.OK
    elif not test_ok:
        status = TestStatus(test_status)
    else:
        if newton_status == ITER_LIM:
            status = TestStatus.ITERATIVE_ALG_NONCONVERGENCE
        else:
            status = TestStatus.BAD_PARAMETER

    return TestResult(
        test_name="cusum",
        statistic=stat,
        df=float64(np.nan),
        dist=ReferenceDistribution.CUSUM,
        pval_method=PvalMethod.SF,
        alpha=float64(alpha),
        status=status,
        _auto_pval=_auto_pval,
    )
