from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from numba import njit

from scipy.stats import norm
from scipy.stats._distn_infrastructure import rv_frozen

from SymbolicDSGE._diag_tests.distributions import PvalMethod, ReferenceDistribution
from SymbolicDSGE._diag_tests.result import TestResult

from .cusum_utils import recursive_residuals
from .status import TestStatus
from .._ckernels.diag import (
    cusumsq_stat as _native_cusumsq_stat,
    FALLBACK as DIAG_FALLBACK,
)

if TYPE_CHECKING:
    import optype.numpy as onp

NDF: TypeAlias = NDArray[float64]
DistributionOutput: TypeAlias = float | float64 | NDF

OK = int(TestStatus.OK)


def _cusumsq_stat(y: NDF, X: NDF) -> tuple[int, int, float64]:
    """CUSUM-of-squares statistic; native fast path, numba fallback."""
    if y.shape[0] == X.shape[0]:
        status, n, stat = _native_cusumsq_stat(
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(X, dtype=np.float64),
        )
        if status != DIAG_FALLBACK:
            return status, int(n), float64(stat)
    nb_status, nb_n, nb_stat = _cusumsq_stat_numba(y, X)
    return int(nb_status), int(nb_n), float64(nb_stat)


@njit(cache=True)
def _cusumsq_stat_numba(y: NDF, X: NDF) -> tuple[int, int, float64]:
    status, rec_eps = recursive_residuals(y, X)
    N = rec_eps.size
    if status != OK:
        return status, N, float64(np.nan)
    sq = rec_eps * rec_eps
    s = np.cumsum(sq) / np.sum(sq)
    # Brown-Durbin-Evans CUSUM of squares: under coefficient stability the
    # normalized partial sums s_t track the line t/N, so the statistic is the
    # maximum absolute departure from that line (a Kolmogorov-Smirnov-type
    # deviation). The squared recursive residuals are chi^2(1) with variance 2,
    # so s_t - t/N converges to sqrt(2/N) * Brownian bridge; dividing by sqrt(2)
    # standardizes it onto the unit bridge whose crossing law the survival
    # function (with the sqrt(n) scaling) approximates.
    expected = np.arange(1, N + 1) / N
    stat = np.max(np.abs(s - expected)) / np.sqrt(2.0)
    return OK, N, float64(stat)


@njit(cache=True)
def _cusumsq_sf(stat: float64, n: int, iter: int = 20) -> float64:
    statsq = stat * stat
    sum = float64(0.0)
    for j in range(1, iter + 1):
        sum += (-1) ** (j - 1) * np.exp(-2.0 * j * j * statsq * n)
    return float64(2.0 * sum)


@njit(cache=True)
def _cusumsq_sf_array(stat: NDF, n: int) -> NDF:
    out = np.empty_like(stat)
    for i in range(stat.size):
        out[i] = _cusumsq_sf(stat[i], n)
    return out


class CusumSq(rv_frozen):
    """Kolmogorov-type approximation to the distribution of the CUSUM-of-squares
    statistic of recursive residuals.

    Unlike the parameter-free CUSUM distribution, the boundary-crossing
    probability depends on the number of recursive residuals ``n = T - p``, so
    the distribution is parameterized by ``n`` (forwarded through ``freeze`` as
    the ``df``, validated as an integer). The survival function *is*
    ``_cusumsq_sf``; exposing it as a ``FrozenDistribution`` lets CUSUMSQ reuse
    the standard ``TestResult`` p-value machinery via ``PvalMethod.SF``.
    """

    def __init__(self, n: int | np.integer[Any]) -> None:
        super().__init__(norm)  # placeholder base; all moments overridden
        self.n = int(n)

    @overload
    def sf(self, x: onp.ToFloat, /) -> DistributionOutput: ...
    @overload
    def sf(self, x: onp.ToFloatND, /) -> NDF: ...
    def sf(self, x: Any, /) -> DistributionOutput:  # pyright: ignore
        # The truncated alternating series can overshoot [0, 1] for small
        # statistics; clamp so the survival function is always a valid
        # probability.
        return np.clip(self._eval_sf(x), 0.0, 1.0)

    @overload
    def cdf(self, x: onp.ToFloat, /) -> DistributionOutput: ...
    @overload
    def cdf(self, x: onp.ToFloatND, /) -> NDF: ...
    def cdf(self, x: Any, /) -> DistributionOutput:  # pyright: ignore
        return cast(DistributionOutput, 1.0 - self.sf(x))

    def _eval_sf(self, x: Any, /) -> DistributionOutput:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim == 0:
            return float64(_cusumsq_sf(float64(values.item()), self.n))
        flat = np.ascontiguousarray(values.reshape(-1))
        return cast(NDF, _cusumsq_sf_array(flat, self.n).reshape(values.shape))


def cusumsq_test(
    y: NDF, X: NDF, alpha: float = 0.05, _auto_pval: bool = True
) -> TestResult:
    status, n, stat = _cusumsq_stat(y, X)
    return TestResult(
        test_name="CUSUMSQ",
        status=TestStatus(status),
        statistic=stat,
        df=n,
        dist=ReferenceDistribution.CUSUMSQ,
        pval_method=PvalMethod.SF,
        alpha=float64(alpha),
        _auto_pval=_auto_pval,
    )
