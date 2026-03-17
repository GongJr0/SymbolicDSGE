from .distribution import Distribution, RandomState, Size, VecF64
from ..support import OutOfSupportError, Support

import numpy as np
from numpy import float64
from scipy.special import ndtr, ndtri
from numba import njit
from typing import TypedDict, cast, overload


class LogNormalParams(TypedDict):
    mean: float  # Mean of the underlying normal distribution
    std: float  # Standard deviation of the underlying normal distribution
    random_state: RandomState


LOGNORM_DEFAULTS = LogNormalParams(
    mean=0.0,
    std=1.0,
    random_state=None,
)


@njit(cache=True)
def _logpdf_scalar(meanlog: float64, stdlog: float64, x: float64) -> float64:
    return float64(
        -np.log(stdlog)
        - np.log(x)
        - 0.5 * np.log(2.0 * np.pi)
        - 0.5 * ((np.log(x) - meanlog) / stdlog) ** 2
    )


@njit(cache=True)
def _logpdf_vectorized(meanlog: float64, stdlog: float64, x: VecF64) -> VecF64:
    return (  # type: ignore
        -np.log(stdlog)
        - np.log(x)
        - 0.5 * np.log(2.0 * np.pi)
        - 0.5 * ((np.log(x) - meanlog) / stdlog) ** 2
    ).astype(float64)


@njit(cache=True)
def _grad_logpdf_scalar(meanlog: float64, stdlog: float64, x: float64) -> float64:
    return float64(-(1.0 / x) * (1.0 + (np.log(x) - meanlog) / (stdlog**2)))


@njit(cache=True)
def _grad_logpdf_vectorized(meanlog: float64, stdlog: float64, x: VecF64) -> VecF64:
    return (-(1.0 / x) * (1.0 + (np.log(x) - meanlog) / (stdlog**2))).astype(float64)


@njit
def _rvs(
    meanlog: float64, stdlog: float64, size: tuple[int, ...], rng: np.random.Generator
) -> VecF64:
    return rng.lognormal(mean=meanlog, sigma=stdlog, size=size).astype(float64)


class LogNormal(Distribution[float64, VecF64]):
    def __init__(self, mean: float, std: float, random_state: RandomState = None):
        self._meanlog = float64(mean)
        self._stdlog = float64(std)
        self._random_state = random_state

    @overload
    def logpdf(self, x: float64) -> float64: ...
    @overload
    def logpdf(self, x: VecF64) -> VecF64: ...

    def logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        if isinstance(x, float64):
            return cast(float64, _logpdf_scalar(self._meanlog, self._stdlog, x))
        return cast(VecF64, _logpdf_vectorized(self._meanlog, self._stdlog, x))

    @overload
    def grad_logpdf(self, x: float64) -> float64: ...
    @overload
    def grad_logpdf(self, x: VecF64) -> VecF64: ...

    def grad_logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        if isinstance(x, float64):
            return cast(float64, _grad_logpdf_scalar(self._meanlog, self._stdlog, x))
        return cast(VecF64, _grad_logpdf_vectorized(self._meanlog, self._stdlog, x))

    @overload
    def cdf(self, x: float64) -> float64: ...
    @overload
    def cdf(self, x: VecF64) -> VecF64: ...

    def cdf(self, x: float64 | VecF64) -> float64 | VecF64:
        if isinstance(x, float64):
            return float64(ndtr((np.log(x) - self._meanlog) / self._stdlog))
        return ndtr((np.log(x) - self._meanlog) / self._stdlog).astype(float64)

    @overload
    def ppf(self, q: float64) -> float64: ...
    @overload
    def ppf(self, q: VecF64) -> VecF64: ...

    def ppf(self, q: float64 | VecF64) -> float64 | VecF64:
        if isinstance(q, float64):
            return float64(np.exp(self._meanlog + self._stdlog * ndtri(q)))
        return np.exp(self._meanlog + self._stdlog * ndtri(q)).astype(float64)

    def rvs(self, size: Size = 1, random_state: RandomState = None) -> VecF64:
        rng = self._rng(random_state or self._random_state)
        if isinstance(size, int):
            size = (size,)
        return cast(VecF64, _rvs(self._meanlog, self._stdlog, size, rng))

    def __repr__(self) -> str:
        return self.__class__.__name__

    @property
    def support(self) -> Support:
        return Support(
            low=float64(0.0),
            high=float64(np.inf),
            low_inclusive=False,
            high_inclusive=False,
        )

    @property
    def mean(self) -> float64:
        return float64(np.exp(self._meanlog + 0.5 * self._stdlog**2))

    @property
    def var(self) -> float64:
        es2 = np.exp(self._stdlog**2)
        return float64(np.exp(2.0 * self._meanlog) * (es2 - 1.0) * es2)

    @property
    def mode(self) -> float64:
        return float64(np.exp(self._meanlog - self._stdlog**2))
