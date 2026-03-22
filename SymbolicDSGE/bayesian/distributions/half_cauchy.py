from .distribution import Distribution, Size, RandomState, VecF64
from ..support import OutOfSupportError, Support

import numpy as np
from numpy import float64
from numba import njit

from typing import TypedDict, cast, overload


class HalfCauchyParams(TypedDict):
    gamma: float
    random_state: RandomState


HALF_CAUCHY_DEFAULTS = HalfCauchyParams(
    gamma=1.0,
    random_state=None,
)


@njit(cache=True)
def _logpdf_scalar(x: float64, gamma: float64) -> float64:
    centered = x / gamma
    return float64(np.log(2.0 / np.pi) - np.log(gamma) - np.log1p(centered**2))


@njit(cache=True)
def _logpdf_vectorized(x: VecF64, gamma: float64) -> VecF64:
    centered = x / gamma
    return (np.log(2.0 / np.pi) - np.log(gamma) - np.log1p(centered**2)).astype(float64)  # type: ignore


@njit(cache=True)
def _grad_logpdf_scalar(x: float64, gamma: float64) -> float64:
    return float64((-2.0 * x) / (gamma**2.0 + x**2.0))


@njit(cache=True)
def _grad_logpdf_vectorized(x: VecF64, gamma: float64) -> VecF64:
    return ((-2.0 * x) / (gamma**2.0 + x**2.0)).astype(float64)


@njit(cache=True)
def _rvs(gamma: float64, size: tuple[int, ...], rng: np.random.Generator) -> VecF64:
    return np.abs(gamma * rng.standard_cauchy(size=size)).astype(float64)


class HalfCauchy(Distribution):
    def __init__(self, gamma: float, random_state: RandomState = None) -> None:
        self._gamma = float64(gamma)
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
            return cast(float64, _logpdf_scalar(x, self._gamma))
        return cast(VecF64, _logpdf_vectorized(x, self._gamma))

    @overload
    def grad_logpdf(self, x: float64) -> float64: ...
    @overload
    def grad_logpdf(self, x: VecF64) -> VecF64: ...

    def grad_logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        if isinstance(x, float64):
            return cast(float64, _grad_logpdf_scalar(x, self._gamma))
        return cast(VecF64, _grad_logpdf_vectorized(x, self._gamma))

    @overload
    def cdf(self, x: float64) -> float64: ...
    @overload
    def cdf(self, x: VecF64) -> VecF64: ...

    def cdf(self, x: float64 | VecF64) -> float64 | VecF64:
        if isinstance(x, float64):
            if x < 0.0:
                return float64(0.0)
            return float64((2.0 / np.pi) * np.arctan(x / self._gamma))
        return np.where(x < 0.0, 0.0, (2.0 / np.pi) * np.arctan(x / self._gamma))

    @overload
    def ppf(self, q: float64) -> float64: ...
    @overload
    def ppf(self, q: VecF64) -> VecF64: ...

    def ppf(self, q: float64 | VecF64) -> float64 | VecF64:
        return self._gamma * np.tan(0.5 * np.pi * q)

    def rvs(self, size: Size = 1, random_state: RandomState = None) -> VecF64:
        rng = self._rng_with_fallback(random_state, self._random_state)
        if isinstance(size, int):
            size = (size,)
        return cast(VecF64, _rvs(self._gamma, size, rng))

    def __repr__(self) -> str:
        return self.__class__.__name__

    @property
    def support(self) -> Support:
        return Support(
            float64(0.0),
            float64(np.inf),
            low_inclusive=True,
            high_inclusive=False,
        )

    @property
    def mean(self) -> float64:
        return float64(np.nan)

    @property
    def var(self) -> float64:
        return float64(np.nan)

    @property
    def mode(self) -> float64:
        return float64(0.0)
