from .distribution import Distribution, Size, RandomState, VecF64
from ..support import OutOfSupportError, Support

import numpy as np
from numpy import float64
from numba import njit

from typing import TypedDict, cast, overload


class UniformParams(TypedDict):
    low: float
    high: float
    random_state: RandomState


UNIFORM_DEFAULTS = UniformParams(
    low=0.0,
    high=1.0,
    random_state=None,
)


@njit(cache=True)
def _logpdf_scalar(low: float64, high: float64, width: float64, x: float64) -> float64:
    if x < low or x > high:
        return float64(-np.inf)
    return float64(-np.log(width))


@njit(cache=True)
def _logpdf_vectorized(
    low: float64, high: float64, width: float64, x: VecF64
) -> VecF64:
    log_const = float64(-np.log(width))
    return np.where((x < low) | (x > high), -np.inf, log_const).astype(float64)


@njit(cache=True)
def _grad_logpdf_scalar(low: float64, high: float64, x: float64) -> float64:
    if x < low or x > high:
        return float64(-np.inf)
    return float64(0.0)


@njit(cache=True)
def _grad_logpdf_vectorized(low: float64, high: float64, x: VecF64) -> VecF64:
    return np.where((x < low) | (x > high), -np.inf, 0.0).astype(float64)


@njit(cache=True)
def _rvs(
    low: float64, high: float64, size: tuple[int, ...], rng: np.random.Generator
) -> VecF64:
    return rng.uniform(low=low, high=high, size=size).astype(float64)


class Uniform(Distribution[float64, VecF64]):
    def __init__(self, low: float, high: float, random_state: RandomState) -> None:
        self._low = float64(low)
        self._high = float64(high)
        self._width = float64(high - low)
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
            return cast(float64, _logpdf_scalar(self._low, self._high, self._width, x))
        return cast(VecF64, _logpdf_vectorized(self._low, self._high, self._width, x))

    @overload
    def grad_logpdf(self, x: float64) -> float64: ...
    @overload
    def grad_logpdf(self, x: VecF64) -> VecF64: ...

    def grad_logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        if isinstance(x, float64):
            return cast(float64, _grad_logpdf_scalar(self._low, self._high, x))
        return cast(VecF64, _grad_logpdf_vectorized(self._low, self._high, x))

    @overload
    def cdf(self, x: float64) -> float64: ...
    @overload
    def cdf(self, x: VecF64) -> VecF64: ...

    def cdf(self, x: float64 | VecF64) -> float64 | VecF64:
        if isinstance(x, float64):
            if x < self._low:
                return float64(0.0)
            if x > self._high:
                return float64(1.0)
            return float64((x - self._low) / self._width)
        return np.where(
            x < self._low,
            0.0,
            np.where(x > self._high, 1.0, (x - self._low) / self._width),
        )

    @overload
    def ppf(self, q: float64) -> float64: ...
    @overload
    def ppf(self, q: VecF64) -> VecF64: ...

    def ppf(self, q: float64 | VecF64) -> float64 | VecF64:
        if isinstance(q, float64):
            if q < 0.0 or q > 1.0:
                return float64(np.nan)
            return float64(self._low + q * self._width)
        return np.where((0.0 <= q) & (q <= 1.0), self._low + q * self._width, np.nan)

    def rvs(self, size: Size = 1, random_state: RandomState = None) -> VecF64:
        rng = self._rng(random_state or self._random_state)
        if isinstance(size, int):
            size = (size,)
        return cast(VecF64, _rvs(self._low, self._high, size, rng))

    def __repr__(self) -> str:
        return self.__class__.__name__

    @property
    def support(self) -> Support:
        return Support(
            self._low,
            self._high,
            low_inclusive=True,
            high_inclusive=True,
        )

    @property
    def mean(self) -> float64:
        return float64(0.5 * (self._low + self._high))

    @property
    def var(self) -> float64:
        return float64((self._width**2) / 12.0)

    @property
    def mode(self) -> float64:
        raise ValueError("Uniform distribution does not have a unique mode.")
