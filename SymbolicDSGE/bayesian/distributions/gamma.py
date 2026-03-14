from .distribution import (
    Distribution,
    Size,
    RandomState,
    VecF64,
    x_logy_scalar,
    x_logy_vectorized,
)
from ..support import OutOfSupportError, Support

import numpy as np
from numpy import float64
from scipy.special import gammainc, gammaincinv
import math
from numba import njit


from typing import TypedDict, cast, overload


class GammaParams(TypedDict):
    mean: float
    std: float
    random_state: RandomState


GAMMA_DEFAULTS = GammaParams(
    mean=1.0,
    std=1.0,
    random_state=None,
)


@njit(cache=True)
def _logpdf_scalar(
    a: float64, theta: float64, log_norm: float64, x: float64
) -> float64:
    return float64(x_logy_scalar(a - 1.0, x) - x / theta - log_norm)


@njit(cache=True)
def _logpdf_vectorized(
    a: float64, theta: float64, log_norm: float64, x: VecF64
) -> VecF64:
    return (x_logy_vectorized(a - 1.0, x) - x / theta - log_norm).astype(float64)  # type: ignore


@njit(cache=True)
def _grad_logpdf_scalar(a: float64, theta: float64, x: float64) -> float64:
    return float64((a - 1) / x - (1 / theta))


@njit(cache=True)
def _grad_logpdf_vectorized(a: float64, theta: float64, x: VecF64) -> VecF64:
    return ((a - 1) / x - (1 / theta)).astype(float64)


@njit(cache=True)
def _rvs(
    a: float64, theta: float64, size: tuple[int, ...], rng: np.random.Generator
) -> VecF64:
    return rng.gamma(shape=a, scale=theta, size=size).astype(float64)


class Gamma(Distribution[float64, VecF64]):
    def __init__(self, mean: float, std: float, random_state: RandomState) -> None:
        self._mean = float64(mean)
        self._std = float64(std)
        self._a = self.to_shape(mean, std)
        self._theta = self.to_scale(mean, std)
        self._log_norm = float64(
            math.lgamma(float(self._a)) + float(self._a) * math.log(float(self._theta))
        )
        self._random_state = random_state

    @staticmethod
    def to_shape(mean: float, std: float) -> float64:
        return float64(mean / std) ** 2

    @staticmethod
    def to_scale(mean: float, std: float) -> float64:
        return float64(std**2 / mean)

    @overload
    def logpdf(self, x: float64) -> float64: ...
    @overload
    def logpdf(self, x: VecF64) -> VecF64: ...

    def logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)

        if isinstance(x, float64):
            return cast(
                float64, _logpdf_scalar(self._a, self._theta, self._log_norm, x)
            )
        return cast(VecF64, _logpdf_vectorized(self._a, self._theta, self._log_norm, x))

    @overload
    def grad_logpdf(self, x: float64) -> float64: ...
    @overload
    def grad_logpdf(self, x: VecF64) -> VecF64: ...

    def grad_logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        if isinstance(x, float64):
            return cast(float64, _grad_logpdf_scalar(self._a, self._theta, x))
        return cast(VecF64, _grad_logpdf_vectorized(self._a, self._theta, x))

    @overload
    def cdf(self, x: float64) -> float64: ...
    @overload
    def cdf(self, x: VecF64) -> VecF64: ...

    def cdf(self, x: float64 | VecF64) -> float64 | VecF64:
        return float64(gammainc(self._a, x / self._theta))

    @overload
    def ppf(self, q: float64) -> float64: ...
    @overload
    def ppf(self, q: VecF64) -> VecF64: ...

    def ppf(self, q: float64 | VecF64) -> float64 | VecF64:
        return float64(self._theta * gammaincinv(self._a, q))

    def rvs(self, size: Size = 1, random_state: RandomState = None) -> VecF64:
        rng = self._rng(random_state or self._random_state)
        if isinstance(size, int):
            size = (size,)
        return cast(VecF64, _rvs(self._a, self._theta, size, rng))

    def __repr__(self) -> str:
        return self.__class__.__name__

    @property
    def support(self) -> Support:
        return Support(
            low=float64(0.0),
            high=float64(np.inf),
            low_inclusive=True,
            high_inclusive=False,
        )

    @property
    def mean(self) -> float64:
        return self._mean

    @property
    def var(self) -> float64:
        return float64(self._std**2)

    @property
    def mode(self) -> float64:
        return (
            float64(0.0) if self._a <= 1.0 else float64((self._a - 1.0) * self._theta)
        )
