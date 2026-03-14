from .distribution import Distribution, Size, RandomState, VecF64
from ..support import OutOfSupportError, Support

import numpy as np
from numpy import float64
from scipy.special import gammaincc, gammainccinv
import math
from numba import njit

from typing import TypedDict, cast, overload


class InvGammaParams(TypedDict):
    mean: float
    std: float
    random_state: RandomState


INVGAMMA_DEFAULTS = InvGammaParams(
    mean=1.0,
    std=1.0,
    random_state=None,
)


@njit(cache=True)
def _logpdf_scalar(
    a: float64, beta: float64, log_prefactor: float64, x: float64
) -> float64:
    return float64(log_prefactor - (a + 1.0) * math.log(float(x)) - beta / x)


@njit(cache=True)
def _logpdf_vectorized(
    a: float64, beta: float64, log_prefactor: float64, x: VecF64
) -> VecF64:
    return (log_prefactor - (a + 1.0) * np.log(x) - beta / x).astype(float64)


@njit(cache=True)
def _grad_logpdf_scalar(a: float64, beta: float64, x: float64) -> float64:
    return float64(beta / (x * x) - (a + 1.0) / x)


@njit(cache=True)
def _grad_logpdf_vectorized(a: float64, beta: float64, x: VecF64) -> VecF64:
    return (beta / (x * x) - (a + 1.0) / x).astype(float64)


@njit(cache=True)
def _rvs(
    a: float64, beta: float64, size: tuple[int, ...], rng: np.random.Generator
) -> VecF64:
    gamma_samples = rng.gamma(shape=a, scale=1.0, size=size)
    return beta / gamma_samples


class InvGamma(Distribution[float64, VecF64]):
    def __init__(self, mean: float, std: float, random_state: RandomState) -> None:
        self._mean = float64(mean)
        self._std = float64(std)
        self._a = self.to_shape(mean, std)
        self._beta = self.to_scale(mean, std)
        self._log_prefactor = float64(
            float(self._a) * math.log(float(self._beta)) - math.lgamma(float(self._a))
        )
        self._random_state = random_state

    @staticmethod
    def to_shape(mean: float, std: float) -> float64:
        ratio = float64(mean / std)
        return float64(2.0 + ratio**2)

    @classmethod
    def to_scale(cls, mean: float, std: float) -> float64:
        shape = cls.to_shape(mean, std)
        return float64(mean) * float64(shape - 1.0)

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
                float64, _logpdf_scalar(self._a, self._beta, self._log_prefactor, x)
            )
        return cast(
            VecF64, _logpdf_vectorized(self._a, self._beta, self._log_prefactor, x)
        )

    @overload
    def grad_logpdf(self, x: float64) -> float64: ...
    @overload
    def grad_logpdf(self, x: VecF64) -> VecF64: ...

    def grad_logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        if isinstance(x, float64):
            return cast(float64, _grad_logpdf_scalar(self._a, self._beta, x))
        return cast(VecF64, _grad_logpdf_vectorized(self._a, self._beta, x))

    @overload
    def cdf(self, x: float64) -> float64: ...
    @overload
    def cdf(self, x: VecF64) -> VecF64: ...

    def cdf(self, x: float64 | VecF64) -> float64 | VecF64:
        return float64(gammaincc(self._a, self._beta / x))

    @overload
    def ppf(self, q: float64) -> float64: ...
    @overload
    def ppf(self, q: VecF64) -> VecF64: ...

    def ppf(self, q: float64 | VecF64) -> float64 | VecF64:
        return float64(self._beta / gammainccinv(self._a, q))

    def rvs(self, size: Size = 1, random_state: RandomState = None) -> VecF64:
        rng = self._rng(random_state or self._random_state)
        if isinstance(size, int):
            size = (size,)

        return cast(VecF64, _rvs(self._a, self._beta, size, rng))

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
        return self._mean

    @property
    def var(self) -> float64:
        return float64(self._std**2)

    @property
    def mode(self) -> float64:
        return float64(self._beta / (self._a + 1.0))
