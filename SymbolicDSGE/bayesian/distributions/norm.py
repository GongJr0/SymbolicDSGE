from .distribution import Distribution, RandomState, Size, VecF64
from ..support import OutOfSupportError, Support

import numpy as np
from numpy import float64
from typing import TypedDict, overload, cast
from scipy.special import ndtr, ndtri
from numba import njit


class NormalParameters(TypedDict):
    mean: float
    std: float
    random_state: RandomState


NORM_DEFAULTS = NormalParameters(
    mean=0.0,
    std=1.0,
    random_state=None,
)


@njit(cache=True)
def _logpdf_scalar(mean: float64, var: float64, x: float64) -> float64:
    return float64(
        -0.5 * np.log(2.0 * np.pi * var) - 0.5 * ((x - mean) ** 2) / var,
    )


@njit(cache=True)
def _logpdf_vectorized(mean: float64, var: float64, x: VecF64) -> VecF64:
    return (  # type: ignore
        -0.5 * np.log(2.0 * np.pi * var) - 0.5 * ((x - mean) ** 2) / var
    ).astype(float64)


@njit(cache=True)
def _grad_logpdf_scalar(mean: float64, var: float64, x: float64) -> float64:
    return float64(-(x - mean) / var)


@njit(cache=True)
def _grad_logpdf_vectorized(mean: float64, var: float64, x: VecF64) -> VecF64:
    return (-(x - mean) / var).astype(float64)


@njit(cache=True)
def _rvs(
    mean: float64, std: float64, size: tuple[int, ...], rng: np.random.Generator
) -> VecF64:
    return rng.normal(loc=mean, scale=std, size=size).astype(float64)


class Normal(Distribution[float64, VecF64]):
    def __init__(self, mean: float, std: float, random_state: RandomState = None):
        self._mean = float64(mean)
        self._std = float64(std)
        self._var = float64(std**2)
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
            return cast(float64, _logpdf_scalar(self._mean, self._var, x))
        return cast(VecF64, _logpdf_vectorized(self._mean, self._var, x))

    @overload
    def grad_logpdf(self, x: float64) -> float64: ...
    @overload
    def grad_logpdf(self, x: VecF64) -> VecF64: ...

    def grad_logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        if isinstance(x, float64):
            return cast(float64, _grad_logpdf_scalar(self._mean, self._var, x))
        return cast(VecF64, _grad_logpdf_vectorized(self._mean, self._var, x))

    @overload
    def cdf(self, x: float64) -> float64: ...
    @overload
    def cdf(self, x: VecF64) -> VecF64: ...

    def cdf(self, x: float64 | VecF64) -> float64 | VecF64:
        return float64(ndtr((x - self._mean) / self._std))

    @overload
    def ppf(self, q: float64) -> float64: ...
    @overload
    def ppf(self, q: VecF64) -> VecF64: ...

    def ppf(self, q: float64 | VecF64) -> float64 | VecF64:
        return float64(self._mean + self._std * ndtri(q))

    def rvs(self, size: Size = 1, random_state: RandomState = None) -> VecF64:
        rng = self._rng(random_state or self._random_state)
        if isinstance(size, int):
            size = (size,)
        return cast(VecF64, _rvs(self._mean, self._std, size, rng))

    def __repr__(self) -> str:
        return self.__class__.__name__

    @property
    def rng(self) -> np.random.Generator:
        return self._rng(self._random_state)

    @property
    def support(self) -> Support:
        return Support(
            float64(-np.inf),
            float64(np.inf),
            low_inclusive=False,
            high_inclusive=False,
        )

    @property
    def mean(self) -> float64:
        return self._mean

    @property
    def var(self) -> float64:
        return self._var

    @property
    def mode(self) -> float64:
        return self._mean
