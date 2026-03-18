from .distribution import Distribution, Size, RandomState, VecF64
from ..support import OutOfSupportError, Support
from typing import TypedDict, cast, overload, Callable

import numpy as np
from numpy import float64
from scipy.special import erf, erfinv
from numba import njit


class HalfNormalParameters(TypedDict):
    std: float
    random_state: RandomState


HALFNORM_DEFAULTS = HalfNormalParameters(
    std=1.0,
    random_state=None,
)


@njit(cache=True)
def _logpdf_scalar(x: float64, std: float64) -> float64:
    if x < 0.0:
        return float64(-np.inf)
    return float64(0.5 * np.log(2.0 / np.pi) - np.log(std) - 0.5 * (x / std) ** 2)


@njit(cache=True)
def _logpdf_vectorized(x: VecF64, std: float64) -> VecF64:
    return np.where(
        x < 0.0,
        float64(-np.inf),
        (0.5 * np.log(2.0 / np.pi) - np.log(std) - 0.5 * (x / std) ** 2).astype(
            float64
        ),
    )


@njit(cache=True)
def _grad_logpdf_scalar(x: float64, std: float64) -> float64:
    if x < 0.0:
        return float64(0.0)
    return float64(-x / std**2)


@njit(cache=True)
def _grad_logpdf_vectorized(x: VecF64, std: float64) -> VecF64:
    return np.where(x < 0.0, float64(0.0), float64(-x / std**2)).astype(float64)


@njit(cache=True)
def _rvs(std: float64, size: tuple[int, ...], rng: np.random.Generator) -> VecF64:
    return np.abs(rng.normal(loc=0.0, scale=std, size=size)).astype(float64)


class HalfNormal(Distribution[float64, VecF64]):
    def __init__(self, std: float, random_state: RandomState = None):
        self._std = float64(std)
        self._mean = float64(self._std * np.sqrt(2.0 / np.pi))
        self._var = float64(self._std**2 * (1.0 - 2.0 / np.pi))
        self._mode = float64(0.0)
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
            return cast(float64, _logpdf_scalar(x, self._std))
        return cast(VecF64, _logpdf_vectorized(x, self._std))

    @overload
    def grad_logpdf(self, x: float64) -> float64: ...
    @overload
    def grad_logpdf(self, x: VecF64) -> VecF64: ...

    def grad_logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        if isinstance(x, float64):
            return cast(float64, _grad_logpdf_scalar(x, self._std))
        return cast(VecF64, _grad_logpdf_vectorized(x, self._std))

    @overload
    def cdf(self, x: float64) -> float64: ...
    @overload
    def cdf(self, x: VecF64) -> VecF64: ...

    def cdf(self, x: float64 | VecF64) -> float64 | VecF64:
        if isinstance(x, float64):
            if x < 0.0:
                return float64(0.0)
            return float64(erf(x / (self._std * np.sqrt(2.0))))
        return np.where(
            x < 0.0, float64(0.0), erf(x / (self._std * np.sqrt(2.0)))
        ).astype(float64)

    @overload
    def ppf(self, q: float64) -> float64: ...
    @overload
    def ppf(self, q: VecF64) -> VecF64: ...

    def ppf(self, q: float64 | VecF64) -> float64 | VecF64:
        if isinstance(q, float64):
            return float64(self._std * np.sqrt(2.0) * erfinv(q))
        return cast(VecF64, self._std * np.sqrt(2.0) * erfinv(q)).astype(float64)

    def rvs(self, size: Size = (1,), random_state: RandomState = None) -> VecF64:
        rng = self._rng(random_state or self._random_state)
        if isinstance(size, int):
            size = (size,)
        return cast(VecF64, _rvs(self._std, size, rng))

    def __repr__(self) -> str:
        return self.__class__.__name__

    @property
    def rng(self) -> np.random.Generator:
        return self._rng(self._random_state)

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
        return self._mean

    @property
    def var(self) -> float64:
        return self._var

    @property
    def mode(self) -> float64:
        return self._mode

    @property
    def std(self) -> float64:
        return self._std
