from .distribution import (
    Distribution,
    Size,
    RandomState,
    VecF64,
    log_beta,
    xlog1py_scalar,
    xlog1py_vectorized,
    x_logy_scalar,
    x_logy_vectorized,
)
from ..support import OutOfSupportError, Support

import numpy as np
from numpy import float64
from scipy.special import betainc, betaincinv
from numba import njit


from typing import TypedDict, cast, overload


class BetaParams(TypedDict):
    a: float
    b: float
    random_state: RandomState


BETA_DEFAULTS = BetaParams(
    a=1.0,
    b=1.0,
    random_state=None,
)


@njit(cache=True)
def _logpdf_scalar(a: float64, b: float64, log_norm: float64, x: float64) -> float64:
    return x_logy_scalar(a - 1.0, x) + xlog1py_scalar(b - 1.0, -x) - log_norm  # type: ignore


@njit(cache=True)
def _logpdf_vectorized(a: float64, b: float64, log_norm: float64, x: VecF64) -> VecF64:
    return x_logy_vectorized(a - 1.0, x) + xlog1py_vectorized(b - 1.0, -x) - log_norm  # type: ignore


@njit(cache=True)
def _grad_logpdf_scalar(a: float64, b: float64, x: float64) -> float64:
    return (a - 1.0) / x - (b - 1.0) / (1.0 - x)


@njit(cache=True)
def _grad_logpdf_vectorized(a: float64, b: float64, x: VecF64) -> VecF64:
    return (a - 1.0) / x - (b - 1.0) / (1.0 - x)


@njit(cache=True)
def _rvs(
    a: float64, b: float64, size: tuple[int, ...], rng: np.random.Generator
) -> VecF64:
    return rng.beta(a, b, size=size).astype(float64)


class Beta(Distribution[float64, VecF64]):
    def __init__(self, a: float, b: float, random_state: RandomState = None) -> None:
        self._a = float64(a)
        self._b = float64(b)
        self._log_norm = float64(log_beta(self._a, self._b))
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
            return cast(float64, _logpdf_scalar(self._a, self._b, self._log_norm, x))
        return cast(VecF64, _logpdf_vectorized(self._a, self._b, self._log_norm, x))

    @overload
    def grad_logpdf(self, x: float64) -> float64: ...
    @overload
    def grad_logpdf(self, x: VecF64) -> VecF64: ...

    def grad_logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        if isinstance(x, float64):
            return cast(float64, _grad_logpdf_scalar(self._a, self._b, x))
        return cast(VecF64, _grad_logpdf_vectorized(self._a, self._b, x))

    @overload
    def cdf(self, x: float64) -> float64: ...
    @overload
    def cdf(self, x: VecF64) -> VecF64: ...

    def cdf(self, x: float64 | VecF64) -> float64 | VecF64:
        if isinstance(x, float64):
            if x < 0.0:
                return float64(0.0)
            if x > 1.0:
                return float64(1.0)
            return float64(betainc(self._a, self._b, x))
        return np.where(
            x < 0.0,
            float64(0.0),
            np.where(x > 1.0, float64(1.0), betainc(self._a, self._b, x)),
        )

    @overload
    def ppf(self, q: float64) -> float64: ...
    @overload
    def ppf(self, q: VecF64) -> VecF64: ...

    def ppf(self, q: float64 | VecF64) -> float64 | VecF64:
        return float64(betaincinv(self._a, self._b, q))

    def rvs(self, size: Size = None, random_state: RandomState = None) -> VecF64:
        rng = self._rng_with_fallback(random_state, self._random_state)
        if isinstance(size, int):
            size = (size,)
        return cast(VecF64, _rvs(self._a, self._b, size, rng))

    def __repr__(self) -> str:
        return self.__class__.__name__

    @property
    def support(self) -> Support:
        return Support(float64(0.0), float64(1.0))

    @property
    def mean(self) -> float64:
        return float64(self._a / (self._a + self._b))

    @property
    def var(self) -> float64:
        denom = (self._a + self._b) ** 2 * (self._a + self._b + 1.0)
        return float64((self._a * self._b) / denom)

    @property
    def mode(self) -> float64:
        if self._a > 1.0 and self._b > 1.0:
            return float64((self._a - 1.0) / (self._a + self._b - 2.0))
        if self._a <= 1.0 and self._b > 1.0:
            return float64(0.0)
        if self._a > 1.0 and self._b <= 1.0:
            return float64(1.0)
        raise ValueError(
            "Beta distribution does not have a unique mode for these parameters."
        )
