from .distribution import Distribution, RandomState, Size, VecF64, _scalar_or_array
from ..support import OutOfSupportError, Support

import numpy as np
from numpy import float64
from typing import TypedDict, overload, cast

from scipy.stats import norm


class NormalParameters(TypedDict):
    mean: float
    std: float
    random_state: RandomState


NORM_DEFAULTS = NormalParameters(
    mean=0.0,
    std=1.0,
    random_state=None,
)


class Normal(Distribution[float64, VecF64]):
    def __init__(self, mean: float, std: float, random_state: RandomState = None):
        self._mean = float64(mean)
        self._var = float64(std**2)
        self._random_state = random_state

        std = float64(std)
        self.dist = norm(loc=self._mean, scale=std)

    @overload
    def logpdf(self, x: float64) -> float64: ...
    @overload
    def logpdf(self, x: VecF64) -> VecF64: ...

    def logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        x_arr = np.asarray(x, dtype=float64)
        log_density = (
            -0.5 * np.log(2.0 * np.pi * self._var)
            - 0.5 * ((x_arr - self._mean) ** 2) / self._var
        )
        return _scalar_or_array(log_density)

    @overload
    def grad_logpdf(self, x: float64) -> float64: ...
    @overload
    def grad_logpdf(self, x: VecF64) -> VecF64: ...

    def grad_logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        grad_logpdf = -(x - self.mean) / self.var
        return float64(grad_logpdf)

    @overload
    def cdf(self, x: float64) -> float64: ...
    @overload
    def cdf(self, x: VecF64) -> VecF64: ...

    def cdf(self, x: float64 | VecF64) -> float64 | VecF64:
        return float64(self.dist.cdf(x))

    @overload
    def ppf(self, q: float64) -> float64: ...
    @overload
    def ppf(self, q: VecF64) -> VecF64: ...

    def ppf(self, q: float64 | VecF64) -> float64 | VecF64:
        return float64(self.dist.ppf(q))

    def rvs(self, size: Size = 1, random_state: RandomState = None) -> VecF64:
        rng = self._rng(random_state or self._random_state)
        if isinstance(size, int):
            size = (size,)

        samples = self.dist.rvs(size=size, random_state=rng)
        return cast(VecF64, samples)

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
