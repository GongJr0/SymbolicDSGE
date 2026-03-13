from .distribution import Distribution, Size, RandomState, VecF64, _scalar_or_array
from ..support import OutOfSupportError, Support

import numpy as np
from numpy import float64
from scipy.stats import gamma
from scipy.special import gammaln, xlogy

from typing import TypedDict, overload, cast


class GammaParams(TypedDict):
    mean: float
    std: float
    random_state: RandomState


GAMMA_DEFAULTS = GammaParams(
    mean=1.0,
    std=1.0,
    random_state=None,
)


class Gamma(Distribution[float64, VecF64]):
    def __init__(self, mean: float, std: float, random_state: RandomState) -> None:
        self._a = self.to_shape(mean, std)
        self._scale = self.to_scale(mean, std)

        self._loc = float64(0.0)
        self._random_state = random_state

        self.dist = gamma(a=self._a, loc=self._loc, scale=self._scale)

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
        x_arr = np.asarray(x, dtype=float64)
        z = x_arr - self._loc
        log_density = (
            xlogy(self._a - 1.0, z)
            - z / self._scale
            - gammaln(self._a)
            - self._a * np.log(self._scale)
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
        return float64((self._a - 1) / (x - self._loc) - (1 / self._scale))

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
        sample = self.dist.rvs(size=size, random_state=rng)
        return cast(VecF64, sample)

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
        return float64(self.dist.mean())

    @property
    def var(self) -> float64:
        return float64(self.dist.var())

    @property
    def mode(self) -> float64:
        return (self._a - 1) * self._scale + self._loc
