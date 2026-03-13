from .distribution import Distribution, Size, RandomState, VecF64, _scalar_or_array
from ..support import OutOfSupportError, Support

import numpy as np
from numpy import float64
from scipy.stats import halfcauchy

from typing import TypedDict, overload, cast


class HalfCauchyParams(TypedDict):
    low: float
    scale: float
    random_state: RandomState


HALF_CAUCHY_DEFAULTS = HalfCauchyParams(
    low=0.0,
    scale=1.0,
    random_state=None,
)


class HalfCauchy(Distribution):
    def __init__(self, low: float, scale: float, random_state: RandomState) -> None:
        self._low = float64(low)
        self._scale = float64(scale)
        self._random_state = random_state

        self.dist = halfcauchy(loc=self._low, scale=self._scale)

    @overload
    def logpdf(self, x: float64) -> float64: ...
    @overload
    def logpdf(self, x: VecF64) -> VecF64: ...

    def logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        x_arr = np.asarray(x, dtype=float64)
        centered = (x_arr - self._low) / self._scale
        log_density = np.log(2.0 / np.pi) - np.log(self._scale) - np.log1p(centered**2)
        return _scalar_or_array(log_density)

    @overload
    def grad_logpdf(self, x: float64) -> float64: ...
    @overload
    def grad_logpdf(self, x: VecF64) -> VecF64: ...

    def grad_logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        return float64((-2 * x) / (self._scale**2 + x**2))

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
        return cast(VecF64, self.dist.rvs(size=size, random_state=rng))

    def __repr__(self) -> str:
        return self.__class__.__name__

    @property
    def support(self) -> Support:
        return Support(
            float64(self._low),
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
        return float64(self._low)
