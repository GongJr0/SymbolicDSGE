from .distribution import Distribution, Size, RandomState, VecF64
from ..support import Support, bounded

import numpy as np
from numpy import float64
from scipy.stats import halfcauchy

from typing import TypedDict, overload, cast


class HalfCauchyParams(TypedDict):
    loc: float
    scale: float
    random_state: RandomState


HALF_CAUCHY_DEFAULTS = HalfCauchyParams(
    loc=0.0,
    scale=1.0,
    random_state=None,
)


class HalfCauchy(Distribution):
    def __init__(self, loc: float, scale: float, random_state: RandomState) -> None:
        self._loc = float64(loc)
        self._scale = float64(scale)
        self._random_state = random_state

        self.dist = halfcauchy(loc=self._loc, scale=self._scale)

    @overload
    def logpdf(self, x: float64) -> float64: ...
    @overload
    def logpdf(self, x: VecF64) -> VecF64: ...

    @bounded
    def logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        return float64(self.dist.logpdf(x))

    @overload
    def grad_logpdf(self, x: float64) -> float64: ...
    @overload
    def grad_logpdf(self, x: VecF64) -> VecF64: ...

    @bounded
    def grad_logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
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

    @property
    def support(self) -> Support:
        return Support(
            float64(self._loc),
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
        return float64(self._loc)
