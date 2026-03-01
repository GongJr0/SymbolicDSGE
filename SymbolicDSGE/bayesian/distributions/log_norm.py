from .distribution import Distribution, RandomState, Size, VecF64
from ..support import Support, bounded

import numpy as np
from numpy import float64
from scipy.stats import lognorm
from typing import TypedDict, overload, cast


class LogNormalParams(TypedDict):
    s: float  # Shape parameter (standard deviation of the underlying normal distribution)
    loc: float  # Location parameter (mean of the underlying normal distribution)
    scale: float  # Scale parameter (exp(mean) of the underlying normal distribution)
    random_state: RandomState


LOGNORM_DEFAULTS = LogNormalParams(
    s=1.0,
    loc=0.0,
    scale=1.0,
    random_state=None,
)


class LogNormal(Distribution[float64, VecF64]):
    def __init__(self, s: float, loc: float, scale: float, random_state: RandomState):
        self._s = float64(s)
        self._loc = float64(loc)
        self._scale = float64(scale)
        self._random_state = random_state

        self.dist = lognorm(s=self._s, loc=self._loc, scale=self._scale)

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
        return float64(-(1.0 / x) * (1 + (np.log(x) - self.mean) / self.var))

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
        return cast(VecF64, float64(samples))

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
        return float64(self.dist.mean())

    @property
    def var(self) -> float64:
        return float64(self.dist.var())

    @property
    def mode(self) -> float64:
        return float64(np.exp(self._loc - self._s**2))
