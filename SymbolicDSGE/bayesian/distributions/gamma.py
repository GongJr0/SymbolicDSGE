from .distribution import Distribution, Size, RandomState, VecF64
from ..support import Support, bounded

import numpy as np
from numpy import float64
from scipy.stats import gamma

from typing import TypedDict, overload, cast


class GammaParams(TypedDict):
    a: float
    loc: float
    scale: float
    random_state: RandomState


class Gamma(Distribution[float64, VecF64]):
    def __init__(
        self, a: float, loc: float, scale: float, random_state: RandomState
    ) -> None:
        self._a = float64(a)
        self._loc = float64(loc)
        self._scale = float64(scale)
        self._random_state = random_state

        self.dist = gamma(a=self._a, loc=self._loc, scale=self._scale)

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
