from .distribution import Distribution, Size, RandomState, VecF64, _scalar_or_array
from ..support import OutOfSupportError, Support
from typing import TypedDict, overload, cast

import numpy as np
from numpy import float64

from scipy.stats import halfnorm


class HalfNormalParameters(TypedDict):
    low: float  # lower bound of support
    scale: float  # != std;
    random_state: RandomState


HALFNORM_DEFAULTS = HalfNormalParameters(
    low=0.0,
    scale=1.0,
    random_state=None,
)


class HalfNormal(Distribution[float64, VecF64]):
    def __init__(self, low: float, scale: float, random_state: RandomState = None):
        self.dist = halfnorm(loc=low, scale=scale)

        self._mean = float64(self.dist.mean())
        self._var = float64(self.dist.var())
        self._mode = float64(low)
        self._low = float64(low)
        self._scale = float64(scale)
        self._random_state = random_state

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
            0.5 * np.log(2.0 / np.pi)
            - np.log(self._scale)
            - 0.5 * ((x_arr - self._low) / self._scale) ** 2
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
        return float64(-(x - self.low) / self.scale**2)

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

    def rvs(self, size: Size = None, random_state: RandomState = None) -> VecF64:
        rng = self._rng(random_state or self._random_state)
        if isinstance(size, int):
            size = (size,)
        samples = self.dist.rvs(size=size, random_state=rng)
        return cast(VecF64, float64(samples))

    def __repr__(self) -> str:
        return self.__class__.__name__

    @property
    def rng(self) -> np.random.Generator:
        return self._rng(self._random_state)

    @property
    def support(self) -> Support:
        return Support(
            self._low,
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
    def low(self) -> float64:
        return self._low

    @property
    def scale(self) -> float64:
        return self._scale
