from .distribution import Distribution, Size, RandomState, VecF64
from ..support import Support
from typing import TypedDict, Any, overload, cast

import numpy as np
from numpy import float64

from scipy.stats import halfnorm


class HalfNormalParameters(TypedDict):
    loc: float  # != mean; used like loc + HN(0, 1) * scale
    scale: float  # != std;
    transform: str
    transform_kwargs: dict[str, Any]
    random_state: RandomState


class HalfNormal(Distribution[float64, VecF64]):
    def __init__(self, parameters: HalfNormalParameters):
        self.dist = halfnorm(loc=parameters["loc"], scale=parameters["scale"])

        self._mean = float64(self.dist.mean())
        self._var = float64(self.dist.var())
        self._mode = float64(parameters["loc"])
        self._loc = float64(parameters["loc"])
        self._scale = float64(parameters["scale"])
        self._random_state = parameters["random_state"]

    @overload
    def logpdf(self, x: float64) -> float64: ...
    @overload
    def logpdf(self, x: VecF64) -> VecF64: ...

    def logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        return float64(self.dist.logpdf(x))

    @overload
    def grad_logpdf(self, x: float64) -> float64: ...
    @overload
    def grad_logpdf(self, x: VecF64) -> VecF64: ...

    def grad_logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        return float64(-(x - self.loc) / self.scale**2)

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
        rng = self.rng if random_state is None else self._rng(random_state)
        return cast(VecF64, self.dist.rvs(size=size, random_state=rng))

    @property
    def rng(self) -> np.random.Generator:
        return self._rng(self._random_state)

    @property
    def support(self) -> Support:
        return Support(
            float64(0), float64(np.inf), low_inclusive=True, high_inclusive=False
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
    def loc(self) -> float64:
        return self._loc

    @property
    def scale(self) -> float64:
        return self._scale
