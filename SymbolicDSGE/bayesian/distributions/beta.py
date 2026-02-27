from .distribution import Distribution, Size, RandomState, VecF64
from ..support import Support, bounded

from numpy import float64
from scipy.stats import beta

from typing import TypedDict, overload, cast


class BetaParams(TypedDict):
    loc: float
    scale: float
    a: float
    b: float
    random_state: RandomState


class Beta(Distribution[float64, VecF64]):
    def __init__(
        self, a: float, b: float, loc: float, scale: float, random_state: RandomState
    ) -> None:
        self._loc = float64(loc)
        self._scale = float64(scale)
        self._a = float64(a)
        self._b = float64(b)
        self._random_state = random_state

        self.dist = beta(loc=loc, scale=scale, a=a, b=b)

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
        return float64(
            (self._a - 1) / (x - self._loc)
            - (self._b - 1) / (self._loc + self._scale - x)
        )

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
        sample = cast(VecF64, self.dist.rvs(size=size, random_state=rng))
        return sample

    @property
    def support(self) -> Support:
        return Support(self._loc, self._loc + self._scale)

    @property
    def mean(self) -> float64:
        return float64(self.dist.mean())

    @property
    def var(self) -> float64:
        return float64(self.dist.var())

    @property
    def mode(self) -> float64:
        return self._loc + self._scale * ((self._a - 1) / (self._a + self._b - 2))
