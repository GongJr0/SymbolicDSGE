from .distribution import Distribution, Size, RandomState, VecF64
from ..support import Support, bounded

from numpy import float64
from scipy.stats import uniform

from typing import TypedDict, Tuple, overload, cast


class UniformParams(TypedDict):
    a: float
    b: float
    random_state: RandomState


UNIFORM_DEFAULTS = UniformParams(
    a=0.0,
    b=1.0,
    random_state=None,
)


class Uniform(Distribution[float64, VecF64]):
    def __init__(self, a: float, b: float, random_state: RandomState) -> None:
        self._a = float64(a)
        self._b = float64(b)
        self._loc, self._scale = self._a_b_to_loc_scale(a, b)
        self._random_state = random_state

        self.dist = uniform(loc=self._loc, scale=self._scale)

    @overload
    def logpdf(self, x: float64) -> float64: ...
    @overload
    def logpdf(self, x: VecF64) -> VecF64: ...

    @bounded
    def logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        return float64(
            self.dist.logpdf(x)
        )  # PDF is invariant in terms of x, this function acts like a bounds checker only

    @overload
    def grad_logpdf(self, x: float64) -> float64: ...
    @overload
    def grad_logpdf(self, x: VecF64) -> VecF64: ...

    @bounded
    def grad_logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        return float64(
            0.0
        )  # PDF is invariant in terms of x, this function acts like a bounds checker only

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
            self._a,
            self._b,
            low_inclusive=True,
            high_inclusive=True,
        )

    @property
    def mean(self) -> float64:
        return float64(self.dist.mean())

    @property
    def var(self) -> float64:
        return float64(self.dist.var())

    @property
    def mode(self) -> float64:
        raise ValueError("Uniform distribution does not have a unique mode.")

    @staticmethod
    def _a_b_to_loc_scale(a: float, b: float) -> Tuple[float64, float64]:
        loc = float64(a)
        scale = float64(b - a)
        return loc, scale
