from .distribution import Distribution, RandomState, Size, VecF64
from ..support import Support, bounded

from numpy import float64
from scipy.stats import truncnorm
from typing import TypedDict, Tuple, overload, cast


class TruncNormParams(TypedDict):
    loc: float
    scale: float
    a: float  # Scalar lower bound (scipy expects standard deviations)
    b: float  # Scalar upper bound (scipy expects standard deviations)
    random_state: RandomState


TRUNCNORM_DEFAULTS = TruncNormParams(
    loc=0.0,
    scale=1.0,
    a=-6.0,  # Effectively unbounded
    b=6.0,
    random_state=None,
)


class TruncNormal(Distribution[float64, VecF64]):
    def __init__(
        self, a: float, b: float, loc: float, scale: float, random_state: RandomState
    ):
        self._loc = float64(loc)
        self._scale = float64(scale)
        self._a_trunc = float64(a)
        self._b_trunc = float64(b)
        self._a, self._b = self._scalar_to_std(loc, scale, a, b)
        self._random_state = random_state

        self.dist = truncnorm(a=self._a, b=self._b, loc=self._loc, scale=self._scale)

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
        # Assume gradient at bounds is approaching from the defined region (+ for lower bound, - for upper bound).
        # This avoids non-finite gradients at bounds but isn't mathematically exact.
        return -(x - self._loc) / self._scale**2

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

    @staticmethod
    def _scalar_to_std(
        loc: float, scale: float, a: float, b: float
    ) -> Tuple[float, float]:
        # Z-transform to get unit std
        return (
            (a - loc) / scale,
            (b - loc) / scale,
        )

    @property
    def support(self) -> Support:
        return Support(
            float64(self._a_trunc),
            float64(self._b_trunc),
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
        mu = self.mean
        hi = self.support.high
        lo = self.support.low

        return float64(min(max(mu, lo), hi))
