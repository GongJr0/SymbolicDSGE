from .distribution import Distribution, RandomState, Size, VecF64
from ..support import OutOfSupportError, Support

from numpy import float64
from scipy.stats import truncnorm
from typing import TypedDict, Tuple, overload, cast


class TruncNormParams(TypedDict):
    loc: float
    scale: float
    low: float  # Scalar lower bound (scipy expects standard deviations)
    high: float  # Scalar upper bound (scipy expects standard deviations)
    random_state: RandomState


TRUNCNORM_DEFAULTS = TruncNormParams(
    loc=0.0,
    scale=1.0,
    low=-6.0,  # Effectively unbounded
    high=6.0,
    random_state=None,
)


class TruncNormal(Distribution[float64, VecF64]):
    def __init__(
        self,
        low: float,
        high: float,
        loc: float,
        scale: float,
        random_state: RandomState,
    ):
        self._loc = float64(loc)
        self._scale = float64(scale)
        self._low_trunc = float64(low)
        self._high_trunc = float64(high)
        self._a, self._b = self._scalar_to_std(loc, scale, low, high)
        self._random_state = random_state

        self.dist = truncnorm(a=self._a, b=self._b, loc=self._loc, scale=self._scale)

    @overload
    def logpdf(self, x: float64) -> float64: ...
    @overload
    def logpdf(self, x: VecF64) -> VecF64: ...

    def logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        return float64(self.dist.logpdf(x))

    @overload
    def grad_logpdf(self, x: float64) -> float64: ...
    @overload
    def grad_logpdf(self, x: VecF64) -> VecF64: ...

    def grad_logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
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

    def __repr__(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def _scalar_to_std(
        loc: float, scale: float, low: float, high: float
    ) -> Tuple[float, float]:
        # Z-transform to get unit std
        return (
            (low - loc) / scale,
            (high - loc) / scale,
        )

    @property
    def support(self) -> Support:
        return Support(
            float64(self._low_trunc),
            float64(self._high_trunc),
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
