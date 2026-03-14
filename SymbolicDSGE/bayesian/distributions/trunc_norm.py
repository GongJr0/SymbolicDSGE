from .distribution import Distribution, RandomState, Size, VecF64, _std_norm_cdf_scalar
from ..support import OutOfSupportError, Support

import math
import numpy as np
from numpy import float64
from scipy.stats import truncnorm
from typing import TypedDict, Tuple, overload


class TruncNormParams(TypedDict):
    low: float
    high: float
    mean: float
    std: float
    random_state: RandomState


TRUNCNORM_DEFAULTS = TruncNormParams(
    low=-6.0,  # Effectively unbounded
    high=6.0,
    mean=0.0,
    std=1.0,
    random_state=None,
)


class TruncNormal(Distribution[float64, VecF64]):
    def __init__(
        self,
        low: float,
        high: float,
        mean: float,
        std: float,
        random_state: RandomState,
    ):
        self._mean = float64(mean)
        self._std = float64(std)
        self._low_trunc = float64(low)
        self._high_trunc = float64(high)
        self._a, self._b = self._scalar_to_std(mean, std, low, high)
        z = _std_norm_cdf_scalar(float64(self._b)) - _std_norm_cdf_scalar(
            float64(self._a)
        )
        self._log_norm = float64(
            math.log(float(self._std))
            + 0.5 * math.log(2.0 * math.pi)
            + math.log(float(z))
        )
        self._random_state = random_state

    @overload
    def logpdf(self, x: float64) -> float64: ...
    @overload
    def logpdf(self, x: VecF64) -> VecF64: ...

    def logpdf(self, x: float64 | VecF64) -> float64 | VecF64:
        support = self.support
        if not support.contains(x):
            raise OutOfSupportError(x, support)
        z = (x - self._mean) / self._std
        return float64(-0.5 * z * z - self._log_norm)

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
        return float64(-(x - self._mean) / self._std**2)

    @overload
    def cdf(self, x: float64) -> float64: ...
    @overload
    def cdf(self, x: VecF64) -> VecF64: ...

    def cdf(self, x: float64 | VecF64) -> float64 | VecF64:
        z = (x - self._mean) / self._std
        return float64(truncnorm.cdf(z, a=self._a, b=self._b))

    @overload
    def ppf(self, q: float64) -> float64: ...
    @overload
    def ppf(self, q: VecF64) -> VecF64: ...

    def ppf(self, q: float64 | VecF64) -> float64 | VecF64:
        return float64(
            self._mean + self._std * float64(truncnorm.ppf(q, a=self._a, b=self._b)),
        )

    def rvs(self, size: Size = (1,), random_state: RandomState = None) -> VecF64:
        rng = self._rng(random_state or self._random_state)
        if isinstance(size, int):
            size = (size,)
        return self._mean + self._std * truncnorm.rvs(
            size=size,  # type: ignore
            random_state=rng,
            a=self._a,
            b=self._b,
        )

    def __repr__(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def _scalar_to_std(
        mean: float, std: float, low: float, high: float
    ) -> Tuple[float, float]:
        # Z-transform to get unit std
        return (
            (low - mean) / std,
            (high - mean) / std,
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
        return float64(self._mean + self._std * truncnorm.mean(a=self._a, b=self._b))

    @property
    def var(self) -> float64:
        return float64((self._std**2) * truncnorm.var(a=self._a, b=self._b))

    @property
    def mode(self) -> float64:
        mu = self._mean
        hi = self.support.high
        lo = self.support.low

        return float64(min(max(mu, lo), hi))
