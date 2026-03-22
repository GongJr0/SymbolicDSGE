from .distribution import Distribution, RandomState, Size, VecF64, _std_norm_cdf_scalar
from ._as241 import ndtri_as241
from ..support import OutOfSupportError, Support

import math
import numpy as np
from numpy import float64
from numba import njit
from scipy.stats import truncnorm
from typing import TypedDict, Tuple, overload, cast


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


@njit(cache=True)
def _logpdf_scalar(
    x: float64, mean: float64, std: float64, log_norm: float64
) -> float64:
    z = (x - mean) / std
    return float64(-0.5 * z * z - log_norm)


@njit(cache=True)
def _logpdf_vectorized(
    x: VecF64, mean: float64, std: float64, log_norm: float64
) -> VecF64:
    z = (x - mean) / std
    return (-0.5 * z * z - log_norm).astype(float64)


@njit(cache=True)
def _grad_logpdf_scalar(x: float64, mean: float64, std: float64) -> float64:
    return float64(-(x - mean) / std**2)


@njit(cache=True)
def _grad_logpdf_vectorized(x: VecF64, mean: float64, std: float64) -> VecF64:
    return (-(x - mean) / std**2).astype(float64)


@njit(cache=True)
def _rvs(
    mean: float64,
    std: float64,
    a: float64,
    b: float64,
    size: tuple[int, ...],
    rng: np.random.Generator,
) -> VecF64:
    u = rng.uniform(size=size).astype(float64)
    u_flat = u.ravel()
    size_flat = u_flat.shape[0]
    Fa = _std_norm_cdf_scalar(a)
    Fb = _std_norm_cdf_scalar(b)

    z = np.empty((size_flat,), dtype=float64)
    for i in range(size_flat):
        z[i] = ndtri_as241(Fa + u_flat[i] * (Fb - Fa))

    return (mean + std * z).reshape(size)


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
        if isinstance(x, float64):
            return cast(
                float64, _logpdf_scalar(x, self._mean, self._std, self._log_norm)
            )
        return cast(
            VecF64, _logpdf_vectorized(x, self._mean, self._std, self._log_norm)
        )

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
        if isinstance(x, float64):
            return cast(float64, _grad_logpdf_scalar(x, self._mean, self._std))
        return cast(VecF64, _grad_logpdf_vectorized(x, self._mean, self._std))

    @overload
    def cdf(self, x: float64) -> float64: ...
    @overload
    def cdf(self, x: VecF64) -> VecF64: ...

    def cdf(self, x: float64 | VecF64) -> float64 | VecF64:
        z = (x - self._mean) / self._std
        out = truncnorm.cdf(z, a=self._a, b=self._b)
        if isinstance(x, float64):
            return float64(out)
        return np.asarray(out, dtype=float64)

    @overload
    def ppf(self, q: float64) -> float64: ...
    @overload
    def ppf(self, q: VecF64) -> VecF64: ...

    def ppf(self, q: float64 | VecF64) -> float64 | VecF64:
        out = self._mean + self._std * truncnorm.ppf(q, a=self._a, b=self._b)
        if isinstance(q, float64):
            return float64(out)
        return np.asarray(out, dtype=float64)

    def rvs(self, size: Size = (1,), random_state: RandomState = None) -> VecF64:
        rng = self._rng_with_fallback(random_state, self._random_state)
        if isinstance(size, int):
            size = (size,)
        return cast(VecF64, _rvs(self._mean, self._std, self._a, self._b, size, rng))

    def __repr__(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def _scalar_to_std(
        mean: float, std: float, low: float, high: float
    ) -> Tuple[float64, float64]:
        # Z-transform to get unit std
        return (
            float64((low - mean) / std),
            float64((high - mean) / std),
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
