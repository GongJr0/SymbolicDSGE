from ..support import Support

from abc import ABC, abstractmethod
from typing import Tuple, Union, TypeVar, TypeAlias, Generic, overload, cast
from enum import StrEnum
import math

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from numba import njit

NDF = NDArray[float64]
VecF64: TypeAlias = NDArray[np.float64]
MatF64: TypeAlias = NDArray[np.float64]

EventT = TypeVar("EventT", float64, VecF64, MatF64)
BatchT = TypeVar("BatchT", VecF64, MatF64)

Size = Union[int, Tuple[int, ...]]
RandomState = Union[None, int, np.random.Generator, np.random.RandomState]
T = TypeVar("T", float64, NDArray[float64])


class DistributionFamily(StrEnum):
    NORMAL = "normal"
    LOGNORMAL = "log_normal"
    HALFNORMAL = "half_normal"
    TRUNCNORMAL = "trunc_normal"
    HALFCAUCHY = "half_cauchy"
    BETA = "beta"
    GAMMA = "gamma"
    INVGAMMA = "inv_gamma"
    UNIFORM = "uniform"
    LKJCHOL = "lkj_chol"


def _coerce_rng(random_state: RandomState) -> np.random.Generator:
    """Accepts None | int seed | Generator | RandomState and returns a Generator."""
    if random_state is None:
        return np.random.default_rng(0)
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(int(random_state))
    # Legacy RandomState support: wrap by seeding a new Generator from it
    if isinstance(random_state, np.random.RandomState):
        # draw a seed from RandomState deterministically
        # RandomState.randint uses platform-dependent C-long bounds; on Windows
        # this is typically int32, so keep the upper bound int32-safe.
        seed = int(random_state.randint(0, np.iinfo(np.int32).max))
        return np.random.default_rng(seed)
    raise TypeError(f"Unsupported random_state type: {type(random_state)}")


def _scalar_or_array(value: object) -> float64 | VecF64:
    arr = np.asarray(value, dtype=float64)
    if arr.ndim == 0:
        return float64(arr)
    return arr


@njit
def x_logy_scalar(coeff: float64, x: float64) -> float64:
    if coeff == 0.0:
        return float64(0.0)
    return float64(coeff * np.log(x))


@njit
def x_logy_vectorized(coeff: float64, x: VecF64) -> VecF64:
    if coeff == 0.0:
        return np.zeros_like(x, dtype=float64)
    return (coeff * np.log(x)).astype(float64)


@njit
def xlog1py_scalar(coeff: float64, y: float64) -> float64:
    if coeff == 0.0:
        return float64(0.0)
    return float64(coeff * np.log1p(y))


@njit
def xlog1py_vectorized(coeff: float64, y: VecF64) -> VecF64:
    if coeff == 0.0:
        return np.zeros_like(y, dtype=float64)
    return (coeff * np.log1p(y)).astype(float64)


@njit
def log_beta(a: float64, b: float64) -> float64:
    return float64(
        math.lgamma(float(a)) + math.lgamma(float(b)) - math.lgamma(float(a + b))
    )


@njit
def _std_norm_cdf_scalar(x: float64) -> float64:
    return float64(0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0))))


class Distribution(ABC, Generic[EventT, BatchT]):
    @abstractmethod
    def __repr__(self) -> str: ...

    @overload
    def pdf(self, x: EventT) -> float64: ...
    @overload
    def pdf(self, x: BatchT) -> VecF64: ...

    def pdf(self, x: EventT | BatchT) -> float64 | VecF64:
        return float64(np.exp(self.logpdf(x)))

    @overload
    def logpdf(self, x: EventT) -> float64: ...
    @overload
    def logpdf(self, x: BatchT) -> VecF64: ...

    @abstractmethod
    def logpdf(self, x: EventT | BatchT) -> float64 | VecF64: ...

    @overload
    def grad_logpdf(self, x: EventT) -> float64 | MatF64: ...
    @overload
    def grad_logpdf(self, x: BatchT) -> VecF64: ...

    @abstractmethod
    def grad_logpdf(self, x: EventT | BatchT) -> float64 | VecF64: ...

    @overload
    def cdf(self, x: EventT) -> float64 | MatF64: ...
    @overload
    def cdf(self, x: BatchT) -> VecF64: ...

    @abstractmethod
    def cdf(self, x: EventT | BatchT) -> float64 | VecF64: ...

    @overload
    def ppf(self, q: EventT) -> float64 | MatF64: ...
    @overload
    def ppf(self, q: BatchT) -> VecF64: ...

    @abstractmethod
    def ppf(self, q: EventT | BatchT) -> float64 | VecF64: ...

    @abstractmethod
    def rvs(self, size: Size, random_state: RandomState = None) -> BatchT: ...

    @property
    @abstractmethod
    def support(self) -> Support: ...

    @property
    @abstractmethod
    def mean(self) -> EventT: ...

    @property
    @abstractmethod
    def var(self) -> EventT: ...

    @property
    @abstractmethod
    def mode(self) -> EventT: ...

    def is_valid(self, x: EventT | BatchT) -> bool:
        return bool(self.support.contains(x))

    def _rng(self, random_state: RandomState) -> np.random.Generator:
        return _coerce_rng(random_state)
