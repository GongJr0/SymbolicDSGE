from sympy.core import parameters

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
    """Distributions for :class:`Prior` objects.
    Each member has a corresponding :class:`Distribution` implementation.

    Attributes
    ----------
    NORMAL : Literal["normal"]
        Normal distribution.
    LOGNORMAL : Literal["log_normal"]
        Log-normal distribution.
    HALFNORMAL : Literal["half_normal"]
        Half-normal distribution.
    TRUNCNORMAL : Literal["trunc_normal"]
        Truncated normal distribution.
    HALFCAUCHY : Literal["half_cauchy"]
        Half-Cauchy distribution.
    BETA : Literal["beta"]
        Beta distribution.
    GAMMA : Literal["gamma"]
        Gamma distribution.
    INVGAMMA : Literal["inv_gamma"]
        Inverse gamma distribution.
    UNIFORM : Literal["uniform"]
        Uniform distribution.
    LKJCHOL : Literal["lkj_chol"]
        LKJ distribution of the lower Cholesky factor of a correlation matrix.
        Must be paired with a :class:`CholeskyCorrTransform` ("cholesky_corr" in :class:`TransformMethod`
        or `make_prior(...,transform="cholesky_corr")`) to produce a valid correlation matrix.

    """

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
    """Distribution base class for all prior families. Any a priori routine uses this as the distribution of the {dist, transform} prior pair."""

    @abstractmethod
    def __repr__(self) -> str: ...

    def to_spec(self) -> tuple[str, dict[str, float]]:
        """Return ``(family, parameters)`` for serialization into a ``PriorSpec``.

        ``parameters`` is keyed exactly as :func:`make_prior` expects, so
        ``make_prior(family, parameters, ...)`` rebuilds an equivalent
        distribution. Subclasses declare their own mapping — stored attributes
        are not uniformly named after the constructor params, so this is not
        introspected generically. The default raises for families without a
        round-trippable spec (e.g. ``lkj_chol``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support spec serialization."
        )

    @overload
    def pdf(self, x: EventT) -> float64: ...
    @overload
    def pdf(self, x: BatchT) -> VecF64: ...

    def pdf(self, x: EventT | BatchT) -> float64 | VecF64:
        """Probability density function (PDF) of the distribution evaluated at x.

        Parameters
        ----------
        x : EventT | BatchT
            Value(s) at which to evaluate the PDF. Can be a single event or a batch of events.

        Returns
        -------
        float64 | VecF64
            Density of the distribution evaluated at the given value(s).

        """
        return float64(np.exp(self.logpdf(x)))

    @overload
    def logpdf(self, x: EventT) -> float64: ...
    @overload
    def logpdf(self, x: BatchT) -> VecF64: ...

    @abstractmethod
    def logpdf(self, x: EventT | BatchT) -> float64 | VecF64:
        """Log density (log PDF) of the distribution evaluated at x.

        Parameters
        ----------
        x : EventT | BatchT
            Value(s) at which to evaluate the log density. Can be a single event or a batch of events.

        Returns
        -------
        float64 | VecF64
            Log density of the distribution evaluated at the given value(s).

        """
        pass

    @overload
    def grad_logpdf(self, x: EventT) -> float64 | MatF64: ...
    @overload
    def grad_logpdf(self, x: BatchT) -> VecF64: ...

    @abstractmethod
    def grad_logpdf(self, x: EventT | BatchT) -> float64 | VecF64:
        """Gradient of the log density (log PDF) of the distribution evaluated at x.

        Parameters
        ----------
        x : EventT | BatchT
            Value(s) at which to evaluate the gradient of the log density. Can be a single event or a batch of events.

        Returns
        -------
        float64 | VecF64
            Gradient of the log density of the distribution evaluated at the given value(s).

        """
        pass

    @overload
    def cdf(self, x: EventT) -> float64 | MatF64: ...
    @overload
    def cdf(self, x: BatchT) -> VecF64: ...

    @abstractmethod
    def cdf(self, x: EventT | BatchT) -> float64 | VecF64:
        """Cumulative distribution function (CDF) of the distribution evaluated at x.

        Parameters
        ----------
        x : EventT | BatchT
            Value(s) at which to evaluate the CDF. Can be a single event or a batch of events.

        Returns
        -------
        float64 | VecF64
            Cumulative probability of the distribution evaluated at the given value(s).

        """
        pass

    @overload
    def ppf(self, q: EventT) -> float64 | MatF64: ...
    @overload
    def ppf(self, q: BatchT) -> VecF64: ...

    @abstractmethod
    def ppf(self, q: EventT | BatchT) -> float64 | VecF64:
        """Percent-point function (inverse of CDF) of the distribution evaluated at q.

        Parameters
        ----------
        q : EventT | BatchT
            Quantile(s) at which to evaluate the inverse CDF. Can be a single quantile or a batch of quantiles.

        Returns
        -------
        float64 | VecF64
            Value(s) corresponding to the given quantile(s) of the distribution.

        """
        pass

    @abstractmethod
    def rvs(self, size: Size, random_state: RandomState = None) -> BatchT:
        """Sample random variates from the distribution.

        Parameters
        ----------
        size : Size
            Shape of the output array. Can be an integer for a single dimension or a tuple for multiple dimensions.
        random_state : RandomState
            Random state or seed for reproducibility of the samples. If None, a default random state is used.

        Returns
        -------
        BatchT
            Samples drawn from the distribution with the specified shape.

        """
        pass

    @property
    @abstractmethod
    def support(self) -> Support:
        """Support of the distribution, defining the valid range of values."""
        pass

    @property
    @abstractmethod
    def mean(self) -> EventT:
        """The mean (expected value) of the distribution."""
        pass

    @property
    @abstractmethod
    def var(self) -> EventT:
        """The variance of the distribution."""
        pass

    @property
    def std(self) -> EventT:
        """The standard deviation of the distribution, the root of :attr:`var`.

        Families without a defined variance report the absence here the same
        way they report it on :attr:`var`.
        """
        return cast(EventT, np.sqrt(self.var))

    @property
    @abstractmethod
    def mode(self) -> EventT:
        """The mode (most probable value) of the distribution."""
        pass

    def is_valid(self, x: EventT | BatchT) -> bool:
        """Evaluate whether the given value(s) are within the support of the distribution.

        Parameters
        ----------
        x : EventT | BatchT
            Value(s) to check for validity against the distribution's support. Can be a single event or a batch of events.

        Returns
        -------
        bool
            Validity of the given value(s) with respect to the distribution's support. True if all values are within the support, False otherwise.

        """
        return bool(self.support.contains(x))

    def _rng(self, random_state: RandomState) -> np.random.Generator:
        return _coerce_rng(random_state)

    def _rng_with_fallback(
        self,
        random_state: RandomState,
        fallback: RandomState,
    ) -> np.random.Generator:
        return _coerce_rng(fallback if random_state is None else random_state)
