from ..support import Support

from abc import ABC, abstractmethod
from typing import Tuple, Union, TypeVar, TypeAlias, Generic, overload

import numpy as np
from numpy import float64
from numpy.typing import NDArray

NDF = NDArray[float64]
VecF64: TypeAlias = NDArray[np.float64]
MatF64: TypeAlias = NDArray[np.float64]

EventT = TypeVar("EventT", float64, VecF64, MatF64)
BatchT = TypeVar("BatchT", VecF64, MatF64)

Size = Union[int, Tuple[int, ...]]
RandomState = Union[None, int, np.random.Generator, np.random.RandomState]
T = TypeVar("T", float64, NDArray[float64])


def _coerce_rng(random_state: RandomState) -> np.random.Generator:
    """Accepts None | int seed | Generator | RandomState and returns a Generator."""
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(int(random_state))
    # Legacy RandomState support: wrap by seeding a new Generator from it
    if isinstance(random_state, np.random.RandomState):
        # draw a seed from RandomState deterministically
        seed = int(random_state.randint(0, 2**32 - 1))
        return np.random.default_rng(seed)
    raise TypeError(f"Unsupported random_state type: {type(random_state)}")


class Distribution(ABC, Generic[EventT, BatchT]):
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
    def cdf(self, x: EventT) -> float64: ...
    @overload
    def cdf(self, x: BatchT) -> VecF64: ...

    @abstractmethod
    def cdf(self, x: EventT | BatchT) -> float64 | VecF64: ...

    @overload
    def ppf(self, q: EventT) -> float64: ...
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
