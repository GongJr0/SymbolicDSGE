from ..support import Support

from abc import ABC, abstractmethod
from numpy import float64
from numpy.typing import NDArray
import numpy as np

from typing import Tuple, Union, TypeVar, Generic, cast

NDF = NDArray[float64]
FLOAT_VEC_SCA = Union[float64, NDArray[float64]]
Size = Union[int, Tuple[int, ...]]
RandomState = Union[None, int, np.random.Generator, np.random.RandomState]

X = TypeVar("X", float64, NDArray[float64])  # input type
Y = TypeVar("Y", float64, NDArray[float64])  # output type


def _get_rng(random_state: RandomState = None) -> np.random.Generator:
    if random_state is None:
        return np.random.default_rng()
    elif isinstance(random_state, int):
        return np.random.default_rng(random_state)
    elif isinstance(random_state, np.random.Generator):
        return random_state
    elif isinstance(random_state, np.random.RandomState):
        return np.random.default_rng(random_state.randint(0, 2**32 - 1))
    else:
        raise ValueError("Invalid random state provided.")


class Distribution(ABC, Generic[X, Y]):

    # Defined Shared Methods
    def is_valid(self, x: X) -> bool:
        return self.support.contains(x)

    def _rng(self, random_state: RandomState = None) -> np.random.Generator:
        return _get_rng(random_state)

    def pdf(self, x: X) -> Y:
        return cast(Y, np.exp(self.logpdf(x)))

    # Abstract Methods
    @abstractmethod
    def logpdf(self, x: X) -> Y:
        pass

    @abstractmethod
    def grad_logpdf(self, x: X) -> Y:
        pass

    @abstractmethod
    def cdf(self, x: X) -> Y:
        pass

    @abstractmethod
    def ppf(self, q: X) -> Y:
        pass

    @abstractmethod
    def rvs(self, size: Size = 1, random_state: RandomState = None) -> Y:
        pass

    @property
    @abstractmethod
    def support(self) -> Support:
        pass

    @property
    @abstractmethod
    def mean(self) -> Y:
        pass

    @property
    @abstractmethod
    def var(self) -> Y:
        pass

    @property
    @abstractmethod
    def mode(self) -> Y:
        pass
