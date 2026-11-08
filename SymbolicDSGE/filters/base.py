from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Mapping, TypeVar, ClassVar, Generic

from numpy.typing import NDArray
from numpy import float64

from ..core.solved_model.base import SolvedModel
from .resolvers import FilterMode
from .filter import FilterResult

ResultT = TypeVar("ResultT", bound=FilterResult)


@dataclass
class Filter(ABC, Generic[ResultT]):
    """Abstract base class for filters."""

    mode: ClassVar[FilterMode]

    @abstractmethod
    def filter(
        self,
        model: SolvedModel,
        y: NDArray[float64],
        observables: list[str] | tuple[str] | None = None,
        R: NDArray[float64] | None = None,
        x0: Mapping[str, float] | None = None,
    ) -> ResultT: ...

    @abstractmethod
    def to_spec(self) -> dict: ...

    @classmethod
    @abstractmethod
    def from_spec(cls, spec: dict) -> Filter[ResultT]: ...
