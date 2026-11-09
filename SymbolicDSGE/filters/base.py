from __future__ import annotations

from dataclasses import dataclass, fields
from abc import ABC, abstractmethod
from typing import Any, Mapping, TypeVar, ClassVar, Generic

from numpy.typing import NDArray
from numpy import float64, ndarray

from ..core.solved_model.base import SolvedModel
from .resolvers import FilterMode
from .filter import FilterResult

ResultT = TypeVar("ResultT", bound=FilterResult)


def _normalize(val: Any) -> Any:
    if isinstance(val, ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return [_normalize(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _normalize(v) for k, v in val.items()}
    return val


@dataclass
class Filter(ABC, Generic[ResultT]):
    """Abstract base class for filters."""

    mode: ClassVar[FilterMode]

    @abstractmethod
    def filter(
        self,
        model: SolvedModel,
        y: NDArray[float64],
        observables: list[str] | tuple[str, ...] | None = None,
        R: NDArray[float64] | None = None,
        x0: Mapping[str, float] | None = None,
    ) -> ResultT: ...

    def to_spec(self) -> dict[str, Any]:
        """Convert the filter to a specification dictionary."""
        spec = {f.name: _normalize(getattr(self, f.name)) for f in fields(self)}
        return spec

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> Filter[ResultT]:
        """Create a filter from a specification dictionary."""
        return cls(**spec)
