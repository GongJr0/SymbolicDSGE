from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Literal, Mapping, Protocol, Union

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from .._diag_tests.result import MCResult, TestResult
from ..core.shock_generators import Shock
from ..core.solved_model import SolvedModel
from ..kalman.filter import FilterResult
from ..regression.ols import MCRegressionResult, OLSResult, Status

NDF = NDArray[float64]
NDB = NDArray[np.bool_]
ShockValue = Union[Shock, Callable[[float | NDF], NDF], NDF]
ShockMapping = Mapping[str, ShockValue]
SeedIncrement = Union[int, Literal["auto"]]


class OpType(StrEnum):
    DATAGEN = "datagen"
    TRANSFORM = "transform"
    FILTER = "filter"
    TEST = "test"
    REGRESSION = "regression"
    POSTPROC = "postproc"


@dataclass(frozen=True)
class MCData:
    """Standard data payload generated for one Monte Carlo replication."""

    states: NDF | None = None
    observables: NDF | None = None
    n_exog: int = -1
    raw: Mapping[str, NDF] = field(default_factory=dict)
    observable_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataGenReturn:
    """Legacy simulation-data container kept for compatibility."""

    state_mat: NDF | None
    obs_mat: NDF | None
    n_exog: int


@dataclass
class MCContext:
    rep_idx: int
    reference: SolvedModel
    dgp: SolvedModel | None
    data: MCData | None = None
    payloads: dict[str, Any] = field(default_factory=dict)
    results: dict[str, TestResult] = field(default_factory=dict)
    regressions: dict[str, OLSResult] = field(default_factory=dict)

    def require_data(self) -> MCData:
        if self.data is None:
            raise ValueError(
                "MC context has no generated data. Add a DATAGEN step first."
            )
        return self.data

    def require_payload(self, key: str) -> Any:
        if key not in self.payloads:
            raise KeyError(f"MC context payload '{key}' is not available.")
        return self.payloads[key]


class DataGenOp(Protocol):
    def __call__(
        self,
        *,
        reference: SolvedModel,
        dgp: SolvedModel | None,
        rep_idx: int,
        **kwargs: Any,
    ) -> MCData: ...


class ContextOp(Protocol):
    def __call__(
        self,
        *,
        context: MCContext,
        reference: SolvedModel,
        dgp: SolvedModel | None,
        rep_idx: int,
        **kwargs: Any,
    ) -> Any: ...


class FilterOp(Protocol):
    def __call__(
        self,
        *,
        context: MCContext,
        reference: SolvedModel,
        dgp: SolvedModel | None,
        rep_idx: int,
        **kwargs: Any,
    ) -> FilterResult: ...


class TestOp(Protocol):
    def __call__(
        self,
        *,
        context: MCContext,
        reference: SolvedModel,
        dgp: SolvedModel | None,
        rep_idx: int,
        **kwargs: Any,
    ) -> TestResult: ...


class RegressionOp(Protocol):
    def __call__(
        self,
        *,
        context: MCContext,
        reference: SolvedModel,
        dgp: SolvedModel | None,
        rep_idx: int,
        **kwargs: Any,
    ) -> OLSResult: ...


@dataclass(frozen=True)
class MCStep:
    name: str
    op_type: OpType
    func: Callable[..., Any]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    store_key: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MCStep name must be non-empty.")
        object.__setattr__(self, "op_type", OpType(self.op_type))
        object.__setattr__(self, "kwargs", dict(self.kwargs))

    @property
    def output_key(self) -> str:
        return self.store_key if self.store_key is not None else self.name


@dataclass(frozen=True)
class MCFailure:
    rep_idx: int
    step_name: str
    error_type: str
    message: str


@dataclass(frozen=True)
class MCPipelineResult:
    n_rep: int
    n_successful: int
    test_summaries: Mapping[str, MCResult]
    test_results: Mapping[str, tuple[TestResult, ...]] | None
    payloads: tuple[Mapping[str, Any], ...] | None
    contexts: tuple[MCContext, ...] | None
    failures: tuple[MCFailure, ...] = ()
    regression_summaries: Mapping[str, MCRegressionResult] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return len(self.failures) == 0

    @property
    def statistic_traces(self) -> Mapping[str, NDF]:
        return {
            name: summary.statistic_trace
            for name, summary in self.test_summaries.items()
        }

    @property
    def pval_traces(self) -> Mapping[str, NDF]:
        return {
            name: summary.pval_trace for name, summary in self.test_summaries.items()
        }

    @property
    def rejection_traces(self) -> Mapping[str, NDB]:
        return {
            name: np.asarray(summary.pval_trace < summary.alpha, dtype=bool)
            for name, summary in self.test_summaries.items()
        }

    @property
    def coefficient_traces(self) -> Mapping[str, NDF]:
        return {
            name: summary.coef_trace
            for name, summary in self.regression_summaries.items()
        }

    @property
    def regression_status_traces(self) -> Mapping[str, tuple[Status, ...]]:
        return {
            name: summary.status_trace
            for name, summary in self.regression_summaries.items()
        }
