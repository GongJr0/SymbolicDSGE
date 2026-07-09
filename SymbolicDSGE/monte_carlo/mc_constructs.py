from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Literal, Mapping, Protocol, Union, NamedTuple

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from .._diag_tests.result import MCResult, TestResult
from .._diag_tests.status import TestStatus
from ..core.shock_generators import Shock
from ..core.solved_model import SolvedModel
from ..kalman.filter import FilterRawResult, UnscentedFilterRawResult
from ..regression.enums import RegressionStatus
from ..regression.ols import MCRegressionResult
from ..regression.result import RegressionResult
from .custom_op import PandasCustomFunc

NDF = NDArray[float64]
NDB = NDArray[np.bool_]
ShockValue = Union[Shock, Callable[[float | NDF], NDF], NDF]
ShockMapping = Mapping[str, ShockValue]
SeedIncrement = Union[int, Literal["auto"]]

MC_DATA_SOURCE_FIELDS: tuple[str, ...] = ("states", "observables", "raw")
DYNAMIC_SOURCE_FIELDS: tuple[str, ...] = ("payload",)

FILTER_RAW_SOURCE_FIELDS: tuple[str, ...] = FilterRawResult._fields
UNSCENTED_FILTER_RAW_SOURCE_FIELDS: tuple[str, ...] = UnscentedFilterRawResult._fields
UNSCENTED_ONLY_FILTER_SOURCE_FIELDS: tuple[str, ...] = tuple(
    field
    for field in UNSCENTED_FILTER_RAW_SOURCE_FIELDS
    if field not in FILTER_RAW_SOURCE_FIELDS
)
FILTER_SOURCE_FIELDS: tuple[str, ...] = (
    FILTER_RAW_SOURCE_FIELDS + UNSCENTED_ONLY_FILTER_SOURCE_FIELDS
)

SOURCE_FIELDS: tuple[str, ...] = (
    MC_DATA_SOURCE_FIELDS
    + DYNAMIC_SOURCE_FIELDS
    + FILTER_RAW_SOURCE_FIELDS
    + UNSCENTED_ONLY_FILTER_SOURCE_FIELDS
)
SOURCE_FIELD_INDEX: dict[str, int] = {
    field: index for index, field in enumerate(SOURCE_FIELDS)
}
SOURCE_INDEX_FIELD: tuple[str, ...] = SOURCE_FIELDS

if len(SOURCE_FIELD_INDEX) != len(SOURCE_FIELDS) or len(
    set(SOURCE_FIELD_INDEX.values())
) != len(SOURCE_FIELDS):
    raise RuntimeError("MC source field indices must be unique.")

# Array-valued sources currently exposed to MC operations and the catalogue.
ARRAY_SOURCE_FIELDS: tuple[str, ...] = (
    "states",
    "observables",
    "x_pred",
    "x_filt",
    "x1_pred",
    "x2_pred",
    "x1_filt",
    "x2_filt",
    "y_pred",
    "y_filt",
    "innov",
    "std_innov",
    "eps_hat",
)


class OpType(StrEnum):
    DATAGEN = "datagen"
    TRANSFORM = "transform"
    FILTER = "filter"
    TEST = "test"
    REGRESSION = "regression"
    POSTPROC = "postproc"


class MCData(NamedTuple):
    """One Monte Carlo replication's data payload.

    Produced by a DATAGEN step and exposed to per-replication ops as
    ``context.data`` (via ``context.require_data()``). Fields:

    - ``states``: ``(T, n_state)`` latent state matrix, or None.
    - ``observables``: ``(T, k)`` observable matrix, or None.
    - ``n_exog``: number of exogenous shocks (-1 if unknown).
    - ``raw``: mapping of named series (each model variable, plus "_X" = states).
    - ``observable_names``: column names for ``observables``.
    """

    states: NDF | None = None
    observables: NDF | None = None
    n_exog: int = -1
    raw: Mapping[str, NDF] = field(default_factory=dict)
    observable_names: tuple[str, ...] = ()


@dataclass
class MCContext:
    """Per-replication state handed to every Monte Carlo op.

    One ``MCContext`` exists per replication; ops read the generated data and
    prior results from it and write their outputs back. Fields:

    - ``rep_idx``: 0-based replication index.
    - ``reference`` / ``dgp``: the reference and data-generating ``SolvedModel``s.
    - ``data``: this replication's :class:`MCData` (None until a DATAGEN step
      runs; prefer ``require_data()``).
    - ``payloads``: transform outputs keyed by step name (see
      ``require_payload()``).
    - ``results`` / ``regressions``: test / regression results by step name.
    """

    rep_idx: int
    reference: SolvedModel
    dgp: SolvedModel | None
    data: MCData | None = None
    payloads: dict[str, Any] = field(default_factory=dict)
    results: dict[str, TestResult] = field(default_factory=dict)
    regressions: dict[str, RegressionResult] = field(default_factory=dict)

    def require_data(self) -> MCData:
        """Return ``data``, raising if no DATAGEN step has populated it yet."""
        if self.data is None:
            raise ValueError(
                "MC context has no generated data. Add a DATAGEN step first."
            )
        return self.data

    def require_payload(self, key: str) -> Any:
        """Return the payload stored by transform step ``key``, raising if absent."""
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
    ) -> FilterRawResult | UnscentedFilterRawResult: ...


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
    ) -> RegressionResult: ...


@dataclass(frozen=True)
class MCStep:
    name: str
    op_type: OpType
    func: Callable[..., Any]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    store_key: str | None = None
    #: Catalog step kind (e.g. ``"wald"``, ``"standardize"``, ``"simulation"``)
    #: or ``"custom"`` for user-supplied ops. Stamped by the step factories;
    #: lets a live ``MCPipeline`` be compiled back to a serializable
    #: ``PipelineSpec`` without a ``func``-to-kind reverse map. ``None`` only for
    #: hand-built steps that bypassed the factories.
    step_type: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MCStep name must be non-empty.")
        object.__setattr__(self, "op_type", OpType(self.op_type))
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        # The pandas namespace is a post-loop-only privilege: a PandasCustomFunc
        # in a per-rep step would reference pandas inside the replication loop,
        # which the looser contract is not meant to sanction.
        if (
            isinstance(self.func, PandasCustomFunc)
            and self.op_type is not OpType.POSTPROC
        ):
            raise ValueError(
                f"MCStep {self.name!r}: a PandasCustomFunc is only allowed in a "
                "post-loop (POSTPROC) step, not a "
                f"{self.op_type.value!r} step."
            )

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
class MCMeta:
    n_rep: int

    payloads_retained: bool
    test_results_retained: bool
    contexts_retained: bool

    #: Wall-clock seconds of the replication loop alone; the basis for ``it_s``.
    #: Post-loop aggregation and postproc are excluded (see ``postproc_elapsed_s``).
    elapsed_s: float = 0.0
    #: Per-replication step timings (postproc excluded; see ``postproc_elapsed_s``).
    step_elapsed_s: Mapping[str, float] = field(default_factory=dict)
    step_counts: Mapping[str, int] = field(default_factory=dict)
    step_failures: Mapping[str, int] = field(default_factory=dict)
    #: Wall-clock seconds per post-loop (``OpType.POSTPROC``) step. Postproc runs
    #: once, so it is reported as runtime only, never folded into the it/s rates.
    postproc_elapsed_s: Mapping[str, float] = field(default_factory=dict)

    failed_steps: dict[str, int] = field(default_factory=dict)
    failed_postprocs: set[str] = field(default_factory=set)

    @property
    def it_s(self) -> float:
        return _iterations_per_second(self.n_rep, self.elapsed_s)

    @property
    def step_it_s(self) -> Mapping[str, float]:
        return {
            name: _iterations_per_second(
                self.step_counts.get(name, 0),
                elapsed_s,
            )
            for name, elapsed_s in self.step_elapsed_s.items()
        }

    @property
    def postproc_total_s(self) -> float:
        """Total wall-clock seconds spent in the post-loop phase."""
        return sum(self.postproc_elapsed_s.values())

    @property
    def steps_success(self) -> bool:
        """Whether all per-replication steps succeeded (no failures recorded)."""
        return self.failed_steps == {}

    @property
    def postproc_success(self) -> bool:
        """Whether all post-loop steps succeeded (no failures recorded)."""
        return self.failed_postprocs == set()


@dataclass(frozen=True)
class MCPipelineResult:
    meta: MCMeta
    n_rep: int
    n_successful: int
    test_summaries: Mapping[str, MCResult]
    test_results: Mapping[str, tuple[TestResult, ...]] | None
    payloads: tuple[Mapping[str, Any], ...] | None
    contexts: tuple[MCContext, ...] | None
    failures: tuple[MCFailure, ...] = ()
    regression_summaries: Mapping[str, MCRegressionResult] = field(default_factory=dict)

    #: Post-loop (``OpType.POSTPROC``) artifacts, keyed by step name (or
    #: ``"<step>.<key>"`` for multi-artifact ops). Values are
    #: :class:`~SymbolicDSGE.monte_carlo.postproc.Summary` / ``Raw`` wrappers.
    postproc: Mapping[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """Whether the run succeeded (no per-replication or post-loop failures)."""
        return self.meta.steps_success and self.meta.postproc_success

    def report_performance(
        self,
        *,
        print_func: Callable[[str], None] = print,
    ) -> None:
        report_mc_performance(self.meta, print_func=print_func)

    def report_step_performance(
        self,
        *,
        print_func: Callable[[str], None] = print,
    ) -> None:
        report_mc_step_performance(self.meta, print_func=print_func)

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
    def test_status_traces(self) -> Mapping[str, tuple[TestStatus, ...]]:
        return {
            name: summary.status_trace for name, summary in self.test_summaries.items()
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
    def regression_status_traces(
        self,
    ) -> Mapping[str, tuple[RegressionStatus, ...]]:
        return {
            name: summary.status_trace
            for name, summary in self.regression_summaries.items()
        }


def _iterations_per_second(n_iter: int, elapsed_s: float) -> float:
    if n_iter == 0:
        return 0.0
    if elapsed_s <= 0.0:
        return float("inf")
    return n_iter / elapsed_s


def _conclusion_word(succeeded: bool) -> str:
    return "successfully" if succeeded else "unsuccessfully"


def report_mc_performance(
    meta: MCMeta,
    *,
    print_func: Callable[[str], None] = print,
) -> None:
    print_func(
        f"MC run concluded {_conclusion_word(meta.steps_success)} in {meta.elapsed_s:.2f}s with {meta.it_s:.2f} it/s."
    )
    if meta.postproc_elapsed_s:
        print_func(
            "Post-processing concluded "
            f"{_conclusion_word(meta.postproc_success)} in {meta.postproc_total_s:.4f}s."
        )


def report_mc_step_performance(
    meta: MCMeta,
    *,
    print_func: Callable[[str], None] = print,
) -> None:
    step_rates = meta.step_it_s
    print_func(
        f"MC run concluded {_conclusion_word(meta.failed_steps == {})} in {meta.elapsed_s:.2f}s with {meta.it_s:.2f} it/s."
    )
    print_func(f"Per-step Report:\n")
    for step_name in meta.step_elapsed_s:
        print_func(
            f"\t{step_name}: {meta.failed_steps.get(step_name, 0)} faliures, "
            f"{step_rates.get(step_name, 0.0):.2f} it/s ({meta.step_elapsed_s.get(step_name, 0.0):.2f}s)."
        )

    if meta.postproc_elapsed_s:
        print_func(f"\nPost-processing Report:\n")
        for step_name, elapsed_s in meta.postproc_elapsed_s.items():
            step_succeeded = (
                "Succeeded" if step_name not in meta.failed_postprocs else "Failed"
            )
            print_func(f"\t{step_name}: {step_succeeded} in {elapsed_s:.4f}s.")


def failed_postproc_names(fails: list[MCFailure]) -> set[str]:
    """Names of post-loop steps that failed (recorded with the ``-1`` sentinel)."""
    return {f.step_name for f in fails if f.rep_idx == -1}


def failed_step_counts(fails: list[MCFailure]) -> dict[str, int]:
    """Count of failed per-replication steps (recorded with non-negative rep_idx)."""
    counts: dict[str, int] = {}
    for f in fails:
        if f.rep_idx != -1:
            counts[f.step_name] = counts.get(f.step_name, 0) + 1
    return counts
