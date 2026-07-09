from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import StrEnum
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    NamedTuple,
    Protocol,
    Sequence,
    Union,
)

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
ColumnSelector = int | Sequence[int] | slice | NDArray[Any] | None
CompiledColumnSelector = Sequence[int] | slice | None
ShockValue = Union[Shock, Callable[[float | NDF], NDF], NDF]
ShockMapping = Mapping[str, ShockValue]
SeedIncrement = Union[int, Literal["auto"]]


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
    raw: Mapping[str, NDF] = {}
    observable_names: tuple[str, ...] = ()


MC_DATA_SOURCE_FIELDS: tuple[str, ...] = ("states", "observables")
DYNAMIC_SOURCE_FIELDS: tuple[str, ...] = ("payload",)
FILTER_RAW_SOURCE_FIELDS: tuple[str, ...] = UnscentedFilterRawResult._fields
FILTER_SOURCE_FIELDS: tuple[str, ...] = (
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

MC_DATA_FIELD_INDEX: dict[str, int] = {
    field: index for index, field in enumerate(MC_DATA_SOURCE_FIELDS)
}
DYNAMIC_FIELD_INDEX: dict[str, int] = {
    field: index for index, field in enumerate(DYNAMIC_SOURCE_FIELDS)
}
FILTER_RAW_FIELD_INDEX: dict[str, int] = {
    field: index for index, field in enumerate(FILTER_RAW_SOURCE_FIELDS)
}


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
    payload_slots: list[Any] = dataclass_field(default_factory=list)
    payloads: dict[str, Any] = dataclass_field(default_factory=dict)
    results: dict[str, TestResult] = dataclass_field(default_factory=dict)
    regressions: dict[str, RegressionResult] = dataclass_field(default_factory=dict)

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


SOURCE_KIND_DATA = 0
SOURCE_KIND_PAYLOAD = 1
SOURCE_KIND_FILTER = 2


@dataclass(frozen=True, slots=True)
class SourceArgs:
    arg: str
    source_step: str
    source_idx: int
    source_kind: int
    field: str
    field_idx: int
    columns: ColumnSelector = None
    column_selector: Sequence[int] | slice = dataclass_field(
        default_factory=lambda: slice(None)
    )
    row_start: int = 0

    burn_in: int = 0
    drop_initial: bool = False

    def __post_init__(self) -> None:
        columns = _normalize_columns(self.columns)
        row_start = int(self.burn_in)
        if row_start < 0:
            raise ValueError("burn_in must be non-negative.")
        if self.drop_initial and row_start == 0:
            row_start = 1
        object.__setattr__(self, "columns", columns)
        object.__setattr__(
            self,
            "column_selector",
            columns if columns is not None else slice(None),
        )
        object.__setattr__(self, "row_start", row_start)


@dataclass(frozen=True)
class MCStep:
    name: str
    op_type: OpType
    func: Callable[..., Any]
    kwargs: Mapping[str, Any] = dataclass_field(default_factory=dict)
    source_args: tuple[SourceArgs, ...] = ()
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
        object.__setattr__(self, "source_args", tuple(self.source_args))
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


def _compile_source_args(
    *,
    arg: str,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
) -> SourceArgs:
    source_step = str(source)
    if not source_step:
        raise ValueError("source must be non-empty.")
    source_field = str(field)
    if source_field in MC_DATA_FIELD_INDEX:
        return SourceArgs(
            arg=arg,
            source_step=source_step,
            source_idx=-1,
            source_kind=SOURCE_KIND_DATA,
            field=source_field,
            field_idx=MC_DATA_FIELD_INDEX[source_field],
            columns=columns,
            burn_in=burn_in,
            drop_initial=bool(drop_initial),
        )
    if source_field in FILTER_RAW_FIELD_INDEX:
        return SourceArgs(
            arg=arg,
            source_step=source_step,
            source_idx=-1,
            source_kind=SOURCE_KIND_FILTER,
            field=source_field,
            field_idx=FILTER_RAW_FIELD_INDEX[source_field],
            columns=columns,
            burn_in=burn_in,
            drop_initial=bool(drop_initial),
        )

    if source_field in DYNAMIC_FIELD_INDEX:
        return SourceArgs(
            arg=arg,
            source_step=source_step,
            source_idx=-1,
            source_kind=SOURCE_KIND_PAYLOAD,
            field=source_field,
            field_idx=DYNAMIC_FIELD_INDEX[source_field],
            columns=columns,
            burn_in=burn_in,
            drop_initial=bool(drop_initial),
        )

    raise ValueError(f"Unknown MC source field: {source_field!r}.")


def _normalize_columns(value: ColumnSelector) -> CompiledColumnSelector:
    if value is None or isinstance(value, slice):
        return value
    if isinstance(value, int):
        return (value,)
    if isinstance(value, np.ndarray):
        return tuple(int(item) for item in value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(int(item) for item in value)
    raise TypeError("Column selectors must be an int, a sequence of ints, or a slice.")


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
    step_elapsed_s: Mapping[str, float] = dataclass_field(default_factory=dict)
    step_counts: Mapping[str, int] = dataclass_field(default_factory=dict)
    step_failures: Mapping[str, int] = dataclass_field(default_factory=dict)
    #: Wall-clock seconds per post-loop (``OpType.POSTPROC``) step. Postproc runs
    #: once, so it is reported as runtime only, never folded into the it/s rates.
    postproc_elapsed_s: Mapping[str, float] = dataclass_field(default_factory=dict)

    failed_steps: dict[str, int] = dataclass_field(default_factory=dict)
    failed_postprocs: set[str] = dataclass_field(default_factory=set)

    @property
    def it_s(self) -> float:
        return _iterations_per_second(self.n_rep, self.elapsed_s)

    @property
    def step_it_s(self) -> Mapping[str, float]:
        return {
            name: _iterations_per_second(
                self.step_counts[name],
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
    regression_summaries: Mapping[str, MCRegressionResult] = dataclass_field(
        default_factory=dict
    )

    #: Post-loop (``OpType.POSTPROC``) artifacts, keyed by step name (or
    #: ``"<step>.<key>"`` for multi-artifact ops). Values are
    #: :class:`~SymbolicDSGE.monte_carlo.postproc.Summary` / ``Raw`` wrappers.
    postproc: Mapping[str, Any] = dataclass_field(default_factory=dict)

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
            f"\t{step_name}: {meta.step_failures[step_name]} faliures, "
            f"{step_rates[step_name]:.2f} it/s ({meta.step_elapsed_s[step_name]:.2f}s)."
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
