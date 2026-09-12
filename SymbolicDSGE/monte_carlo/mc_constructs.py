"""Containers for use in Monte Carlo pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field as dcf, fields
from functools import cached_property
from enum import StrEnum
from typing import (
    Any,
    Callable,
    Mapping,
    Sequence,
    Union,
    cast,
)

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..core.sim_result import SimResult, OccBinDiagnostics
from ..kalman.filter import FilterResult, UnscentedFilterResult
from .._diag_tests.result import MCTestResult
from .._diag_tests.status import TestStatus
from ..core.shock_generators import Shock
from ..regression.enums import RegressionStatus
from .postproc import Artifact
from ..regression.result import MCRegressionResult
from .custom_op import PandasCustomFunc
from .spec import SourceSpec, StepMeta, StepSpec

NDF = NDArray[float64]
NDI = NDArray[np.int_]
NDB = NDArray[np.bool_]
ColumnSelector = int | Sequence[int] | slice | NDArray[Any] | None
CompiledColumnSelector = Sequence[int] | slice | None
ShockValue = Union[Shock, Callable[[float | NDF], NDF], NDF]
ShockMapping = Mapping[str, ShockValue]


MC_DATA_SOURCE_FIELDS: tuple[str, ...] = ("states", "shocks", "observables")
DYNAMIC_SOURCE_FIELDS: tuple[str, ...] = ("payload",)
# The array-valued filter outputs, in tuple order. ``status`` is a scalar error
# code carried on the raw result, not a selectable source, so it is excluded.
FILTER_RAW_SOURCE_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(UnscentedFilterResult) if f.name != "status"
)
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
    """Operation role a pipeline step fills.

    Determines when a step runs and what contract it is held to: ``DATAGEN``,
    ``TRANSFORM``, ``FILTER``, ``TEST`` and ``REGRESSION`` run per replication,
    ``POSTPROC`` runs once after the loop.
    """

    DATAGEN = "datagen"
    TRANSFORM = "transform"
    FILTER = "filter"
    TEST = "test"
    REGRESSION = "regression"
    POSTPROC = "postproc"


@dataclass(frozen=True, slots=True)
class SourceArgs:
    """Compiled source selector used by transforms, tests and regressions.

    Factories build it from the public ``source`` and ``field`` arguments, and the
    native lowering layer resolves it to concrete buffer offsets before the run
    starts. Source fields are tied to the producer type: data steps expose
    ``states`` and ``observables``, transform steps expose ``payload``, and filter
    steps expose their raw filter fields. Array consumers expect the selected field
    to resolve to a two-dimensional numeric array.

    Attributes
    ----------
    arg : str
        Role the selected array fills, such as ``"sample"``, ``"y"`` or ``"X"``.
    source_step : str
        Producer step name, after pipeline binding.
    field : str
        Field read from the producer, such as ``"observables"``, ``"std_innov"``
        or ``"payload"``.
    columns : ColumnSelector
        Author-supplied column selector, normalized to a tuple of ints or a slice
        at construction.
    burn_in : int
        Number of leading rows to drop.
    """

    arg: str
    source_step: str
    field: str
    columns: ColumnSelector = None
    burn_in: int = 0

    def __post_init__(self) -> None:
        if int(self.burn_in) < 0:
            raise ValueError("burn_in must be non-negative.")
        object.__setattr__(self, "columns", _normalize_columns(self.columns))

    @property
    def _compiled_columns(self) -> CompiledColumnSelector:
        """``columns`` as :meth:`__post_init__` normalized it."""
        return cast(CompiledColumnSelector, self.columns)

    @property
    def column_selector(self) -> Sequence[int] | slice:
        """Selected columns, with an unset selector standing for all of them."""
        columns = self._compiled_columns
        return slice(None) if columns is None else columns

    def to_spec(self) -> SourceSpec:
        """Selector as plain data, keyed by constructor argument.

        Explicit indices travel as a list. A slice carries no width to resolve
        against, so it travels as its own ``start``/``stop``/``step``; the
        differing JSON types keep the two forms apart.
        """
        columns = self._compiled_columns
        spec_columns: list[int] | dict[str, int | None] | None
        if columns is None:
            spec_columns = None
        elif isinstance(columns, slice):
            spec_columns = {
                name: None if bound is None else int(bound)
                for name, bound in (
                    ("start", columns.start),
                    ("stop", columns.stop),
                    ("step", columns.step),
                )
            }
        else:
            spec_columns = [int(column) for column in columns]
        return {
            "arg": self.arg,
            "source_step": self.source_step,
            "field": self.field,
            "columns": spec_columns,
            "burn_in": int(self.burn_in),
        }

    @classmethod
    def from_spec(cls, spec: SourceSpec) -> SourceArgs:
        """Rebuild a selector from the form :meth:`to_spec` records.

        A slice's arguments are read by name, so a hand-authored spec that orders
        them differently still rebuilds the slice it spells.
        """
        recorded = spec["columns"]
        columns: ColumnSelector
        if isinstance(recorded, Mapping):
            columns = slice(
                recorded.get("start"), recorded.get("stop"), recorded.get("step")
            )
        else:
            columns = recorded
        return cls(
            arg=spec["arg"],
            source_step=spec["source_step"],
            field=spec["field"],
            columns=columns,
            burn_in=spec["burn_in"],
        )


@dataclass(frozen=True)
class MCStep:
    """A single step in a Monte Carlo pipeline, including its name, type, and source arguments.

    Attributes
    ----------
    name : str
        Name of the step, used for identification and referencing in the pipeline.
    op_type : OpType
        :class:`OpType` indicating the type of operation (e.g., ``DATAGEN``,
        ``FILTER``, ``TEST``, etc.).
    func : Callable[..., Any] | None
        Callable for a custom operation, or ``None`` for built-in operations.
    kwargs : Mapping[str, Any]
        Mapping of keyword arguments to pass to the operation function.
    source_args : tuple[SourceArgs, ...]
        Tuple of :class:`SourceArgs` specifying the sources of data for this step.
    step_type : str | None
        Kind specifier of the step.
    n_retain : int
        Number of replications to retain after this step; -1 means retain all.
    """

    name: str
    op_type: OpType
    func: Callable[..., Any] | None = None
    kwargs: Mapping[str, Any] = dcf(default_factory=dict)
    source_args: tuple[SourceArgs, ...] = ()
    #: Catalog step kind (e.g. ``"wald"``, ``"standardize"``, ``"simulation"``)
    #: or ``"custom"`` for user-supplied ops. Stamped by the step factories;
    #: lets a live ``MCPipeline`` be compiled back to a serializable
    #: ``PipelineSpec`` without a ``func``-to-kind reverse map. ``None`` only for
    #: hand-built steps that bypassed the factories.
    step_type: str | None = None

    n_retain: int = -1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MCStep name must be non-empty.")

        if self.n_retain < -1:
            raise ValueError("MCStep n_retain must be -1 (retain all) or non-negative.")

        if (
            isinstance(self.func, PandasCustomFunc)
            and self.op_type is not OpType.POSTPROC
        ):
            raise ValueError(
                f"MCStep {self.name!r}: a PandasCustomFunc is only allowed in a "
                "post-loop (POSTPROC) step, not a "
                f"{self.op_type.value!r} step."
            )

    def to_spec(self) -> StepSpec:
        """Step as data, plus the two things data cannot hold.

        Bulk array kwargs are lifted into :attr:`StepSpec.arrays` and a custom
        op's callable rides :attr:`StepSpec.func`, so :attr:`StepSpec.meta` is
        JSON on its own and a bundle writes the other two as their own members.
        """
        if (
            self.func is not None
            and self.op_type is not OpType.POSTPROC
            and self.step_type != "transform:custom"
        ):
            raise ValueError(
                f"MCStep {self.name!r} carries a callable that no replication "
                f"step kind can restore; only a custom transform ships one."
            )
        arrays: dict[str, NDArray[Any]] = {}
        kwargs: dict[str, Any] = {}
        for key, value in self.kwargs.items():
            name = str(key)
            if name == "shocks" and isinstance(value, Mapping):
                kwargs[name] = {
                    str(shock): _shock_spec(str(shock), entry)
                    for shock, entry in value.items()
                }
            elif isinstance(value, np.ndarray):
                if value.size <= 50:
                    kwargs[name] = _jsonable(value)
                else:
                    arrays[name] = value
            else:
                kwargs[name] = _jsonable(value)
        return StepSpec(
            meta=StepMeta(
                name=self.name,
                op_type=self.op_type.value,
                step_type=self.step_type,
                kwargs=kwargs,
                source_args=[selector.to_spec() for selector in self.source_args],
                n_retain=int(self.n_retain),
            ),
            func=self.func,
            arrays=arrays,
        )

    @classmethod
    def from_spec(cls, spec: StepSpec) -> MCStep:
        """Rebuild a step from the form :meth:`to_spec` records.

        The arrays go back under the kwarg names they were lifted from. A
        ``shocks`` mapping is read as one, whichever step carries it, because
        :meth:`Shock.to_dict` leaves a plain mapping that nothing else about the
        value distinguishes.
        """
        meta = spec.meta
        kwargs: dict[str, Any] = {**meta["kwargs"], **spec.arrays}
        shocks = kwargs.get("shocks")
        if isinstance(shocks, Mapping):
            kwargs["shocks"] = {
                str(name): _restore_shock(entry) for name, entry in shocks.items()
            }
        return cls(
            name=meta["name"],
            op_type=OpType(meta["op_type"]),
            func=spec.func,
            kwargs=kwargs,
            source_args=tuple(
                SourceArgs.from_spec(source) for source in meta["source_args"]
            ),
            step_type=meta["step_type"],
            n_retain=meta["n_retain"],
        )


def _jsonable(value: Any) -> Any:
    """One kwarg value as data.

    Anything carrying a ``to_dict`` is asked for it, so a value only has to
    declare how it serializes once to travel in any step's kwargs.
    """
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _shock_spec(name: str, shock: Any) -> Any:
    """One named simulation shock as data.

    A generator spec serializes itself and a bare shock path travels as nested
    lists. A callable travels as nothing.
    """
    if callable(shock):
        raise TypeError(
            f"Shock {name!r} is a callable, which cannot be serialized. Pass the "
            f"`Shock` itself rather than the generator it produces."
        )
    return _jsonable(shock)


def _restore_shock(value: Any) -> Shock | NDF:
    """One named shock as a simulation takes it.

    A mapping is a generator spec; anything else is a shock path, which the walk
    out left as nested lists.
    """
    if isinstance(value, Shock):
        return value
    if isinstance(value, Mapping):
        return Shock.from_dict(value)
    return np.asarray(value, dtype=float64)


def _compile_source_args(
    *,
    arg: str,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
) -> SourceArgs:
    source_step = str(source)
    if not source_step:
        raise ValueError("source must be non-empty.")
    source_field = str(field)
    known_fields = (
        *MC_DATA_SOURCE_FIELDS,
        *FILTER_SOURCE_FIELDS,
        *DYNAMIC_SOURCE_FIELDS,
    )
    if source_field in known_fields:
        return SourceArgs(
            arg=arg,
            source_step=source_step,
            field=source_field,
            columns=columns,
            burn_in=burn_in,
        )

    if source_field in FILTER_RAW_SOURCE_FIELDS:
        raise ValueError(
            f"Filter output {source_field!r} cannot be read as a source. A "
            f"source is staged row by column, and {source_field!r} is not two "
            f"dimensional per replication. Readable filter fields: "
            f"{list(FILTER_SOURCE_FIELDS)}."
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


@dataclass(frozen=True, slots=True)
class MCFailure:
    """One collected replication failure from a run made with fail-fast disabled.

    Post-loop failures are recorded with ``rep_idx`` of -1.

    Attributes
    ----------
    rep_idx : int
        Replication index that failed, or -1 for a post-loop step.
    step_name : str
        Step executing when the failure occurred.
    error_type : str
        Exception type name.
    message : str
        Exception message.
    """

    rep_idx: int
    step_name: str
    error_type: str
    message: str


@dataclass(frozen=True)
class MCMeta:
    """Run accounting and performance counters for a Monte Carlo run.

    Attributes
    ----------
    n_rep : int
        Requested replication count.
    n_retained_by_step : Mapping[str, int]
        Replications whose output was retained, by producer step.
    elapsed_s : float
        Wall time for the replication loop alone, excluding post-loop aggregation
        and postproc.
    step_elapsed_s : Mapping[str, float]
        Accumulated worker seconds by per-replication step. Populated only when the
        run is started at the highest verbosity.
    step_counts : Mapping[str, int]
        Attempted calls by per-replication step.
    step_failures : Mapping[str, int]
        Collected failures by per-replication step.
    postproc_elapsed_s : Mapping[str, float]
        Wall time by post-loop step.
    failed_steps : dict[str, int]
        Collected per-replication failures by step.
    failed_postprocs : set[str]
        Post-loop steps that failed.
    """

    n_rep: int

    n_retained_by_step: Mapping[str, int]

    #: Wall-clock seconds of the replication loop alone; the basis for ``it_s``.
    #: Post-loop aggregation and postproc are excluded (see ``postproc_elapsed_s``).
    elapsed_s: float = 0.0
    #: Per-replication step timings (postproc excluded; see ``postproc_elapsed_s``).
    step_elapsed_s: Mapping[str, float] = dcf(default_factory=dict)
    step_counts: Mapping[str, int] = dcf(default_factory=dict)
    step_failures: Mapping[str, int] = dcf(default_factory=dict)
    #: Wall-clock seconds per post-loop (``OpType.POSTPROC``) step. Postproc runs
    #: once, so it is reported as runtime only, never folded into the it/s rates.
    postproc_elapsed_s: Mapping[str, float] = dcf(default_factory=dict)

    failed_steps: dict[str, int] = dcf(default_factory=dict)
    failed_postprocs: set[str] = dcf(default_factory=set)

    @property
    def it_s(self) -> float:
        """Replications attempted per replication loop second."""
        return _iterations_per_second(self.n_rep, self.elapsed_s)

    @property
    def step_it_s(self) -> Mapping[str, float]:
        """Exclusive per-step throughput against accumulated worker seconds.

        Alias for ``step_worker_it_s``.
        """
        return self.step_worker_it_s

    @property
    def step_worker_it_s(self) -> Mapping[str, float]:
        """Exclusive per-step throughput from accumulated worker-seconds."""
        return {
            name: _iterations_per_second(
                self.step_counts[name],
                elapsed_s,
            )
            for name, elapsed_s in self.step_elapsed_s.items()
        }

    @property
    def step_wall_it_s(self) -> Mapping[str, float]:
        """Per-step throughput against the replication loop's wall time."""
        return {
            name: _iterations_per_second(self.step_counts[name], self.elapsed_s)
            for name in self.step_elapsed_s
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


@dataclass(frozen=True, eq=False, repr=False)
class MCDataGenResult:
    """Stacked datagen output across the replications a run retained.

    Stacks the simulation result a datagen step produces on each replication into
    one container, adding a leading replication axis to every path. A pipeline has
    exactly one datagen step, so the pipeline result holds this container directly
    rather than a mapping.

    A datagen step declares what it writes, and a field it never produced still
    reports a block filled with NaN at the width its names imply, rather than being
    omitted.

    Attributes
    ----------
    n_rep : int
        Total number of replications.
    n_retained : int
        Number of replications whose output the step's arena kept.
    retained_reps : NDI
        Replication indices behind each row of the stacked paths, so a row can be
        mapped back to its replication.
    var_names : Sequence[str]
        Names of the state-path columns, in compiled canonical order.
    X : NDF
        Full state paths, shape ``(n_retained, T, n_var)``.
    shock_names : Sequence[str]
        Names of the shock columns, in canonical order.
    eps : NDF
        Shock paths, shape ``(n_retained, T, n_shock)``.
    observable_names : Sequence[str]
        Names of the observable-path columns, in observable order. Empty when the
        step produced no observables.
    y : NDF
        Observable paths, shape ``(n_retained, T, n_obs)``.
    """

    n_rep: int
    n_retained: int
    retained_reps: NDI

    var_names: Sequence[str]
    X: NDF  # (n_retained, T, n_var)
    shock_names: Sequence[str]
    eps: NDF  # (n_retained, T, n_shock)
    observable_names: Sequence[str] = ()
    y: NDF = dcf(  # (n_retained, T, n_obs)
        default_factory=lambda: np.empty((0, 0, 0), dtype=np.float64)
    )

    _regimes: NDI | None = None  # (n_retained, T, H)
    _diagnostics: Sequence[OccBinDiagnostics] | None = None  # (n_retained,)

    def replication(self, idx: int) -> SimResult:
        """Return a :class:`~SymbolicDSGE.core.sim_result.SimResult` for a single replication."""
        if idx < 0 or idx >= self.n_retained:
            raise IndexError(
                f"Replication index {idx} out of bounds for {self.n_retained} retained replications."
            )

        return SimResult(
            var_names=self.var_names,
            X=self.X[idx],
            shock_names=self.shock_names,
            eps=self.eps[idx],
            observable_names=self.observable_names,
            y=self.y[idx],
            _regimes=self._regimes[idx] if self._regimes is not None else None,
            _diagnostics=(
                self._diagnostics[idx] if self._diagnostics is not None else None
            ),
        )

    @cached_property
    def states(self) -> dict[str, NDF]:
        """Each model variable's path, as a column view of ``X``.

        Returns ``(n_retained, T)`` views per variable, keyed by variable name.
        """
        return {name: self.X[:, :, i] for i, name in enumerate(self.var_names)}

    @cached_property
    def shocks(self) -> dict[str, NDF]:
        """Each shock's path, as a column view of ``eps``.

        Returns ``(n_retained, T)`` views per shock, keyed by shock name.
        """
        return {name: self.eps[:, :, i] for i, name in enumerate(self.shock_names)}

    @cached_property
    def observables(self) -> dict[str, NDF]:
        """Each observable's path, as a column view of ``y``.

        Returns ``(n_retained, T)`` views per observable, keyed by observable name.
        """
        return {name: self.y[:, :, i] for i, name in enumerate(self.observable_names)}

    @property
    def regimes(self) -> NDI:
        """``(n_retained, T, H)`` accepted regime guess per period, per replication.

        ``H`` is the longest check-ahead horizon any period used, so a period
        that settled sooner is padded. Column 0 is the regime realized at that
        date.
        """
        if self._regimes is None:
            raise AttributeError(
                "Regimes are only recorded for models with constraints. "
                "Regular first/second-order simulations, including "
                ":func:`PiecewiseSolvedModel.sim_reference` on a constrained "
                "model, do not record them."
            )
        return self._regimes

    @property
    def diagnostics(self) -> Sequence[OccBinDiagnostics]:
        """Each replication's per-period convergence record behind :attr:`regimes`."""
        if self._diagnostics is None:
            raise AttributeError(
                "Diagnostics are only recorded for models with constraints. "
                "Regular first/second-order simulations, including "
                ":func:`PiecewiseSolvedModel.sim_reference` on a constrained "
                "model, do not record them."
            )
        return self._diagnostics


@dataclass(frozen=True, eq=False, repr=False)
class MCFilterResult:
    """Stacked filter output across the replications a run retained.

    Stacks the filter result a filter step produces on each replication into one
    container, adding a leading replication axis to every history. The pipeline
    result maps each filter step name to one of these.

    Attributes
    ----------
    n_rep : int
        Total number of replications.
    n_retained : int
        Number of replications whose output the step's arena kept.
    retained_reps : NDI
        Replication indices behind each row of the stacked histories.
    filter_mode : str
        Kernel the step ran: ``"linear"``, ``"extended"`` or ``"unscented"``.
    x_pred : NDF
        Predicted states over time, shape ``(n_retained, T, n_var)``.
    x_filt : NDF
        Filtered states over time, shape ``(n_retained, T, n_var)``.
    P_pred : NDF
        Predicted state covariance over time, shape ``(n_retained, T, n_var, n_var)``.
    P_filt : NDF
        Filtered state covariance over time, shape ``(n_retained, T, n_var, n_var)``.
    y_pred : NDF
        Predicted observables over time, shape ``(n_retained, T, n_obs)``.
    y_filt : NDF
        Filtered observables over time, shape ``(n_retained, T, n_obs)``.
    S : NDF
        Innovation covariance over time, shape ``(n_retained, T, n_obs, n_obs)``.
    innov : NDF
        Observable innovations, shape ``(n_retained, T, n_obs)``.
    std_innov : NDF
        Innovations standardized by their covariance, shape ``(n_retained, T, n_obs)``.
    loglik : NDF
        Per-replication log likelihood of the measurements, shape ``(n_retained,)``.
    eps_hat : NDF | None
        Conditional estimates of the structural shocks given observed data, shape
        ``(n_retained, T, n_shock)``. None when the step ran without shock recovery.
    x1_pred : NDF
        First-order component of the predicted state, shape
        ``(n_retained, T, n_state)``. Unscented filter steps only.
    x2_pred : NDF
        Second-order component of the predicted state, shape
        ``(n_retained, T, n_state)``. Unscented filter steps only.
    x1_filt : NDF
        First-order component of the filtered state, shape
        ``(n_retained, T, n_state)``. Unscented filter steps only.
    x2_filt : NDF
        Second-order component of the filtered state, shape
        ``(n_retained, T, n_state)``. Unscented filter steps only.
    """

    n_rep: int
    n_retained: int
    retained_reps: NDI

    filter_mode: str
    # Shared
    x_pred: NDF
    x_filt: NDF
    P_pred: NDF
    P_filt: NDF
    y_pred: NDF
    y_filt: NDF

    S: NDF

    innov: NDF
    std_innov: NDF
    loglik: NDF
    eps_hat: NDF | None = None

    # Unscented-specific
    _x1_pred: NDF | None = None
    _x2_pred: NDF | None = None
    _x1_filt: NDF | None = None
    _x2_filt: NDF | None = None

    def replication(self, idx: int) -> FilterResult | UnscentedFilterResult:
        """Return one retained replication's filter output as a single-run result.

        ``idx`` counts the retained replications, which are stored compactly,
        so it is not the replication's index in the run. ``retained_reps[idx]``
        is that index; the two coincide only when the run retained everything.
        """
        if idx < 0 or idx >= self.n_retained:
            raise IndexError(
                f"Replication index {idx} out of bounds for {self.n_retained} "
                f"retained replications."
            )

        if self.filter_mode == "unscented":
            return UnscentedFilterResult(
                x_pred=self.x_pred[idx],
                x_filt=self.x_filt[idx],
                P_pred=self.P_pred[idx],
                P_filt=self.P_filt[idx],
                y_pred=self.y_pred[idx],
                y_filt=self.y_filt[idx],
                S=self.S[idx],
                innov=self.innov[idx],
                std_innov=self.std_innov[idx],
                loglik=self.loglik[idx],
                x1_pred=self.x1_pred[idx],
                x1_filt=self.x1_filt[idx],
                x2_pred=self.x2_pred[idx],
                x2_filt=self.x2_filt[idx],
                eps_hat=self.eps_hat[idx] if self.eps_hat is not None else None,
            )
        return FilterResult(
            x_pred=self.x_pred[idx],
            x_filt=self.x_filt[idx],
            P_pred=self.P_pred[idx],
            P_filt=self.P_filt[idx],
            y_pred=self.y_pred[idx],
            y_filt=self.y_filt[idx],
            S=self.S[idx],
            innov=self.innov[idx],
            std_innov=self.std_innov[idx],
            loglik=self.loglik[idx],
            eps_hat=self.eps_hat[idx] if self.eps_hat is not None else None,
        )

    @cached_property
    def x1_pred(self) -> NDF:
        """First-order component of the predicted state over time.

        Shape ``(n_retained, T, n_state)``. Unscented filter steps only; raises for
        any other filter mode.
        """
        if self._x1_pred is None:
            raise AttributeError("x1_pred is only available for unscented filters.")
        return self._x1_pred

    @cached_property
    def x2_pred(self) -> NDF:
        """Second-order component of the predicted state over time.

        Shape ``(n_retained, T, n_state)``. Unscented filter steps only; raises for
        any other filter mode.
        """
        if self._x2_pred is None:
            raise AttributeError("x2_pred is only available for unscented filters.")
        return self._x2_pred

    @cached_property
    def x1_filt(self) -> NDF:
        """First-order component of the filtered state over time.

        Shape ``(n_retained, T, n_state)``. Unscented filter steps only; raises for
        any other filter mode.
        """
        if self._x1_filt is None:
            raise AttributeError("x1_filt is only available for unscented filters.")
        return self._x1_filt

    @cached_property
    def x2_filt(self) -> NDF:
        """Second-order component of the filtered state over time.

        Shape ``(n_retained, T, n_state)``. Unscented filter steps only; raises for
        any other filter mode.
        """
        if self._x2_filt is None:
            raise AttributeError("x2_filt is only available for unscented filters.")
        return self._x2_filt


@dataclass(frozen=True)
class MCPipelineResult:
    """Result container for a Monte Carlo run, including all retained per-replication and post-loop outputs.

    Attributes
    ----------
    meta : MCMeta
        Metadata about the run, including timing and failure counts.
    n_rep : int
        Total number of replications attempted.
    n_successful : int
        Successful replications (retained after filtering).
    datagen_outputs : MCDataGenResult
        MCDataGenResult containing the retained replications' simulated states,
        shocks, and observables.
    filter_outputs : Mapping[str, MCFilterResult]
        MCFilterResult per filter step, containing the retained replications' filter
        outputs.
    transform_outputs : Mapping[str, NDF]
        Mapping of step name to transformed data (NDF) for each transform step.
    test_summaries : Mapping[str, MCTestResult]
        MCTestResult per test step, containing the retained replications' test
        statistics and p-values.
    regression_summaries : Mapping[str, MCRegressionResult]
        MCRegressionResult per regression step, containing the retained
        replications' regression coefficients and statistics.
    failures : tuple[MCFailure, ...]
        MCFailure instances for any failed replications or post-loop steps.
    postproc : Mapping[str, Artifact]
        Post-loop step artifacts, keyed by step name, containing raw and summary
        outputs.
    run_config : Mapping[str, Any]
        Configuration to reproduce this run, given deterministic steps and the same
        inputs in model/data sources.
    succeeded
    statistic_traces
    pval_traces
    test_status_traces
    rejection_traces
    coefficient_traces
    regression_status_traces
    """

    meta: MCMeta
    n_rep: int
    n_successful: int
    datagen_outputs: MCDataGenResult
    filter_outputs: Mapping[str, MCFilterResult] = dcf(default_factory=dict)
    transform_outputs: Mapping[str, NDF] = dcf(default_factory=dict)
    test_summaries: Mapping[str, MCTestResult] = dcf(default_factory=dict)
    regression_summaries: Mapping[str, MCRegressionResult] = dcf(default_factory=dict)
    failures: tuple[MCFailure, ...] = ()

    #: Post-loop (``OpType.POSTPROC``) artifacts, keyed by step name. Each step
    #: contributes one :class:`~SymbolicDSGE.monte_carlo.postproc.Artifact`,
    #: holding its ``raw`` and ``summary`` slots (either may be ``None``).
    postproc: Mapping[str, Artifact] = dcf(default_factory=dict)
    run_config: Mapping[str, Any] = dcf(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """Whether the run succeeded (no per-replication or post-loop failures)."""
        return self.meta.steps_success and self.meta.postproc_success

    def report_performance(
        self,
        *,
        print_func: Callable[[str], None] = print,
    ) -> None:
        """Repost overall Monte Carlo performance metrics, including throughput and failure counts.

        Parameters
        ----------
        print_func : Callable[[str], None]
            Function to use for printing the report.
        """
        report_mc_performance(self.meta, print_func=print_func)

    def report_step_performance(
        self,
        *,
        print_func: Callable[[str], None] = print,
    ) -> None:
        """Report per-step performance metrics, including throughput and failure counts.

        Parameters
        ----------
        print_func : Callable[[str], None]
            Function to use for printing the report.
        """
        report_mc_step_performance(self.meta, print_func=print_func)

    @property
    def statistic_traces(self) -> Mapping[str, NDF]:
        """Traces of the retained replications' test statistics for each test step."""
        return {
            name: summary.statistic_trace
            for name, summary in self.test_summaries.items()
        }

    @property
    def pval_traces(self) -> Mapping[str, NDF]:
        """P-value traces of the retained replications' test statistics for each test step."""
        return {
            name: summary.pval_trace for name, summary in self.test_summaries.items()
        }

    @property
    def test_status_traces(self) -> Mapping[str, tuple[TestStatus, ...]]:
        """Test status traces of the retained replications' test statistics for each test step."""
        return {
            name: summary.status_trace for name, summary in self.test_summaries.items()
        }

    @property
    def rejection_traces(self) -> Mapping[str, NDB]:
        """Boolean rejection traces of the retained replications' test statistics for each test step, based on the significance level."""
        return {
            name: np.asarray(summary.pval_trace < summary.alpha, dtype=bool)
            for name, summary in self.test_summaries.items()
        }

    @property
    def coefficient_traces(self) -> Mapping[str, NDF]:
        """Regression coefficient traces of the retained replications' regression results for each regression step."""
        return {
            name: summary.coef_trace
            for name, summary in self.regression_summaries.items()
        }

    @property
    def regression_status_traces(
        self,
    ) -> Mapping[str, tuple[RegressionStatus, ...]]:
        """Regression status traces of the retained replications' regression results for each regression step."""
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
    worker_rates = meta.step_worker_it_s
    wall_rates = meta.step_wall_it_s
    print_func(
        f"MC run concluded {_conclusion_word(meta.failed_steps == {})} in {meta.elapsed_s:.2f}s with {meta.it_s:.2f} it/s."
    )
    print_func("Per-step Report:\n")
    for step_name in meta.step_elapsed_s:
        print_func(
            f"\t{step_name}: {meta.step_failures[step_name]} failures, "
            f"{worker_rates[step_name]:.2f} worker it/s "
            f"({meta.step_elapsed_s[step_name]:.2f} worker-s), "
            f"{wall_rates[step_name]:.2f} wall it/s."
        )

    if meta.postproc_elapsed_s:
        print_func("\nPost-processing Report:\n")
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
