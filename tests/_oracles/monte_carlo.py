"""Python reference implementation of the Monte Carlo replication loop,
retained as a parity oracle for the native loop.

This is the driver as it stood before nativization: a Python loop that builds an
:class:`MCContext` per replication, runs each step through ``step.func``, reduces
the per-replication results into traces, and samples a bounded pool of the result
objects themselves. The library's own driver is free to be refactored, deleted,
or replaced by the native loop; this copy stays put so a native run can be
compared against the semantics that were signed off.

Frozen here rather than imported, so an edit to ``SymbolicDSGE.monte_carlo`` can
never silently move the reference:

- the loop driver (:func:`run_replication_loop`) and its step dispatch,
- the retention model: :func:`pool_stride` plus the three pools,
- the in-loop reductions: the test, regression, and data accumulators.

Container types (``MCStep`` / ``MCContext`` / ``MCData`` / the result objects)
are imported, not copied: they are the data model both sides agree on, and a
divergence there should surface as an import error rather than as silent drift.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping, Sequence, cast

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from SymbolicDSGE._diag_tests.result import MCResult, TestResult
from SymbolicDSGE._diag_tests.status import TestStatus
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.kalman.filter import FilterRawResult, UnscentedFilterRawResult
from SymbolicDSGE.monte_carlo.mc_constructs import (
    MCContext,
    MCData,
    MCDataSummary,
    MCFailure,
    MCMeta,
    MCPipelineResult,
    MCStep,
    OpType,
    SourceArgs,
    failed_postproc_names,
    failed_step_counts,
)
from SymbolicDSGE.regression.enums import RegressionStatus
from SymbolicDSGE.regression.ols import MCRegressionResult, OLSResult
from SymbolicDSGE.regression.result import RegressionResult

NDF = NDArray[float64]


# --- retention model ---------------------------------------------------------
def pool_stride(n_rep: int, poolsize: int) -> tuple[bool, int]:
    """Retention flag and replication stride for a fixed-size inspection pool.

    A non-positive ``poolsize`` disables retention and reports a zero stride the
    caller never applies (the flag short-circuits ahead of the modulo). Otherwise
    the stride is ``ceil(n_rep / poolsize)``, which keeps the pool at or under
    ``poolsize`` entries without a running count, and is 1 for a run shorter than
    the pool so every replication is kept. Sampling the tail is not guaranteed:
    the final ``stride - 1`` replications fall outside the sample.
    """
    if poolsize <= 0:
        return False, 0
    return True, (n_rep + poolsize - 1) // poolsize


# --- in-loop reductions ------------------------------------------------------
def df_metadata_matches(a: object, b: object) -> bool:
    """Compare normalized df metadata across replications, treating NaN == NaN
    as a match. Parameter-free reference distributions (CUSUM) carry a NaN df
    placeholder, which a plain ``!=`` would otherwise flag as incompatible."""
    at = a if isinstance(a, tuple) else (a,)
    bt = b if isinstance(b, tuple) else (b,)
    if len(at) != len(bt):
        return False
    for x, y in zip(at, bt):
        if x == y:
            continue
        if (
            isinstance(x, float | np.floating)
            and isinstance(y, float | np.floating)
            and np.isnan(x)
            and np.isnan(y)
        ):
            continue
        return False
    return True


class TestResultAccumulator:
    """Streaming builder for an :class:`MCResult` over a replication loop.

    Pulls the statistic and status off each per-replication :class:`TestResult` at
    push time so the result object itself can be released, and keeps the
    rep-invariant metadata (``dist`` / ``df`` / ``pval_method`` / ``alpha``) from
    the first push, rejecting a later result that disagrees. The statistic buffer
    is sized on ``n_rep`` up front and filled at a cursor, so a run with skipped
    replications finalizes to the count actually pushed.
    """

    def __init__(self, n_rep: int) -> None:
        if n_rep <= 0:
            raise ValueError("TestResultAccumulator requires a positive n_rep.")
        self._statistic: NDF = np.empty(n_rep, dtype=np.float64)
        self._status: list[TestStatus] = []
        self._metadata: tuple[Any, ...] | None = None
        self._cursor = 0

    @property
    def n_pushed(self) -> int:
        """Replications pushed so far."""
        return self._cursor

    def push(self, result: TestResult, *, step_name: str = "") -> None:
        """Record one replication's result, releasing the object to the caller."""
        metadata = (result.dist, result.df, result.pval_method, result.alpha)
        if self._metadata is None:
            self._metadata = metadata
        else:
            dist, df, pval_method, alpha = self._metadata
            if (
                result.dist is not dist
                or result.pval_method is not pval_method
                or not df_metadata_matches(result.df, df)
                or result.alpha != alpha
            ):
                raise ValueError(
                    f"Test results for step '{step_name}' have incompatible metadata."
                )
        if self._cursor >= self._statistic.size:
            raise ValueError(
                f"Test results for step '{step_name}' exceed the accumulator's "
                f"capacity of {self._statistic.size} replications."
            )
        self._statistic[self._cursor] = result.statistic
        self._status.append(result.status)
        self._cursor += 1

    def finalize(self, test_name: str) -> MCResult:
        """Build the :class:`MCResult` over the replications pushed so far."""
        if self._metadata is None:
            raise ValueError(
                f"Test results for step '{test_name}' are empty; MCResult requires "
                "at least one replication."
            )
        dist, df, pval_method, alpha = self._metadata
        return MCResult(
            test_name=test_name,
            dist=dist,
            df=df,
            pval_method=pval_method,
            alpha=alpha,
            statistic_trace=self._statistic[: self._cursor].copy(),
            status_trace=tuple(self._status),
        )


class RegressionAccumulator:
    """Streaming builder for an :class:`MCRegressionResult` over a replication loop.

    Extracts each replication's sufficient statistics (coefficients, status, SSR,
    SST, and OLS standard errors) at push time and releases the
    :class:`RegressionResult`, so neither the design matrix nor the response
    survives the replication. The standard errors are computed here rather than
    lazily, since the design they need is gone once the object is released.

    Buffers are sized on ``n_rep`` up front and filled at a cursor, so a run with
    skipped replications finalizes to the count actually pushed.
    """

    def __init__(self, n_rep: int) -> None:
        if n_rep <= 0:
            raise ValueError("RegressionAccumulator requires a positive n_rep.")
        self._capacity = int(n_rep)
        self._coef: NDF | None = None
        self._se: NDF | None = None
        self._ssr: NDF = np.empty(self._capacity, dtype=np.float64)
        self._sst: NDF = np.empty(self._capacity, dtype=np.float64)
        self._status: list[RegressionStatus] = []
        self._variables: list[str] | None = None
        self._n = -1
        self._k = -1
        self._is_ols = False
        self._cursor = 0

    @property
    def n_pushed(self) -> int:
        """Replications pushed so far."""
        return self._cursor

    def push(self, result: RegressionResult) -> None:
        """Record one replication's result, releasing the object to the caller."""
        if self._variables is None:
            self._variables = list(result.variables)
            self._n = result.n
            self._k = result.k
            self._is_ols = isinstance(result, OLSResult)
            self._coef = np.empty((self._capacity, self._k), dtype=np.float64)
            if self._is_ols:
                self._se = np.empty((self._capacity, self._k), dtype=np.float64)
        else:
            if result.variables != self._variables:
                raise ValueError("MC regression results have incompatible variables.")
            if result.n != self._n or result.k != self._k:
                raise ValueError("MC regression results have incompatible dimensions.")
            if self._is_ols != isinstance(result, OLSResult):
                raise ValueError("MC regression results have incompatible kinds.")
        if self._cursor >= self._capacity:
            raise ValueError(
                "MC regression results exceed the accumulator's capacity of "
                f"{self._capacity} replications."
            )

        assert self._coef is not None
        self._coef[self._cursor] = result.coefficients
        self._ssr[self._cursor] = result.ssr
        self._sst[self._cursor] = result.sst
        self._status.append(result.status)
        if self._se is not None:
            self._se[self._cursor] = cast(OLSResult, result).se
        self._cursor += 1

    def finalize(self) -> MCRegressionResult:
        """Build the summary over the replications pushed so far."""
        if self._variables is None or self._coef is None:
            raise ValueError("MCRegressionResult requires at least one result.")
        cursor = self._cursor
        return MCRegressionResult(
            variables=self._variables,
            coef_trace=self._coef[:cursor].copy(),
            status_trace=tuple(self._status),
            ssr_trace=self._ssr[:cursor].copy(),
            sst_trace=self._sst[:cursor].copy(),
            n=self._n,
            is_ols=self._is_ols,
            stored_se_trace=(None if self._se is None else self._se[:cursor].copy()),
        )


class DataAccumulator:
    """Streaming summary of the :class:`MCData` arrays a run produces.

    One instance covers a whole run: :meth:`push` folds a replication's
    ``states``, ``observables``, and raw series into per-name running sums, and
    :meth:`finalize` divides out. Nothing here scales with ``n_rep`` or with the
    sample dimension, so a replication's arrays are free the moment it ends.

    Statistics are pooled over every finite element of every contributing
    replication, not averaged over per-replication statistics.
    """

    #: Raw series carrying the state matrix under a reserved key; already
    #: summarized as ``states``, so it never gets its own entry.
    _SKIP_RAW = "_X"

    def __init__(self) -> None:
        self._n_rep: dict[str, int] = {}
        self._shape: dict[str, tuple[int, ...]] = {}
        self._n_values: dict[str, int] = {}
        self._n_finite: dict[str, int] = {}
        self._sum: dict[str, float] = {}
        self._square_sum: dict[str, float] = {}
        self._min: dict[str, float] = {}
        self._max: dict[str, float] = {}

    def push(self, data: MCData | None) -> None:
        """Fold one replication's arrays into the running summaries."""
        if data is None:
            return
        if data.states is not None:
            self._push_array("states", np.asarray(data.states))
        if data.observables is not None:
            self._push_array("observables", np.asarray(data.observables))
        for name, value in data.raw.items():
            if name != self._SKIP_RAW:
                self._push_array(f"raw:{name}", np.asarray(value))

    def _push_array(self, name: str, array: np.ndarray) -> None:
        if name not in self._n_rep:
            self._n_rep[name] = 0
            self._shape[name] = array.shape
            self._n_values[name] = 0
            self._n_finite[name] = 0
            self._sum[name] = 0.0
            self._square_sum[name] = 0.0
            self._min[name] = np.inf
            self._max[name] = -np.inf
        self._n_rep[name] += 1
        self._n_values[name] += int(array.size)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return
        self._n_finite[name] += int(finite.size)
        self._sum[name] += float(finite.sum())
        self._square_sum[name] += float(np.square(finite).sum())
        self._min[name] = min(self._min[name], float(finite.min()))
        self._max[name] = max(self._max[name], float(finite.max()))

    def finalize(self) -> dict[str, MCDataSummary]:
        """The per-name summaries, in first-seen order."""
        return {name: self._finalize_one(name) for name in self._n_rep}

    def _finalize_one(self, name: str) -> MCDataSummary:
        n_finite = self._n_finite[name]
        if n_finite == 0:
            return MCDataSummary(
                n_rep=self._n_rep[name],
                shape=self._shape[name],
                n_values=self._n_values[name],
                n_finite=0,
                mean=None,
                std=None,
                min=None,
                max=None,
            )
        mean = self._sum[name] / n_finite
        # Clamped because the sum-of-squares form can go slightly negative on a
        # near-constant series once cancellation bites.
        variance = max(0.0, self._square_sum[name] / n_finite - mean**2)
        return MCDataSummary(
            n_rep=self._n_rep[name],
            shape=self._shape[name],
            n_values=self._n_values[name],
            n_finite=n_finite,
            mean=mean,
            std=variance**0.5,
            min=self._min[name],
            max=self._max[name],
        )


# --- step dispatch -----------------------------------------------------------
def resolve_source_array(context: MCContext, selector: SourceArgs) -> NDF:
    """One step input, read out of an earlier step's payload slot."""
    out: NDF = context.payload_slots[selector.source_idx][selector.field_idx][
        selector.row_start :, selector.column_selector
    ]
    return out


def run_step(context: MCContext, step: MCStep) -> None:
    """Run one step against the replication's context, writing its output back."""
    kwargs = dict(step.kwargs)
    for selector in step.source_args:
        kwargs[selector.arg] = resolve_source_array(context, selector)
    if step.op_type is OpType.DATAGEN:
        out = step.func(
            reference=context.reference,
            dgp=context.dgp,
            rep_idx=context.rep_idx,
            **kwargs,
        )

        if not isinstance(out, MCData):
            raise TypeError("DATAGEN steps must return MCData.")
        context.data = out
        context.payload_slots.append(out)
        context.payloads[step.output_key] = out
        return

    out = step.func(
        context=context,
        reference=context.reference,
        dgp=context.dgp,
        rep_idx=context.rep_idx,
        **kwargs,
    )
    if step.op_type is OpType.TRANSFORM and isinstance(out, MCData):
        context.data = out
    if step.op_type is OpType.FILTER and not isinstance(
        out,
        (FilterRawResult, UnscentedFilterRawResult),
    ):
        raise TypeError("FILTER steps must return a raw filter result.")
    if step.op_type is OpType.REGRESSION and not isinstance(out, RegressionResult):
        raise TypeError("REGRESSION steps must return RegressionResult.")
    if step.op_type is OpType.REGRESSION:
        context.regressions[step.name] = out  # pyright: ignore
    if step.op_type is OpType.TEST:
        if not isinstance(out, TestResult):
            raise TypeError("TEST steps must return TestResult.")
        context.results[step.name] = out
    context.payload_slots.append(source_slot(step, out))
    context.payloads[step.output_key] = out


def source_slot(step: MCStep, out: Any) -> Any:
    if step.op_type is OpType.TRANSFORM:
        if isinstance(out, MCData):
            return (out,)
        return (source_array(out),)
    return out


def source_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"Transform payloads used as sources must be 1D or 2D, got shape {arr.shape}."
        )
    return arr


# --- post-loop payload traces ------------------------------------------------
def payload_to_array(value: object) -> np.ndarray | None:
    """A per-rep payload value as a stackable numeric array, else ``None``.

    Only numeric ndarray / scalar payloads (e.g. transform outputs) qualify;
    structured payloads (``MCData`` / raw filter results / result objects) are
    skipped from the post-loop trace registry.
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (int, float, np.number)):
        return np.asarray(value, dtype=np.float64)
    return None


def accumulate_payload_columns(
    columns: dict[str, list[np.ndarray]], payloads: Mapping[str, object]
) -> None:
    for key, value in payloads.items():
        array = payload_to_array(value)
        if array is not None:
            columns.setdefault(key, []).append(array)


def stack_payload_columns(
    columns: Mapping[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    """Stack per-rep payload arrays into ``payload.<name>`` traces.

    Only keys whose per-rep arrays share a shape across replications are stacked
    (a transform whose output length varies per rep is skipped)."""
    from SymbolicDSGE.monte_carlo.traces import payload_trace_key

    out: dict[str, np.ndarray] = {}
    for name, arrays in columns.items():
        if arrays and len({array.shape for array in arrays}) == 1:
            out[payload_trace_key(name)] = np.stack(arrays)
    return out


def run_postproc(
    postproc_steps: Sequence[MCStep],
    *,
    test_summaries: Mapping[str, Any],
    regression_summaries: Mapping[str, Any],
    payload_columns: Mapping[str, list[np.ndarray]],
    fail_fast: bool,
    failures: list[MCFailure],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run POSTPROC ops once over the assembled traces; collect artifacts.

    Owns its own timing: returns ``(artifacts, postproc_elapsed_s)`` where the
    second maps each step name to its wall-clock seconds. ``traces`` carries
    every keyed across-rep ndarray: the test/regression summary traces (shared
    with the result wire) plus stacked transform payloads. A failing op honors
    ``fail_fast`` (re-raise) or records an :class:`MCFailure` with ``rep_idx=-1``
    (post-loop sentinel) and is skipped.
    """
    postproc_elapsed_s: dict[str, float] = {step.name: 0.0 for step in postproc_steps}
    if not postproc_steps:
        return {}, postproc_elapsed_s

    from SymbolicDSGE.monte_carlo.serialize import traces_from_summaries

    traces: dict[str, np.ndarray] = traces_from_summaries(
        test_summaries, regression_summaries
    )
    traces.update(stack_payload_columns(payload_columns))

    postproc: dict[str, Any] = {}
    for step in postproc_steps:
        step_start = perf_counter()
        out: Any = None
        failed = False
        try:
            out = step.func(traces=traces, **dict(step.kwargs))
        except Exception as exc:
            failed = True
            if fail_fast:
                raise
            failures.append(
                MCFailure(
                    rep_idx=-1,
                    step_name=step.name,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
        finally:
            postproc_elapsed_s[step.name] += perf_counter() - step_start
        if not failed:
            postproc[step.name] = out
    return postproc, postproc_elapsed_s


# --- the loop ----------------------------------------------------------------
def run_replication_loop(
    per_rep_steps: Sequence[MCStep],
    postproc_steps: Sequence[MCStep] = (),
    *,
    reference: SolvedModel,
    dgp: SolvedModel | None = None,
    n_rep: int,
    payload_poolsize: int = 10000,
    test_result_poolsize: int = 10000,
    context_poolsize: int = 0,
    fail_fast: bool = True,
) -> MCPipelineResult:
    """The pre-native replication loop, over already-bound steps.

    Takes the step tuples rather than an ``MCPipeline`` so validation and source
    binding stay with the library: this is the executor, not the compiler. Pass
    ``pipeline.per_rep_steps`` / ``pipeline.postproc_steps`` to drive it from a
    pipeline built the normal way.
    """
    if n_rep <= 0:
        raise ValueError("n_rep must be positive.")

    n_successful = 0
    retain_payloads, payload_stride = pool_stride(n_rep, payload_poolsize)
    retain_test_results, test_stride = pool_stride(n_rep, test_result_poolsize)
    retain_contexts, context_stride = pool_stride(n_rep, context_poolsize)

    contexts: list[MCContext] = []
    payload_traces: list[Mapping[str, object]] = []
    failures: list[MCFailure] = []
    # Test / regression results are reduced to their traces as each replication
    # finishes, so the result objects never accumulate. The strided pools below
    # retain a bounded sample of the objects themselves for inspection; the
    # traces they summarize stay full length, so pooling never costs MC
    # granularity.
    test_accumulators: dict[str, TestResultAccumulator] = {}
    regression_accumulators: dict[str, RegressionAccumulator] = {}
    test_result_pool: dict[str, list[TestResult]] = {}
    # The generated data is the largest per-replication object, so it is
    # summarized on the spot and dropped rather than retained behind a pool.
    data_accumulator = DataAccumulator()
    # Per-replication step timings feed the it/s rates. Postproc runs once after
    # the loop and times itself; its runtime is never folded into the it/s
    # denominator.
    step_elapsed_s: dict[str, float] = {s.name: 0.0 for s in per_rep_steps}
    step_counts: dict[str, int] = {s.name: 0 for s in per_rep_steps}
    step_failures: dict[str, int] = {s.name: 0 for s in per_rep_steps}
    payload_columns: dict[str, list[np.ndarray]] = {}

    loop_start = perf_counter()
    for rep_idx in range(n_rep):
        context = MCContext(rep_idx=rep_idx, reference=reference, dgp=dgp)
        failed_step_name: str | None = None
        try:
            for step in per_rep_steps:
                failed_step_name = step.name
                step_start = perf_counter()
                try:
                    run_step(context, step)
                except Exception:
                    step_failures[step.name] += 1
                    raise
                finally:
                    step_elapsed_s[step.name] += perf_counter() - step_start
                    step_counts[step.name] += 1
        except Exception as exc:
            if fail_fast:
                raise
            failures.append(
                MCFailure(
                    rep_idx=rep_idx,
                    step_name=failed_step_name or "",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            continue

        n_successful += 1
        data_accumulator.push(context.data)
        if retain_contexts and (rep_idx % context_stride == 0):
            contexts.append(context)
        if retain_payloads and (rep_idx % payload_stride == 0):
            payload_traces.append(dict(context.payloads))
        pool_test_results = retain_test_results and (rep_idx % test_stride == 0)
        for name, test_result in context.results.items():
            accumulator = test_accumulators.get(name)
            if accumulator is None:
                accumulator = test_accumulators[name] = TestResultAccumulator(n_rep)
            accumulator.push(test_result, step_name=name)
            if pool_test_results:
                test_result_pool.setdefault(name, []).append(test_result)
        for name, regression_result in context.regressions.items():
            regression_accumulator = regression_accumulators.get(name)
            if regression_accumulator is None:
                regression_accumulator = regression_accumulators[name] = (
                    RegressionAccumulator(n_rep)
                )
            regression_accumulator.push(regression_result)
        if postproc_steps:
            accumulate_payload_columns(payload_columns, context.payloads)

    # Stop the replication-loop clock here; it/s is n_rep over the loop alone.
    # Post-loop aggregation and the once-run postproc phase are timed separately
    # and never enter the it/s denominator.
    elapsed_s = perf_counter() - loop_start

    test_summaries = {
        name: accumulator.finalize(name)
        for name, accumulator in test_accumulators.items()
    }
    regression_summaries = {
        name: accumulator.finalize()
        for name, accumulator in regression_accumulators.items()
    }
    postproc, postproc_elapsed_s = run_postproc(
        postproc_steps,
        test_summaries=test_summaries,
        regression_summaries=regression_summaries,
        payload_columns=payload_columns,
        fail_fast=fail_fast,
        failures=failures,
    )

    meta = MCMeta(
        n_rep=n_rep,
        payloads_retained=retain_payloads,
        test_results_retained=retain_test_results,
        contexts_retained=retain_contexts,
        elapsed_s=elapsed_s,
        step_elapsed_s=step_elapsed_s,
        step_counts=step_counts,
        step_failures=step_failures,
        postproc_elapsed_s=postproc_elapsed_s,
        failed_postprocs=failed_postproc_names(failures),
        failed_steps=failed_step_counts(failures),
    )

    return MCPipelineResult(
        n_rep=n_rep,
        meta=meta,
        n_successful=n_successful,
        test_summaries=test_summaries,
        test_results=(
            {name: tuple(values) for name, values in test_result_pool.items()}
            if retain_test_results
            else None
        ),
        payloads=tuple(payload_traces) if retain_payloads else None,
        contexts=tuple(contexts) if retain_contexts else None,
        failures=tuple(failures),
        regression_summaries=regression_summaries,
        postproc=postproc,
        data_summaries=data_accumulator.finalize(),
    )
