from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from time import perf_counter
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from .graph import PipelineGraph
    from .spec import PipelineSpec

from .._diag_tests.result import MCResultAccumulator, TestResult
from ..core.solved_model import SolvedModel
from ..kalman.filter import FilterRawResult, UnscentedFilterRawResult
from ..regression.ols import MCRegressionAccumulator
from ..regression.result import RegressionResult
from .mc_constructs import (
    MCContext,
    MCData,
    MCDataAccumulator,
    MCFailure,
    MCPipelineResult,
    MCMeta,
    MCStep,
    OpType,
    SOURCE_KIND_DATA,
    SOURCE_KIND_FILTER,
    SOURCE_KIND_PAYLOAD,
    failed_postproc_names,
    failed_step_counts,
    report_mc_performance,
    report_mc_step_performance,
)
from .operations.utils import _resolve_source_array


@dataclass(frozen=True)
class MCPipeline:
    #: Per-replication steps: the dependency DAG, a single DATAGEN root first.
    per_rep_steps: tuple[MCStep, ...]
    #: Post-loop ops, run once after the loop over the assembled across-rep
    #: traces. This is a terminal phase, not part of the graph.
    postproc_steps: tuple[MCStep, ...]

    def __init__(
        self,
        per_rep_steps: Sequence[MCStep],
        postproc_steps: Sequence[MCStep] = (),
    ) -> None:
        rep_tuple = tuple(per_rep_steps)
        postproc_tuple = tuple(postproc_steps)
        self._validate_steps(rep_tuple, postproc_tuple)
        rep_tuple = self._bind_source_args(rep_tuple)
        object.__setattr__(self, "per_rep_steps", rep_tuple)
        object.__setattr__(self, "postproc_steps", postproc_tuple)

    @staticmethod
    def _validate_steps(
        per_rep_steps: tuple[MCStep, ...],
        postproc_steps: tuple[MCStep, ...],
    ) -> None:
        if not per_rep_steps:
            raise ValueError("MCPipeline requires at least one per-replication step.")
        names = [step.name for step in (*per_rep_steps, *postproc_steps)]
        if len(set(names)) != len(names):
            raise ValueError("MCPipeline step names must be unique.")
        if per_rep_steps[0].op_type is not OpType.DATAGEN:
            raise ValueError("MCPipeline first per-rep step must be a DATAGEN step.")
        for step in per_rep_steps[1:]:
            if step.op_type is OpType.DATAGEN:
                raise ValueError(
                    "MCPipeline supports only one DATAGEN step, in first position."
                )
            if step.op_type is OpType.POSTPROC:
                raise ValueError(
                    "POSTPROC steps belong in postproc_steps, not per_rep_steps."
                )
        for step in postproc_steps:
            if step.op_type is not OpType.POSTPROC:
                raise ValueError(
                    f"postproc_steps may only contain POSTPROC steps; {step.name!r} "
                    f"is {step.op_type}."
                )

    @staticmethod
    def _bind_source_args(per_rep_steps: tuple[MCStep, ...]) -> tuple[MCStep, ...]:
        index_by_name: dict[str, int] = {}
        canonical_name: dict[str, str] = {}
        for index, step in enumerate(per_rep_steps):
            index_by_name[step.name] = index
            canonical_name[step.name] = step.name
            index_by_name[step.output_key] = index
            canonical_name[step.output_key] = step.name
        bound_steps: list[MCStep] = []
        for step_index, step in enumerate(per_rep_steps):
            bound_args = []
            for selector in step.source_args:
                producer = selector.source_step
                if producer not in index_by_name:
                    raise ValueError(
                        f"Step {step.name!r} depends on unknown producer {producer!r}."
                    )
                source_idx = index_by_name[producer]
                producer_step = per_rep_steps[source_idx]
                producer = canonical_name[producer]
                if source_idx >= step_index:
                    raise ValueError(
                        f"Step {step.name!r} depends on {producer!r}, which does not "
                        "appear earlier in the pipeline."
                    )
                _validate_source_producer(step, selector, producer_step)
                bound_args.append(
                    replace(selector, source_step=producer, source_idx=source_idx)
                )
            if tuple(bound_args) != step.source_args:
                step = replace(step, source_args=tuple(bound_args))
            bound_steps.append(step)
        return tuple(bound_steps)

    @cached_property
    def graph(self) -> "PipelineGraph":
        """The pipeline's dependency DAG, resolved from compiled source args.

        Built once and cached. Owns the graph structure (parents/children/leaves/
        typed input edges) that serialization and validation read instead of
        re-deriving it. Lazily imported to keep ``core`` light at import time.
        """
        from .graph import PipelineGraph

        return PipelineGraph.from_steps(self.per_rep_steps)

    def to_spec(self) -> "PipelineSpec":
        """Serialize this pipeline to its graph-form :class:`PipelineSpec`.

        The inverse of :func:`build_pipeline`: lets a pipeline authored with
        plain library objects be stored in a bundle without touching the spec
        DTOs. Bulk side-channels (``raw_model_data`` arrays, custom-op blobs) are
        referenced by key and written as bundle members by the bundle builder.
        """
        from .spec_compile import pipeline_to_spec

        return pipeline_to_spec(self)

    def run(
        self,
        *,
        reference: SolvedModel,
        dgp: SolvedModel | None = None,
        n_rep: int,
        payload_poolsize: int = 10000,
        test_result_poolsize: int = 10000,
        # Contexts are off by default: a retained ``MCContext`` transitively holds
        # this replication's ``MCData`` and its result objects (designs included),
        # so a context pool re-retains exactly what the trace accumulators exist to
        # release. The native loop has no per-replication context at all.
        context_poolsize: int = 0,
        fail_fast: bool = True,
        verbosity: int = 1,
    ) -> MCPipelineResult:
        if n_rep <= 0:
            raise ValueError("n_rep must be positive.")
        if verbosity not in (0, 1, 2):
            raise ValueError("verbosity must be 0, 1, or 2.")

        n_successful = 0
        retain_payloads, payload_stride = _pool_stride(n_rep, payload_poolsize)
        retain_test_results, test_stride = _pool_stride(n_rep, test_result_poolsize)
        retain_contexts, context_stride = _pool_stride(n_rep, context_poolsize)

        contexts: list[MCContext] = []
        payload_traces: list[Mapping[str, object]] = []
        failures: list[MCFailure] = []
        # Test / regression results are reduced to their traces as each
        # replication finishes, so the result objects never accumulate. The
        # strided pools below retain a bounded sample of the objects themselves
        # for inspection; the traces they summarize stay full length, so pooling
        # never costs MC granularity.
        test_accumulators: dict[str, MCResultAccumulator] = {}
        regression_accumulators: dict[str, MCRegressionAccumulator] = {}
        test_result_pool: dict[str, list[TestResult]] = {}
        # The generated data is the largest per-replication object, so it is
        # summarized on the spot and dropped rather than retained behind a pool.
        data_accumulator = MCDataAccumulator()
        # Per-replication step timings feed the it/s rates. Postproc runs once
        # after the loop and times itself (see ``_run_postproc``); its runtime is
        # never folded into the it/s denominator.
        step_elapsed_s: dict[str, float] = {s.name: 0.0 for s in self.per_rep_steps}
        step_counts: dict[str, int] = {s.name: 0 for s in self.per_rep_steps}
        step_failures: dict[str, int] = {s.name: 0 for s in self.per_rep_steps}

        # POSTPROC ops don't run per replication. They run once after the loop,
        # over the assembled across-rep traces.
        rep_steps = self.per_rep_steps
        postproc_steps = self.postproc_steps
        payload_columns: dict[str, list[np.ndarray]] = {}

        loop_start = perf_counter()
        for rep_idx in range(n_rep):
            context = MCContext(rep_idx=rep_idx, reference=reference, dgp=dgp)
            failed_step_name: str | None = None
            try:
                for step in rep_steps:
                    failed_step_name = step.name
                    step_start = perf_counter()
                    try:
                        self._run_step(context, step)
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
                    accumulator = test_accumulators[name] = MCResultAccumulator(n_rep)
                accumulator.push(test_result, step_name=name)
                if pool_test_results:
                    test_result_pool.setdefault(name, []).append(test_result)
            for name, regression_result in context.regressions.items():
                regression_accumulator = regression_accumulators.get(name)
                if regression_accumulator is None:
                    regression_accumulator = regression_accumulators[name] = (
                        MCRegressionAccumulator(n_rep)
                    )
                regression_accumulator.push(regression_result)
            if postproc_steps:
                _accumulate_payload_columns(payload_columns, context.payloads)

        # Stop the replication-loop clock here; it/s is n_rep over the loop
        # alone. Post-loop aggregation and the once-run postproc phase are timed
        # separately and never enter the it/s denominator.
        elapsed_s = perf_counter() - loop_start

        test_summaries = {
            name: accumulator.finalize(name)
            for name, accumulator in test_accumulators.items()
        }
        regression_summaries = {
            name: accumulator.finalize()
            for name, accumulator in regression_accumulators.items()
        }
        postproc, postproc_elapsed_s = self._run_postproc(
            postproc_steps,
            test_summaries=test_summaries,
            regression_summaries=regression_summaries,
            payload_columns=payload_columns,
            reference=reference,
            dgp=dgp,
            fail_fast=fail_fast,
            failures=failures,
        )

        failed_postprocs = failed_postproc_names(failures)
        failed_steps = failed_step_counts(failures)

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
            failed_postprocs=failed_postprocs,
            failed_steps=failed_steps,
        )

        result = MCPipelineResult(
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
        if verbosity == 1:
            report_mc_performance(meta)
        elif verbosity == 2:
            report_mc_step_performance(meta)
        return result

    def _run_step(self, context: MCContext, step: MCStep) -> None:
        kwargs = dict(step.kwargs)
        for selector in step.source_args:
            kwargs[selector.arg] = _resolve_source_array(context, selector)
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
        context.payload_slots.append(_source_slot(step, out))
        context.payloads[step.output_key] = out

    def _run_postproc(
        self,
        postproc_steps: Sequence[MCStep],
        *,
        test_summaries: Mapping[str, Any],
        regression_summaries: Mapping[str, Any],
        payload_columns: Mapping[str, list[np.ndarray]],
        reference: SolvedModel,
        dgp: SolvedModel | None,
        fail_fast: bool,
        failures: list[MCFailure],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Run POSTPROC ops once over the assembled traces; collect artifacts.

        Owns its own timing: returns ``(artifacts, postproc_elapsed_s)`` where
        the second maps each step name to its wall-clock seconds. ``traces``
        carries every keyed across-rep ndarray: the test/regression summary
        traces (shared with the result wire) plus stacked transform payloads. A
        failing op honors ``fail_fast`` (re-raise) or records an
        :class:`MCFailure` with ``rep_idx=-1`` (post-loop sentinel) and is skipped.
        """
        postproc_elapsed_s: dict[str, float] = {
            step.name: 0.0 for step in postproc_steps
        }
        if not postproc_steps:
            return {}, postproc_elapsed_s

        from .serialize import traces_from_summaries

        traces: dict[str, np.ndarray] = traces_from_summaries(
            test_summaries, regression_summaries
        )
        traces.update(_stack_payload_columns(payload_columns))

        postproc: dict[str, Any] = {}
        for step in postproc_steps:
            step_start = perf_counter()
            out: Any = None
            failed = False
            try:
                out = step.func(
                    traces=traces,
                    **dict(step.kwargs),
                )
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


def _pool_stride(n_rep: int, poolsize: int) -> tuple[bool, int]:
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


def _validate_source_producer(
    consumer: MCStep,
    selector: Any,
    producer: MCStep,
) -> None:
    if selector.source_kind == SOURCE_KIND_DATA:
        expected = OpType.DATAGEN
    elif selector.source_kind == SOURCE_KIND_PAYLOAD:
        expected = OpType.TRANSFORM
    elif selector.source_kind == SOURCE_KIND_FILTER:
        expected = OpType.FILTER
    else:
        raise ValueError(
            f"Step {consumer.name!r} has unknown source kind {selector.source_kind}."
        )
    if producer.op_type is not expected:
        raise ValueError(
            f"Step {consumer.name!r} reads field {selector.field!r} from "
            f"{producer.name!r}, but that producer is {producer.op_type.value!r}."
        )


def _source_slot(step: MCStep, out: Any) -> Any:
    if step.op_type is OpType.TRANSFORM:
        if isinstance(out, MCData):
            return (out,)
        return (_source_array(out),)
    return out


def _source_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"Transform payloads used as sources must be 1D or 2D, got shape {arr.shape}."
        )
    return arr


def _payload_to_array(value: object) -> np.ndarray | None:
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


def _accumulate_payload_columns(
    columns: dict[str, list[np.ndarray]], payloads: Mapping[str, object]
) -> None:
    for key, value in payloads.items():
        array = _payload_to_array(value)
        if array is not None:
            columns.setdefault(key, []).append(array)


def _stack_payload_columns(
    columns: Mapping[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    """Stack per-rep payload arrays into ``payload.<name>`` traces.

    Only keys whose per-rep arrays share a shape across replications are stacked
    (a transform whose output length varies per rep is skipped)."""
    from .traces import payload_trace_key

    out: dict[str, np.ndarray] = {}
    for name, arrays in columns.items():
        if arrays and len({array.shape for array in arrays}) == 1:
            out[payload_trace_key(name)] = np.stack(arrays)
    return out
