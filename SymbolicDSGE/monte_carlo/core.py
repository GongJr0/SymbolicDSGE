from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from time import perf_counter
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .graph import PipelineGraph
    from .memory import MCMemoryReport
    from .native_lowering import LoweredMCRun
    from .spec import PipelineSpec

from .._ckernels.monte_carlo._runner import NativeRunResult, run
from .._diag_tests.result import MCTestResult
from ..core.solved_model import SolvedModel
from ..regression.ols import MCRegressionResult
from .allocation import BufferPlan, resolve_output_specs
from .traces import traces_from_summaries, trace_keys_for_step
from .postproc import Artifact, normalize_artifacts
from .catalog import STEP_CATALOG
from .mc_constructs import (
    DYNAMIC_SOURCE_FIELDS,
    FILTER_RAW_SOURCE_FIELDS,
    MC_DATA_SOURCE_FIELDS,
    MCFailure,
    MCPipelineResult,
    MCMeta,
    MCStep,
    OpType,
    failed_postproc_names,
    failed_step_counts,
    report_mc_performance,
    report_mc_step_performance,
)

NDF = NDArray[np.float64]

#: Characters a step name may not contain. A step name is written into bundle
#: member paths and prefixes its trace columns as ``{step}.{field}``, so path
#: separators and the column separator are spoken for.
_RESERVED_NAME_CHARS = frozenset(".:/\\")


def is_failed(native_run: NativeRunResult) -> bool:
    """Whether the native runner stopped with a non-success status."""
    return native_run.status != 0


@dataclass(frozen=True)
class MCPipeline:
    #: Per-replication steps: the dependency DAG, a single DATAGEN root first.
    per_rep_steps: tuple[MCStep, ...]

    #: Post-loop ops, run once after the loop over the assembled across-rep
    #: traces. This is a terminal phase, not part of the graph.
    postproc_steps: tuple[MCStep, ...]

    #: Producer indices for each per-replication step's source arguments.
    _source_indices: tuple[tuple[int, ...], ...]

    def __init__(
        self,
        per_rep_steps: Sequence[MCStep],
        postproc_steps: Sequence[MCStep] = (),
    ) -> None:
        rep_tuple = tuple(per_rep_steps)
        postproc_tuple = tuple(postproc_steps)
        self._validate_steps(rep_tuple, postproc_tuple)
        ordered = self._order_steps(rep_tuple)
        source_indices = self._resolve_source_indices(ordered)
        self._validate_postproc_traces(ordered, postproc_tuple)
        object.__setattr__(self, "per_rep_steps", ordered)
        object.__setattr__(self, "postproc_steps", postproc_tuple)
        object.__setattr__(self, "_source_indices", source_indices)

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
        for name in names:
            bad = sorted(set(name) & _RESERVED_NAME_CHARS)
            if bad:
                raise ValueError(
                    f"MCPipeline step name {name!r} uses reserved characters "
                    f"{''.join(bad)!r}. A step name becomes a bundle member path "
                    f"and a trace column qualifier, which reserve them."
                )
        datagens = [step for step in per_rep_steps if step.op_type is OpType.DATAGEN]
        if len(datagens) != 1:
            raise ValueError("MCPipeline requires exactly one DATAGEN step.")
        for step in per_rep_steps:
            if step.op_type is OpType.POSTPROC:
                raise ValueError(
                    "POSTPROC steps can't be specified under per_rep_steps, use postproc_steps."
                )
        for step in postproc_steps:
            if step.op_type is not OpType.POSTPROC:
                raise ValueError(
                    f"postproc_steps may only contain POSTPROC steps; {step.name!r} "
                    f"is {step.op_type}."
                )

    @staticmethod
    def _order_steps(per_rep_steps: tuple[MCStep, ...]) -> tuple[MCStep, ...]:
        """Sort the steps into execution order: datagen, filters, transforms, terminals.

        A caller authors a step list, not a schedule. Filters read only the
        datagen and terminals are read by no one, so those phases keep their
        authored order; transforms chain, so they are walked against theirs.
        """
        datagen: list[MCStep] = []
        filters: list[MCStep] = []
        transforms: list[MCStep] = []
        terminals: list[MCStep] = []
        for step in per_rep_steps:
            if step.op_type is OpType.DATAGEN:
                datagen.append(step)
            elif step.op_type is OpType.FILTER:
                filters.append(step)
            elif step.op_type is OpType.TRANSFORM:
                transforms.append(step)
            else:
                terminals.append(step)
        placed = {step.name for step in (*datagen, *filters)}
        return (*datagen, *filters, *_order_transforms(transforms, placed), *terminals)

    @staticmethod
    def _resolve_source_indices(
        per_rep_steps: tuple[MCStep, ...],
    ) -> tuple[tuple[int, ...], ...]:
        index_by_name = {step.name: index for index, step in enumerate(per_rep_steps)}

        resolved: list[tuple[int, ...]] = []
        for step_index, step in enumerate(per_rep_steps):
            step_indices: list[int] = []
            for selector in step.source_args:
                source_name = selector.source_step
                source_idx = index_by_name.get(source_name)
                if source_idx is None:
                    raise ValueError(
                        f"Step {step.name!r} depends on unknown producer {source_name!r}."
                    )
                producer_step = per_rep_steps[source_idx]
                if source_idx >= step_index:
                    raise ValueError(
                        f"Step {step.name!r} depends on {producer_step.name!r}, which does not "
                        "appear earlier in the pipeline."
                    )
                _validate_source_producer(step, selector, producer_step)
                step_indices.append(source_idx)
            resolved.append(tuple(step_indices))
        return tuple(resolved)

    @staticmethod
    def _validate_postproc_traces(
        per_rep_steps: tuple[MCStep, ...],
        postproc_steps: tuple[MCStep, ...],
    ) -> None:
        """Check each postproc's trace selectors against what the producers emit.

        Catalogue postprocs mark their trace fields with ``type == "trace"``. A
        custom op reads traces in opaque code, so it carries no such field and is
        left to fail at its own hands.
        """
        if not postproc_steps:
            return

        available = {key for step in per_rep_steps for key in trace_keys_for_step(step)}
        for step in postproc_steps:
            definition = STEP_CATALOG.get(step.step_type or "")
            if definition is None:
                continue
            for field in definition.fields:
                if field.type != "trace":
                    continue
                ref = step.kwargs.get(field.key)
                if not ref:
                    raise ValueError(
                        f"POSTPROC step {step.name!r} must select a trace for "
                        f"{field.key!r} (available: {sorted(available)})."
                    )
                if ref not in available:
                    raise ValueError(
                        f"POSTPROC step {step.name!r} field {field.key!r} references "
                        f"trace {ref!r}, which no step in the pipeline produces "
                        f"(available: {sorted(available)})."
                    )

    @cached_property
    def graph(self) -> "PipelineGraph":
        """The pipeline's dependency DAG, resolved from compiled source args.

        Built once and cached. Owns the graph structure (parents/children/leaves/
        typed input edges) that serialization and validation read instead of
        re-deriving it. Lazily imported to keep ``core`` light at import time.
        """
        from .graph import PipelineGraph

        return PipelineGraph.from_steps(self.per_rep_steps, self._source_indices)

    def _resolve_output_specs(
        self,
        reference: SolvedModel,
        dgp: SolvedModel | None,
    ) -> BufferPlan:
        return resolve_output_specs(
            self.per_rep_steps, self._source_indices, reference, dgp
        )

    def lower_native(
        self,
        *,
        reference: SolvedModel,
        dgp: SolvedModel | None = None,
        n_rep: int,
        n_jobs: int | None = None,
        check_memory_availability: bool = True,
    ) -> "LoweredMCRun":
        """Resolve one native runner invocation without executing it."""
        from .native_lowering import lower_native_run

        return lower_native_run(
            self,
            reference=reference,
            dgp=dgp,
            n_rep=n_rep,
            n_jobs=n_jobs,
            check_memory_availability=check_memory_availability,
        )

    def validate_memory_requirements(
        self,
        *,
        reference: SolvedModel,
        dgp: SolvedModel | None = None,
        n_rep: int,
        n_jobs: int | None = None,
    ) -> "MCMemoryReport":
        """Report what these run arguments would allocate, before running them.

        Takes the arguments :meth:`run` takes, since the buffer plan they size
        does not exist without them. Warns when the run spills past physical
        memory, which costs throughput and nothing else, and raises
        :class:`MemoryError` only when it does not fit with swap counted too.
        The breakdown is printed before the raise so the numbers land above the
        traceback rather than after it. The returned report carries that same
        breakdown, naming no step as the one to shrink: which traces are worth
        their memory is not a question the step graph can answer.
        """
        from .memory import MCMemoryProfiler

        plan = self._resolve_output_specs(reference, dgp)
        return MCMemoryProfiler(
            plan,
            self.per_rep_steps,
            reference=reference,
            dgp=dgp,
            n_rep=n_rep,
            n_jobs=n_jobs,
        ).validate()

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
        reference: SolvedModel,
        dgp: SolvedModel | None = None,
        *,
        n_rep: int,
        fail_fast: bool = True,
        verbosity: int = 1,
        n_jobs: int | None = None,
        check_memory_availability: bool = True,
    ) -> MCPipelineResult:
        if n_rep <= 0:
            raise ValueError("n_rep must be positive.")
        if verbosity not in (0, 1, 2):
            raise ValueError("verbosity must be 0, 1, or 2.")

        prep = self.lower_native(
            reference=reference,
            dgp=dgp,
            n_rep=n_rep,
            n_jobs=n_jobs,
            check_memory_availability=check_memory_availability,
        )
        loop_started_s = perf_counter()
        native_res = run(
            prep.allocation,
            prep.steps,
            prep.input_bindings,
            fail_fast=fail_fast,
            profile_steps=verbosity == 2,
        )
        elapsed_s = (
            native_res.wall_elapsed_s
            if verbosity == 2
            else perf_counter() - loop_started_s
        )
        if fail_fast and is_failed(native_res):
            step_name = (
                self.per_rep_steps[native_res.halt_step_idx].name
                if 0 <= native_res.halt_step_idx < len(self.per_rep_steps)
                else "<runner>"
            )
            raise RuntimeError(
                f"Monte Carlo run failed at replication "
                f"{native_res.halt_rep_idx}, step {step_name!r}, with status "
                f"{native_res.halt_status}."
            )
        failures = _resolve_failures(prep)

        tests = []
        regressions = []
        transforms = []
        for s in self.per_rep_steps:
            if s.op_type is OpType.TEST:
                tests.append(s.name)
            elif s.op_type is OpType.REGRESSION:
                regressions.append(s.name)
            elif s.op_type is OpType.TRANSFORM:
                transforms.append(s.name)

        test_summaries = _compile_tests(tests, prep, n_rep)
        regression_summaries = _compile_regressions(regressions, prep, n_rep)
        payload_columns = _resolve_payloads(transforms, prep)

        postprocs, postproc_wall_times = self._run_postproc(
            self.postproc_steps,
            test_summaries=test_summaries,
            regression_summaries=regression_summaries,
            payload_columns=payload_columns,
            fail_fast=fail_fast,
            failures=failures,
        )

        step_names = tuple(step.name for step in prep.steps)
        if verbosity == 2:
            if (
                native_res.step_elapsed_s_by_worker is None
                or native_res.step_counts_by_worker is None
                or native_res.step_failures_by_worker is None
            ):
                raise RuntimeError("Native step profiling was not collected.")
            step_elapsed_s = {
                name: float(elapsed)
                for name, elapsed in zip(
                    step_names,
                    native_res.step_elapsed_s_by_worker.sum(axis=0),
                    strict=True,
                )
            }
            step_counts = {
                name: int(count)
                for name, count in zip(
                    step_names,
                    native_res.step_counts_by_worker.sum(axis=0),
                    strict=True,
                )
            }
            step_failures = {
                name: int(count)
                for name, count in zip(
                    step_names,
                    native_res.step_failures_by_worker.sum(axis=0),
                    strict=True,
                )
            }
        else:
            step_elapsed_s = {}
            step_counts = {}
            step_failures = {}

        meta = MCMeta(
            n_rep=n_rep,
            n_retained_by_step={
                name: int(arena.retained_reps.size)
                for name, arena in prep.allocation.steps.items()
            },
            elapsed_s=elapsed_s,
            step_elapsed_s=step_elapsed_s,
            step_counts=step_counts,
            step_failures=step_failures,
            postproc_elapsed_s=postproc_wall_times,
            failed_steps=failed_step_counts(failures),
            failed_postprocs=failed_postproc_names(failures),
        )
        result = MCPipelineResult(
            n_rep=n_rep,
            meta=meta,
            n_successful=int(
                np.count_nonzero(prep.allocation.failure_status_by_rep == 0)
            ),
            test_summaries=test_summaries,
            transform_outputs=payload_columns,
            failures=tuple(failures),
            regression_summaries=regression_summaries,
            postproc=postprocs,
            run_config={
                "n_rep": int(n_rep),
                "fail_fast": bool(fail_fast),
                "verbosity": int(verbosity),
                "n_jobs": int(n_jobs) if n_jobs is not None else None,
                "check_memory_availability": bool(check_memory_availability),
            },
        )
        if verbosity == 1:
            report_mc_performance(meta)
        elif verbosity == 2:
            report_mc_step_performance(meta)
        return result

    def _run_postproc(
        self,
        postproc_steps: Sequence[MCStep],
        *,
        test_summaries: Mapping[str, MCTestResult],
        regression_summaries: Mapping[str, MCRegressionResult],
        payload_columns: Mapping[str, NDF],
        fail_fast: bool,
        failures: list[MCFailure],
    ) -> tuple[dict[str, Artifact], dict[str, float]]:
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

        traces: dict[str, np.ndarray] = traces_from_summaries(
            test_summaries, regression_summaries
        )
        keyed_payloads = {
            "payload." + name: arr for name, arr in payload_columns.items()
        }
        traces.update(keyed_payloads)

        postproc: dict[str, Any] = {}
        for step in postproc_steps:
            if step.func is None:
                raise ValueError(
                    f"POSTPROC step {step.name!r} has no callable function to run."
                )
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
                postproc[step.name] = normalize_artifacts(out)
        return postproc, postproc_elapsed_s


def _order_transforms(
    transforms: Sequence[MCStep],
    placed: set[str],
) -> list[MCStep]:
    """Kahn-walk the transform phase so each step follows the transforms it reads.

    ``placed`` seeds the walk with the earlier phases. A producer outside the
    transform phase is left to :meth:`MCPipeline._resolve_source_indices`, which
    owns the unknown-producer and wrong-producer-type errors.
    """
    names = {step.name for step in transforms}
    remaining = list(transforms)
    ordered: list[MCStep] = []
    placed = set(placed)
    while remaining:
        progress = False
        next_remaining: list[MCStep] = []
        for step in remaining:
            deps = {
                selector.source_step
                for selector in step.source_args
                if selector.source_step in names
            }
            if deps <= placed:
                ordered.append(step)
                placed.add(step.name)
                progress = True
            else:
                next_remaining.append(step)
        if not progress:
            stuck = [step.name for step in next_remaining]
            raise ValueError(f"Transform dependency cycle among {stuck}.")
        remaining = next_remaining
    return ordered


def _validate_source_producer(
    consumer: MCStep,
    selector: Any,
    producer: MCStep,
) -> None:
    if selector.field in MC_DATA_SOURCE_FIELDS:
        expected = OpType.DATAGEN
    elif selector.field in DYNAMIC_SOURCE_FIELDS:
        expected = OpType.TRANSFORM
    elif selector.field in FILTER_RAW_SOURCE_FIELDS:
        expected = OpType.FILTER
    else:
        raise ValueError(
            f"Step {consumer.name!r} has unknown source field {selector.field!r}."
        )
    if producer.op_type is not expected:
        raise ValueError(
            f"Step {consumer.name!r} reads field {selector.field!r} from "
            f"{producer.name!r}, but that producer is {producer.op_type.value!r}."
        )


def _compile_tests(
    test_names: Sequence[str],
    lowered: LoweredMCRun,
    n_rep: int,
) -> dict[str, MCTestResult]:
    summaries: dict[str, MCTestResult] = {}
    arenas = lowered.allocation.steps
    metas = lowered.test_result_specs

    for name in test_names:
        layout = lowered.plan[name].out_fields
        arena = arenas[name]
        spec = metas[name]

        status = (
            arena.int_retained[:, layout["status"].offset]
            if arena.retained_reps.size > 0
            else np.empty((0,), dtype=np.int64)
        )
        statistic = (
            arena.float_retained[:, layout["statistic"].offset]
            if arena.retained_reps.size > 0
            else np.empty((0,), dtype=np.float64)
        )

        summaries[name] = MCTestResult(
            test_name=spec.name,
            dist=spec.dist,
            df=spec.df,
            pval_method=spec.pval_method,
            alpha=spec.alpha,
            statistic_trace=statistic,
            n_retained=arena.retained_reps.size,
            n_rep=n_rep,
            retained_reps=arena.retained_reps,
            _raw_status=status,
        )
    return summaries


def _resolve_payloads(
    transform_names: list[str], lowered: LoweredMCRun
) -> dict[str, NDF]:
    payloads: dict[str, NDF] = {}
    arenas = lowered.allocation.steps
    for name in transform_names:
        layout = lowered.plan[name].out_fields["payload"]
        arena = arenas[name]
        if arena.retained_reps.size > 0:
            flat = arena.float_retained[
                :, layout.offset : layout.offset + layout.flat_count
            ]
            payloads[name] = flat.reshape(arena.retained_reps.size, *layout.shape)
        else:
            payloads[name] = np.empty((0, *layout.shape), dtype=np.float64)
    return payloads


def _compile_regressions(
    regression_names: Sequence[str],
    lowered: LoweredMCRun,
    n_rep: int,
) -> dict[str, MCRegressionResult]:
    summaries: dict[str, MCRegressionResult] = {}
    arenas = lowered.allocation.steps
    metas = lowered.regression_result_specs

    for name in regression_names:
        layout = lowered.plan[name].out_fields
        arena = arenas[name]
        spec = metas[name]

        status = (
            arena.int_retained[:, layout["status"].offset]
            if arena.retained_reps.size > 0
            else np.empty((0,), dtype=np.int64)
        )
        coef_trace = (
            arena.float_retained[
                :, layout["coef"].offset : layout["coef"].offset + spec.k
            ]
            if arena.retained_reps.size > 0
            else np.empty((0, spec.k), dtype=np.float64)
        )
        ssr_trace = (
            arena.float_retained[:, layout["ssr"].offset]
            if arena.retained_reps.size > 0
            else np.empty((0,), dtype=np.float64)
        )
        sst_trace = (
            arena.float_retained[:, layout["sst"].offset]
            if arena.retained_reps.size > 0
            else np.empty((0,), dtype=np.float64)
        )

        if spec.kind == "ols":
            se_trace = (
                arena.float_retained[
                    :, layout["se"].offset : layout["se"].offset + spec.k
                ]
                if arena.retained_reps.size > 0
                else np.empty((0, spec.k), dtype=np.float64)
            )
        else:
            se_trace = None

        summaries[name] = MCRegressionResult(
            kind=spec.kind,
            variables=spec.variables,
            coef_trace=coef_trace,
            ssr_trace=ssr_trace,
            sst_trace=sst_trace,
            _se_trace=se_trace,
            n_retained=arena.retained_reps.size,
            retained_reps=arena.retained_reps,
            n_rep=n_rep,
            n=spec.n,
            k=spec.k,
            _raw_status=status,
        )
    return summaries


def _resolve_failures(lowered: LoweredMCRun) -> list[MCFailure]:
    """Project native per-replication runner failures onto public failures."""
    step_names = tuple(step.name for step in lowered.steps)
    failure_steps = lowered.allocation.failure_step_by_rep
    failure_statuses = lowered.allocation.failure_status_by_rep
    failures: list[MCFailure] = []

    for rep_idx, (step_idx, status) in enumerate(
        zip(failure_steps, failure_statuses, strict=True)
    ):
        if status == 0:
            continue
        if step_idx < 0:
            continue
        if step_idx >= len(step_names):
            raise RuntimeError(
                f"Native runner reported an invalid step index {step_idx} "
                f"for replication {rep_idx}."
            )
        failures.append(
            MCFailure(
                rep_idx=rep_idx,
                step_name=step_names[step_idx],
                error_type="NativeStepError",
                message=f"Native step returned status {status}.",
            )
        )
    return failures
