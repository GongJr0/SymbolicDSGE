from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from time import perf_counter
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from .graph import PipelineGraph
    from .spec import PipelineSpec

from .._diag_tests.result import MCResult, TestResult
from ..core.solved_model import SolvedModel
from ..kalman.filter import FilterResult
from ..regression.ols import MCRegressionResult
from ..regression.result import RegressionResult
from .mc_constructs import (
    DataGenReturn,
    MCContext,
    MCData,
    MCFailure,
    MCPipelineResult,
    MCStep,
    OpType,
    report_mc_performance,
    report_mc_step_performance,
)


@dataclass(frozen=True)
class MCPipeline:
    #: Per-replication steps: the dependency DAG, a single DATAGEN root first.
    per_rep_steps: tuple[MCStep, ...]
    #: Post-loop ops, run once after the loop over the assembled across-rep
    #: traces. A terminal phase -- not part of the graph.
    postproc_steps: tuple[MCStep, ...]

    def __init__(
        self,
        per_rep_steps: Sequence[MCStep],
        postproc_steps: Sequence[MCStep] = (),
    ) -> None:
        rep_tuple = tuple(per_rep_steps)
        postproc_tuple = tuple(postproc_steps)
        self._validate_steps(rep_tuple, postproc_tuple)
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

    @cached_property
    def graph(self) -> "PipelineGraph":
        """The pipeline's dependency DAG, resolved from the steps' kwargs.

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
        DTOs. Bulk side-channels (``raw_data`` arrays, custom-op blobs) are
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
        retain_payloads: bool = True,
        retain_test_results: bool = True,
        retain_contexts: bool = False,
        fail_fast: bool = True,
        verbosity: int = 1,
    ) -> MCPipelineResult:
        if n_rep <= 0:
            raise ValueError("n_rep must be positive.")
        if verbosity not in (0, 1, 2):
            raise ValueError("verbosity must be 0, 1, or 2.")

        contexts: list[MCContext] = []
        payload_traces: list[Mapping[str, object]] = []
        failures: list[MCFailure] = []
        results_by_step: dict[str, list[TestResult]] = {}
        regression_results_by_step: dict[str, list[RegressionResult]] = {}
        all_steps = (*self.per_rep_steps, *self.postproc_steps)
        step_elapsed_s: dict[str, float] = {step.name: 0.0 for step in all_steps}
        step_counts: dict[str, int] = {step.name: 0 for step in all_steps}
        step_failures: dict[str, int] = {step.name: 0 for step in all_steps}

        # POSTPROC ops don't run per replication — they run once after the loop,
        # over the assembled across-rep traces.
        rep_steps = self.per_rep_steps
        postproc_steps = self.postproc_steps
        payload_columns: dict[str, list[np.ndarray]] = {}

        run_start = perf_counter()
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

            contexts.append(context)
            if retain_payloads:
                payload_traces.append(dict(context.payloads))
            for name, test_result in context.results.items():
                results_by_step.setdefault(name, []).append(test_result)
            for name, regression_result in context.regressions.items():
                regression_results_by_step.setdefault(name, []).append(
                    regression_result
                )
            if postproc_steps:
                _accumulate_payload_columns(payload_columns, context.payloads)

        test_summaries = _summarize_tests(results_by_step)
        regression_summaries = _summarize_regressions(regression_results_by_step)
        postproc = self._run_postproc(
            postproc_steps,
            test_summaries=test_summaries,
            regression_summaries=regression_summaries,
            payload_columns=payload_columns,
            reference=reference,
            dgp=dgp,
            fail_fast=fail_fast,
            failures=failures,
            step_elapsed_s=step_elapsed_s,
            step_counts=step_counts,
            step_failures=step_failures,
        )
        elapsed_s = perf_counter() - run_start
        result = MCPipelineResult(
            n_rep=n_rep,
            n_successful=len(contexts),
            test_summaries=test_summaries,
            test_results=(
                {name: tuple(values) for name, values in results_by_step.items()}
                if retain_test_results
                else None
            ),
            payloads=tuple(payload_traces) if retain_payloads else None,
            contexts=tuple(contexts) if retain_contexts else None,
            failures=tuple(failures),
            regression_summaries=regression_summaries,
            elapsed_s=elapsed_s,
            step_elapsed_s=step_elapsed_s,
            step_counts=step_counts,
            step_failures=step_failures,
            postproc=postproc,
        )
        if verbosity == 1:
            report_mc_performance(result)
        elif verbosity == 2:
            report_mc_step_performance(result)
        return result

    def _run_step(self, context: MCContext, step: MCStep) -> None:
        kwargs = dict(step.kwargs)
        if step.op_type is OpType.DATAGEN:
            out = step.func(
                reference=context.reference,
                dgp=context.dgp,
                rep_idx=context.rep_idx,
                **kwargs,
            )
            if isinstance(out, DataGenReturn):
                out = MCData(
                    states=out.state_mat,
                    observables=out.obs_mat,
                    n_exog=out.n_exog,
                )
            if not isinstance(out, MCData):
                raise TypeError("DATAGEN steps must return MCData.")
            context.data = out
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
        if step.op_type is OpType.FILTER and not isinstance(out, FilterResult):
            raise TypeError("FILTER steps must return FilterResult.")
        if step.op_type is OpType.REGRESSION and not isinstance(out, RegressionResult):
            raise TypeError("REGRESSION steps must return RegressionResult.")
        if step.op_type is OpType.REGRESSION:
            context.regressions[step.name] = out  # pyright: ignore
        if step.op_type is OpType.TEST:
            if not isinstance(out, TestResult):
                raise TypeError("TEST steps must return TestResult.")
            context.results[step.name] = out
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
        step_elapsed_s: dict[str, float],
        step_counts: dict[str, int],
        step_failures: dict[str, int],
    ) -> dict[str, Any]:
        """Run POSTPROC ops once over the assembled traces; collect artifacts.

        ``traces`` carries every keyed across-rep ndarray — the test/regression
        summary traces (shared with the result wire) plus stacked transform
        payloads. A failing op honors ``fail_fast`` (re-raise) or records an
        :class:`MCFailure` with ``rep_idx=-1`` (post-loop sentinel) and is skipped.
        """
        if not postproc_steps:
            return {}

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
                    reference=reference,
                    dgp=dgp,
                    traces=traces,
                    **dict(step.kwargs),
                )
            except Exception as exc:
                failed = True
                step_failures[step.name] += 1
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
                step_elapsed_s[step.name] += perf_counter() - step_start
                step_counts[step.name] += 1
            if not failed:
                postproc[step.name] = out
        return postproc


def _payload_to_array(value: object) -> np.ndarray | None:
    """A per-rep payload value as a stackable numeric array, else ``None``.

    Only numeric ndarray / scalar payloads (e.g. transform outputs) qualify;
    structured payloads (``MCData`` / ``FilterResult`` / result objects) are
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


def _df_metadata_matches(a: object, b: object) -> bool:
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


def _summarize_tests(
    results_by_step: Mapping[str, list[TestResult]],
) -> dict[str, MCResult]:
    test_summaries: dict[str, MCResult] = {}

    for step_name, results in results_by_step.items():
        if not results:
            continue
        first = results[0]
        for result in results[1:]:
            if (
                result.dist is not first.dist
                or result.pval_method is not first.pval_method
                or not _df_metadata_matches(result.df, first.df)
                or result.alpha != first.alpha
            ):
                raise ValueError(
                    f"Test results for step '{step_name}' have incompatible metadata."
                )
        stat_trace = np.asarray(
            [result.statistic for result in results], dtype=np.float64
        )
        summary = MCResult(
            test_name=step_name,
            dist=first.dist,
            df=first.df,
            pval_method=first.pval_method,
            alpha=first.alpha,
            statistic_trace=stat_trace,
            status_trace=tuple(result.status for result in results),
        )
        test_summaries[step_name] = summary

    return test_summaries


def _summarize_regressions(
    results_by_step: Mapping[str, list[RegressionResult]],
) -> dict[str, MCRegressionResult]:
    regression_summaries: dict[str, MCRegressionResult] = {}
    for step_name, results in results_by_step.items():
        if not results:
            continue
        regression_summaries[step_name] = MCRegressionResult.from_results(results)
    return regression_summaries
