from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from time import perf_counter
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from .graph import PipelineGraph

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
    steps: tuple[MCStep, ...]

    def __init__(self, steps: Sequence[MCStep]) -> None:
        if not steps:
            raise ValueError("MCPipeline requires at least one step.")
        step_tuple = tuple(steps)
        self._validate_steps(step_tuple)
        object.__setattr__(self, "steps", step_tuple)

    @staticmethod
    def _validate_steps(steps: tuple[MCStep, ...]) -> None:
        names = [step.name for step in steps]
        if len(set(names)) != len(names):
            raise ValueError("MCPipeline step names must be unique.")
        if steps[0].op_type is not OpType.DATAGEN:
            raise ValueError("MCPipeline first step must be a DATAGEN step.")
        for step in steps[1:]:
            if step.op_type is OpType.DATAGEN:
                raise ValueError(
                    "MCPipeline supports only one DATAGEN step, in first position."
                )

    @cached_property
    def graph(self) -> "PipelineGraph":
        """The pipeline's dependency DAG, resolved from the steps' kwargs.

        Built once and cached. Owns the graph structure (parents/children/leaves/
        typed input edges) that serialization and validation read instead of
        re-deriving it. Lazily imported to keep ``core`` light at import time.
        """
        from .graph import PipelineGraph

        return PipelineGraph.from_steps(self.steps)

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
        step_elapsed_s: dict[str, float] = {step.name: 0.0 for step in self.steps}
        step_counts: dict[str, int] = {step.name: 0 for step in self.steps}
        step_failures: dict[str, int] = {step.name: 0 for step in self.steps}

        run_start = perf_counter()
        for rep_idx in range(n_rep):
            context = MCContext(rep_idx=rep_idx, reference=reference, dgp=dgp)
            failed_step_name: str | None = None
            try:
                for step in self.steps:
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

        test_summaries = _summarize_tests(results_by_step)
        regression_summaries = _summarize_regressions(regression_results_by_step)
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
