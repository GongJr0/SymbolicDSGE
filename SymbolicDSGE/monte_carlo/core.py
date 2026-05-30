from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .._diag_tests.result import MCResult, TestResult
from ..core.solved_model import SolvedModel
from ..kalman.filter import FilterResult
from ..regression.ols import MCRegressionResult, OLSResult
from .mc_constructs import (
    DataGenReturn,
    MCContext,
    MCData,
    MCFailure,
    MCPipelineResult,
    MCStep,
    OpType,
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
    ) -> MCPipelineResult:
        if n_rep <= 0:
            raise ValueError("n_rep must be positive.")

        contexts: list[MCContext] = []
        payload_traces: list[Mapping[str, object]] = []
        failures: list[MCFailure] = []
        results_by_step: dict[str, list[TestResult]] = {}
        regression_results_by_step: dict[str, list[OLSResult]] = {}

        for rep_idx in range(n_rep):
            context = MCContext(rep_idx=rep_idx, reference=reference, dgp=dgp)
            try:
                for step in self.steps:
                    self._run_step(context, step)
            except Exception as exc:
                if fail_fast:
                    raise
                failures.append(
                    MCFailure(
                        rep_idx=rep_idx,
                        step_name=step.name,  # pyright: ignore
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
        return MCPipelineResult(
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
        )

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
        if step.op_type is OpType.REGRESSION and not isinstance(out, OLSResult):
            raise TypeError("REGRESSION steps must return OLSResult.")
        if step.op_type is OpType.REGRESSION:
            context.regressions[step.name] = out
        if step.op_type is OpType.TEST:
            if not isinstance(out, TestResult):
                raise TypeError("TEST steps must return TestResult.")
            context.results[step.name] = out
        context.payloads[step.output_key] = out


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
                or result.df != first.df
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
        )
        test_summaries[step_name] = summary

    return test_summaries


def _summarize_regressions(
    results_by_step: Mapping[str, list[OLSResult]],
) -> dict[str, MCRegressionResult]:
    regression_summaries: dict[str, MCRegressionResult] = {}
    for step_name, results in results_by_step.items():
        if not results:
            continue
        regression_summaries[step_name] = MCRegressionResult.from_results(results)
    return regression_summaries
