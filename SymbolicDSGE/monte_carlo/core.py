"""Monte Carlo pipeline construction, validation and execution."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .._ckernels.monte_carlo._status import MCStatus
from .._ckernels.monte_carlo._arenas import StepArenas
from .._ckernels.monte_carlo._runner import NativeRunResult, run
from .._diag_tests.result import MCTestResult
from ..core.solved_model import SolvedModel
from ..regression.result import MCRegressionResult

from .allocation import (
    FieldLayout,
    _filter_mode,
    get_target_model,
    is_empty,
    resolve_output_specs,
)
from .defaults import DEFAULT_SIMULATION_OBSERVABLES
from .memory import MCMemoryProfiler, MCMemoryReport
from .native_lowering import LoweredMCRun, lower_native_run
from .spec import PipelineSpec
from .mc_constructs import (
    MCDataGenResult,
    MCFilterResult,
    MCFailure,
    MCPipelineResult,
    MCMeta,
    MCStep,
    OpType,
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


class MCPipeline:
    """A Monte Carlo experiment represented as a pipeline; resolving to a Direct Acyclic Graph (DAG) of steps.

    Parameters
    ----------
    *steps : MCStep
        Per-replication step specifications. Must contain exactly one DATAGEN step
        and may contain any number of FILTER, TRANSFORM, TEST, and REGRESSION steps.

    Attributes
    ----------
    steps : tuple[MCStep, ...]
        The per-replication steps in resolved execution order.
    """

    def __init__(
        self,
        *steps: MCStep,
    ) -> None:
        _validate_steps(steps)
        ordered = _order_steps(steps)
        source_indices = _resolve_source_indices(ordered)

        self.steps = ordered
        self._source_indices = source_indices

    def run(
        self,
        models: Mapping[str, SolvedModel] | None = None,
        *,
        n_rep: int,
        fail_fast: bool = False,
        verbosity: int = 1,
        n_jobs: int | None = None,
        check_memory_availability: bool = True,
    ) -> MCPipelineResult:
        """Run a Monte carlo pipeine with the specified per-replication steps.

        Parameters
        ----------
        n_rep : int
            Total number of replications to run. Must be positive.
            Steps do not necessarily keep the data from all replications.
            Refer to documentation on the monte_carlo module to learn about specifying
            high-rep runs with controlled memory usage.
        reference : SolvedModel | None
            The primary model being tested against raw data or a DGP model.
        dgp : SolvedModel | None
            Model to use as the Data Generating Process (DGP) for simulation DATAGEN steps.
            A simulation step defaults to DGP as the target but it is possible to override it to
            use the reference model instead.
        fail_fast : bool
            Whether to raise an exception immediately when any step fails. Set it to ``False`` if you expect
            occasional failures and want to keep collecting results regardless.
        verbosity : int
            Reporting granularity of performance metrics.
            0: silent, 1: report aggregate run performance, 2: report per-step, per-worker performance.
        n_jobs : int | None
            joblib-style parallelism control.
            ``None`` (default) means to use all available cores. A positive integer limits the number of workers.
            -1 means to use all available cores minus one, leaving one core free for other tasks.
        check_memory_availability : bool
            Whether to check if a run and its retainment specifications fit in available memory and swap space.
            A run will warn if it exceeds physical memory, and raise a ``MemoryError`` if it exceeds all available memory (physical + swap).
            It's not recommended to disable this check. A raise will enumerate all steps' memory usage. Use the message to trim out steps that
            are not instrumental to your results.

        Returns
        -------
        MCPipelineResult
            Result object containing run metadata, retained outputs for all step types, and the record of failures collected during the run.
            Step result accessors:
            - :attr:`MCPipelineResult.datagen_outputs`
            - :attr:`MCPipelineResult.filter_outputs`
            - :attr:`MCPipelineResult.transform_outputs`
            - :attr:`MCPipelineResult.test_summaries`
            - :attr:`MCPipelineResult.regression_summaries`

        """
        if n_rep <= 0:
            raise ValueError("n_rep must be positive.")
        if verbosity not in (0, 1, 2):
            raise ValueError("verbosity must be 0, 1, or 2.")

        prep = lower_native_run(
            self,
            models=models,
            n_rep=n_rep,
            n_jobs=n_jobs,
            check_memory_availability=check_memory_availability,
        )
        native_res = run(
            prep.allocation,
            prep.steps,
            prep.input_bindings,
            fail_fast=fail_fast,
            profile_steps=verbosity == 2,
        )
        elapsed_s = native_res.wall_elapsed_s
        if fail_fast and is_failed(native_res):
            step_name = (
                self.steps[native_res.halt_step_idx].name
                if 0 <= native_res.halt_step_idx < len(self.steps)
                else "<runner>"
            )
            status = MCStatus(native_res.halt_status)
            raise RuntimeError(
                f"Monte Carlo run failed at replication "
                f"{native_res.halt_rep_idx}, step {step_name!r}, with status "
                f"{status.name} ({status.value}): {status.message}"
            )
        failures = _resolve_failures(prep)

        tests = []
        regressions = []
        transforms = []
        filters = []
        datagens = []
        for s in self.steps:
            if s.op_type is OpType.DATAGEN:
                datagens.append(s)
            elif s.op_type is OpType.TEST:
                tests.append(s.name)
            elif s.op_type is OpType.REGRESSION:
                regressions.append(s.name)
            elif s.op_type is OpType.TRANSFORM:
                transforms.append(s.name)
            elif s.op_type is OpType.FILTER:
                filters.append(s)

        datagen_result = _compile_datagen(datagens, prep, n_rep)
        test_summaries = _compile_tests(tests, prep, n_rep)
        regression_summaries = _compile_regressions(regressions, prep, n_rep)
        payload_columns = _resolve_payloads(transforms, prep)
        filter_outputs = _compile_filters(filters, prep, n_rep)

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
            failed_steps=failed_step_counts(failures),
        )
        result = MCPipelineResult(
            n_rep=n_rep,
            meta=meta,
            datagen_outputs=datagen_result,
            filter_outputs=filter_outputs,
            n_successful=int(
                np.count_nonzero((prep.allocation.step_status_by_rep == 0).all(axis=1))
            ),
            test_summaries=test_summaries,
            transform_outputs=payload_columns,
            failures=tuple(failures),
            regression_summaries=regression_summaries,
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

    def validate_memory_requirements(
        self,
        *,
        models: Mapping[str, SolvedModel] | None = None,
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
        plan = resolve_output_specs(self.steps, self._source_indices, models)
        return MCMemoryProfiler(
            plan,
            self.steps,
            models=models,
            n_rep=n_rep,
            n_jobs=n_jobs,
        ).validate()

    def to_spec(self) -> PipelineSpec:
        return PipelineSpec(
            steps=[s.to_spec() for s in self.steps],
        )

    @classmethod
    def from_spec(cls, spec: PipelineSpec) -> "MCPipeline":
        return cls(*[MCStep.from_spec(s) for s in spec.steps])


def _order_steps(
    steps: Sequence[MCStep],
) -> tuple[MCStep, ...]:
    """Kahn-walk the steps to sort them in source dependecy order."""
    names = {step.name for step in steps}
    remaining = list(steps)
    ordered: list[MCStep] = []
    placed: set[str] = set()
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
            raise ValueError(f"Step source dependency cycle among {stuck}.")
        remaining = next_remaining
    return tuple(ordered)


def _validate_steps(steps: Sequence[MCStep]) -> None:
    if not steps:
        raise ValueError("MCPipeline requires at least one per-replication step.")
    names = [step.name for step in steps]
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


def _resolve_source_indices(
    steps: tuple[MCStep, ...],
) -> tuple[tuple[int, ...], ...]:
    index_by_name = {step.name: index for index, step in enumerate(steps)}

    resolved: list[tuple[int, ...]] = []
    for step_index, step in enumerate(steps):
        step_indices: list[int] = []
        for selector in step.source_args:
            source_name = selector.source_step
            source_idx = index_by_name.get(source_name)
            if source_idx is None:
                raise ValueError(
                    f"Step {step.name!r} depends on unknown producer {source_name!r}."
                )
            producer_step = steps[source_idx]
            if source_idx >= step_index:
                raise ValueError(
                    f"Step {step.name!r} depends on {producer_step.name!r}, which does not "
                    "appear earlier in the pipeline."
                )
            step_indices.append(source_idx)
        resolved.append(tuple(step_indices))
    return tuple(resolved)


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


def _datagen_names(
    step: MCStep, models: Mapping[str, SolvedModel] | None
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """The variable, shock and observable names of one datagen's columns.

    What the arena cannot say: which model a column belongs to. A simulation
    takes its names from the model it targets, which is the DGP unless the step
    named the reference, and raw data takes them from what its author declared.
    """
    if step.step_type == "raw_model_data":
        return (
            tuple(step.kwargs.get("state_names") or ()),
            tuple(step.kwargs.get("shock_names") or ()),
            tuple(step.kwargs.get("observable_names") or ()),
        )
    if step.step_type != "simulation":
        raise NotImplementedError(
            f"Result semantics are not resolved for datagen step type "
            f"{step.step_type!r}."
        )
    model = get_target_model(step, models)
    comp = model.compiled
    return (
        tuple(comp.var_names),
        tuple(comp.shock_names),
        (
            tuple(comp.observable_names)
            if step.kwargs.get("observables", DEFAULT_SIMULATION_OBSERVABLES)
            else ()
        ),
    )


def _compile_datagen(
    datagen_steps: Sequence[MCStep], lowered: LoweredMCRun, n_rep: int
) -> dict[str, MCDataGenResult]:
    """Read each datagen step's retained fields out of its arena.

    Only declared fields are stored. Missing fields return NaN blocks at the
    width implied by their names.
    """
    results: dict[str, MCDataGenResult] = {}
    for step in datagen_steps:
        layout = lowered.plan[step.name].out_fields
        arena = lowered.allocation.steps[step.name]
        n_retained = int(arena.retained_reps.size)

        def read(field: str, width: int) -> NDF:
            if not _has_field(layout, field):
                return np.full((n_retained, T, width), np.nan)
            return _read_float_field(arena, layout, field)

        # Every field shares its leading axis, so any one of them dates the run.
        T = next(
            (
                entry.shape[0]
                for name, entry in layout.items()
                if entry.dtype == np.float64 and _has_field(layout, name)
            ),
            0,
        )
        var_names, shock_names, observable_names = _datagen_names(step, lowered.models)

        results[step.name] = MCDataGenResult(
            n_rep=n_rep,
            n_retained=n_retained,
            retained_reps=arena.retained_reps,
            var_names=var_names,
            X=read("states", len(var_names)),
            shock_names=shock_names,
            eps=read("shocks", len(shock_names)),
            observable_names=observable_names,
            y=read("observables", len(observable_names)),
        )

    return results


def _compile_filters(
    filter_steps: Sequence[MCStep], lowered: LoweredMCRun, n_rep: int
) -> dict[str, MCFilterResult]:
    """Read each filter step's retained fields out of its arena.

    Takes the steps rather than their names because the mode selected the
    kernel at lowering time and is not recorded on anything the run leaves
    behind, and it is also what says whether the pruned-state fields exist.
    """
    summaries: dict[str, MCFilterResult] = {}
    arenas = lowered.allocation.steps

    for step in filter_steps:
        name = step.name
        mode = _filter_mode(step)
        layout = lowered.plan[name].out_fields
        arena = arenas[name]

        x_pred = _read_float_field(arena, layout, "x_pred")
        x_filt = _read_float_field(arena, layout, "x_filt")
        y_pred = _read_float_field(arena, layout, "y_pred")
        y_filt = _read_float_field(arena, layout, "y_filt")

        P_pred = _read_float_field(arena, layout, "P_pred")
        P_filt = _read_float_field(arena, layout, "P_filt")

        S = _read_float_field(arena, layout, "S")
        eps_hat = _read_float_field(arena, layout, "eps_hat")

        innov = _read_float_field(arena, layout, "innov")
        std_innov = _read_float_field(arena, layout, "std_innov")

        loglik = _read_float_field(arena, layout, "loglik")

        if mode == "unscented":
            x1_pred = _read_float_field(arena, layout, "x1_pred")
            x2_pred = _read_float_field(arena, layout, "x2_pred")
            x1_filt = _read_float_field(arena, layout, "x1_filt")
            x2_filt = _read_float_field(arena, layout, "x2_filt")
        else:
            x1_pred = None
            x2_pred = None
            x1_filt = None
            x2_filt = None

        summaries[name] = MCFilterResult(
            n_rep=n_rep,
            n_retained=arena.retained_reps.size,
            retained_reps=arena.retained_reps,
            filter_mode=mode,
            x_pred=x_pred,
            x_filt=x_filt,
            y_pred=y_pred,
            y_filt=y_filt,
            P_pred=P_pred,
            P_filt=P_filt,
            S=S,
            innov=innov,
            std_innov=std_innov,
            loglik=loglik,
            eps_hat=eps_hat,
            _x1_pred=x1_pred,
            _x2_pred=x2_pred,
            _x1_filt=x1_filt,
            _x2_filt=x2_filt,
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
        coef_trace = _read_float_field(arena, layout, "coef")
        ssr_trace = _read_float_field(arena, layout, "ssr")
        sst_trace = _read_float_field(arena, layout, "sst")

        if not is_empty(layout["se"]):
            se_trace = _read_float_field(arena, layout, "se")
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
    """Project native per-replication runner failures onto public failures.

    The runner records one status per step per replication, so a replication
    reports every step that did not produce output rather than only the first.
    A step the runner declined for reading from a failed source records
    ``MC_NOT_RUN``, which distinguishes a casualty from the failure that
    originated it.

    The record covers every replication, including those the plan did not
    retain, whose statuses reach nowhere else.
    """
    statuses = lowered.allocation.step_status_by_rep
    step_names = [step.name for step in lowered.steps]
    failures: list[MCFailure] = []

    for rep_idx, row in enumerate(statuses):
        failed = {
            name: MCStatus(status)
            for name, status in zip(step_names, row, strict=True)
            if status != 0
        }
        if not failed:
            continue
        failures.append(MCFailure(rep_idx=rep_idx, failures=failed))
    return failures


def _has_field(layout: Mapping[str, FieldLayout], field: str) -> bool:
    """Whether a step's layout reserved a non-empty buffer for a field."""
    entry = layout.get(field)
    return entry is not None and not is_empty(entry)


def _read_float_field(
    arena: StepArenas, layout: Mapping[str, FieldLayout], field: str
) -> NDF:
    """One retained float field, shaped ``(n_retained, *entry.shape)``.

    The arena lane is flat, so the field's planned shape is what restores its
    axes. A scalar field lands as ``(n_retained,)``.
    """
    entry = layout[field]
    n_retained = int(arena.retained_reps.size)
    if n_retained == 0:
        return np.empty((0, *entry.shape), dtype=np.float64)
    flat = arena.float_retained[:, entry.offset : entry.offset + entry.flat_count]
    return flat.reshape(n_retained, *entry.shape)
