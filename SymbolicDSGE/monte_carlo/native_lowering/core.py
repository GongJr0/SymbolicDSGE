"""Shared native Monte Carlo lowering orchestration and input bindings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from ..._diag_tests.distributions import PvalMethod
from ..._ckernels.monte_carlo._arenas import ArenaAllocation, allocate_arenas
from ..._ckernels.monte_carlo._runner import (
    NativeStep,
    payload_step,
    raw_model_data_step,
    transform_step,
)
from ..allocation import BufferPlan
from ..custom_op import NumbaCustomFunc
from ..mc_constructs import MCStep, OpType
from ..memory import MCMemoryProfiler
from .diagnostics import lower_test_step
from .filters import lower_filter_step
from .regressions import lower_regression_step, regression_result_spec
from .simulation import lower_simulation_step
from .utils import (
    FloatInputBinding,
    RegressionResultSpec,
    TestResultSpec,
    _check_raw_model_data_layout,
    _selected_shape,
    _source_binding,
    _supplied,
)

if TYPE_CHECKING:
    from ...core.solved_model import SolvedModel
    from ..core import MCPipeline


@dataclass(frozen=True)
class LoweredMCRun:
    """Run-local native inputs produced from one resolved pipeline."""

    plan: BufferPlan
    allocation: ArenaAllocation
    steps: tuple[NativeStep, ...]
    input_bindings: tuple[tuple[FloatInputBinding, ...], ...]
    test_result_specs: Mapping[str, TestResultSpec]
    regression_result_specs: Mapping[str, RegressionResultSpec]
    reference: SolvedModel
    dgp: SolvedModel | None


def lower_native_run(
    pipeline: MCPipeline,
    *,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    n_rep: int,
    n_jobs: int | None = None,
    check_memory_availability: bool = True,
) -> LoweredMCRun:
    """Resolve and allocate one native run without entering the runner loop."""
    if n_rep <= 0:
        raise ValueError("n_rep must be positive.")

    plan = pipeline._resolve_output_specs(reference, dgp)
    if check_memory_availability:
        MCMemoryProfiler(
            plan,
            pipeline.per_rep_steps,
            reference=reference,
            dgp=dgp,
            n_rep=n_rep,
            n_jobs=n_jobs,
        ).validate()
    allocation = allocate_arenas(plan, n_rep, n_jobs=n_jobs)
    steps: list[NativeStep] = []
    bindings: list[tuple[FloatInputBinding, ...]] = []
    test_result_specs: dict[str, TestResultSpec] = {}
    regression_result_specs: dict[str, RegressionResultSpec] = {}
    for step_idx, step in enumerate(pipeline.per_rep_steps):
        native_step, step_bindings = _lower_step(
            step_idx,
            step,
            pipeline.per_rep_steps,
            pipeline._source_indices[step_idx],
            plan,
            reference,
            dgp,
            n_rep,
        )
        steps.append(native_step)
        bindings.append(step_bindings)
        if step.op_type is OpType.TEST:
            dist = native_step.test_distribution
            df = native_step.test_df
            if dist is None or df is None:
                raise RuntimeError(
                    f"Native diagnostic {step.name!r} has no result metadata."
                )
            test_result_specs[step.name] = TestResultSpec(
                name=step.name,
                dist=dist,
                df=df,
                pval_method=PvalMethod.SF,
                **_supplied(step.kwargs, "alpha"),
            )
        elif step.op_type is OpType.REGRESSION:
            regression_result_specs[step.name] = regression_result_spec(
                step,
                pipeline._source_indices[step_idx],
                pipeline.per_rep_steps,
                plan,
            )
    return LoweredMCRun(
        plan,
        allocation,
        tuple(steps),
        tuple(bindings),
        test_result_specs,
        regression_result_specs,
        reference,
        dgp,
    )


def _lower_step(
    step_idx: int,
    step: MCStep,
    steps: tuple[MCStep, ...],
    source_indices: tuple[int, ...],
    plan: BufferPlan,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    n_rep: int,
) -> tuple[NativeStep, tuple[FloatInputBinding, ...]]:
    match step.op_type:
        case OpType.DATAGEN:
            return _lower_datagen_step(step, plan, reference, dgp, n_rep)
        case OpType.TRANSFORM:
            return _lower_transform_step(step, source_indices, steps, plan)
        case OpType.FILTER:
            return lower_filter_step(step, steps[0], plan, reference, dgp)
        case OpType.REGRESSION:
            return lower_regression_step(step, source_indices, steps, plan)
        case OpType.TEST:
            return lower_test_step(step, source_indices, steps, plan)
        case _:
            raise NotImplementedError(
                "Unrecognized Monte Carlo step type: " f"{step.op_type!r}."
            )


def _lower_datagen_step(
    step: MCStep,
    plan: BufferPlan,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    n_rep: int,
) -> tuple[NativeStep, tuple[FloatInputBinding, ...]]:
    step_plan = plan[step.name]
    if step.step_type == "raw_model_data":
        _check_raw_model_data_layout(step, step_plan.out_fields)
        return (
            raw_model_data_step(
                step.name,
                states=step.kwargs["states"],
                observables=step.kwargs["observables"],
            ),
            (),
        )
    if step.step_type == "simulation":
        return lower_simulation_step(step, step_plan, reference, dgp, n_rep)
    raise NotImplementedError(
        f"Native lowering is not implemented for {step.name!r} ({step.step_type!r})."
    )


def _lower_transform_step(
    step: MCStep,
    source_indices: tuple[int, ...],
    steps: tuple[MCStep, ...],
    plan: BufferPlan,
) -> tuple[NativeStep, tuple[FloatInputBinding, ...]]:
    if step.step_type == "payload":
        return payload_step(step.name, step.kwargs["value"]), ()

    n, p = _selected_shape(source_indices[0], step.source_args[0], steps, plan)
    if step.step_type == "transform:custom":
        if not isinstance(step.func, NumbaCustomFunc):
            raise ValueError(
                f"Native custom transform {step.name!r} requires a NumbaCustomFunc."
            )
        output_layout = plan[step.name].out_fields["payload"]
        if len(output_layout.shape) != 2:
            raise ValueError(
                f"Native custom transform {step.name!r} requires a 2D payload."
            )
        n_out, p_out = output_layout.shape
        native_step = transform_step(
            step.name,
            "custom",
            n,
            p,
            function_address=step.func.address,
            backing=step.func,
            output_n=n_out,
            output_p=p_out,
        )
    else:
        native_step = transform_step(
            step.name,
            step.step_type or "",
            n,
            p,
            **_supplied(step.kwargs, "ddof", "offset", "order", "window"),
        )
    return native_step, (
        _source_binding(
            source_indices[0],
            steps,
            plan,
            step.source_args[0],
            target_offset=0,
            target_row_stride=p,
        ),
    )
