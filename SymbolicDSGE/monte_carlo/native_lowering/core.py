"""Shared native Monte Carlo lowering orchestration and input bindings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray

from ..._ckernels.monte_carlo._arenas import ArenaAllocation, allocate_arenas
from ..._ckernels.monte_carlo._runner import (
    NativeStep,
    payload_step,
    raw_model_data_step,
    transform_step,
)
from ..._diag_tests.distributions import (
    DistributionParameter,
    PvalMethod,
    ReferenceDistribution,
)
from ...core.solved_model import SolvedModel
from ..allocation import BufferPlan, FieldLayout, StepBufferPlan
from ..mc_constructs import MCStep, OpType, SourceArgs

if TYPE_CHECKING:
    from ..core import MCPipeline


_FILL_COLUMNS = np.zeros(1, dtype=np.int64)
_STATIC_SOURCE_STEP = -2

NDF = NDArray[np.float64]
NDC = NDArray[np.complex128]
NDI = NDArray[np.int64]


@dataclass(frozen=True)
class FloatInputBinding:
    """One resolved transfer into a consumer's flat float input lane."""

    source_step_idx: int
    source_offset: int
    source_row_stride: int
    row_start: int
    n_rows: int
    columns: NDI
    target_offset: int
    target_row_stride: int
    fill_value: float = 0.0
    static_values: NDF | None = None
    static_rep_stride: int = 0
    static_batched: bool = False


@dataclass(frozen=True)
class TestResultSpec:
    """Resolved result metadata declared by a native diagnostic step."""

    name: str
    dist: ReferenceDistribution
    df: DistributionParameter | tuple[DistributionParameter, ...]
    pval_method: PvalMethod
    alpha: np.float64


@dataclass(frozen=True)
class RegressionResultSpec:
    """Resolved semantic metadata for one native regression result."""

    name: str
    kind: str
    variables: tuple[str, ...]
    n: int
    k: int


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
) -> LoweredMCRun:
    """Resolve and allocate one native run without entering the runner loop."""
    if n_rep <= 0:
        raise ValueError("n_rep must be positive.")

    plan = pipeline._resolve_output_specs(reference, dgp)
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
                alpha=np.float64(step.kwargs["alpha"]),
            )
        elif step.op_type is OpType.REGRESSION:
            from .regressions import regression_result_spec

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
    step_plan = plan[step.name]
    if step.op_type is OpType.DATAGEN:
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
            from .simulation import lower_simulation_step

            return lower_simulation_step(step, step_plan, reference, dgp, n_rep)

    if step.op_type is OpType.TRANSFORM:
        if step.step_type == "payload":
            return payload_step(step.name, step.kwargs["value"]), ()
        n, p = _selected_shape(source_indices[0], step.source_args[0], steps, plan)
        native_step = transform_step(
            step.name,
            step.step_type or "",
            n,
            p,
            ddof=int(step.kwargs.get("ddof", 0)),
            offset=float(step.kwargs.get("offset", 0.0)),
            order=int(step.kwargs.get("order", 1)),
            window=int(step.kwargs.get("window", 1)),
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

    if step.op_type is OpType.FILTER:
        from .filters import lower_filter_step

        return lower_filter_step(step, steps[0], plan, reference, dgp)
    if step.op_type is OpType.REGRESSION:
        from .regressions import lower_regression_step

        return lower_regression_step(step, source_indices, steps, plan)
    if step.op_type is OpType.TEST:
        from .diagnostics import lower_test_step

        return lower_test_step(step, source_indices, steps, plan)
    raise NotImplementedError(
        f"Native lowering is not implemented for {step.name!r} ({step.step_type!r})."
    )


def _model_params(model: SolvedModel) -> NDF:
    parameters = model.config.calibration.parameters
    return _f64(
        np.asarray([parameters[param] for param in model.compiled.calib_params])
    )


def _f64(values: NDF | NDC) -> NDF:
    return _array_f64(values).reshape(-1)


def _array_f64(values: NDF | NDC) -> NDF:
    return np.ascontiguousarray(np.real_if_close(values), dtype=np.float64)


def _selected_shape(
    source_idx: int,
    source: SourceArgs,
    steps: tuple[MCStep, ...],
    plan: BufferPlan,
) -> tuple[int, int]:
    source_step = steps[source_idx]
    layout = plan[source_step.name].out_fields[source.field]
    if len(layout.shape) != 2:
        raise ValueError(
            f"Native source {source_step.name!r}.{source.field!r} must be 2D."
        )
    n_rows, n_columns = layout.shape
    return n_rows - source.row_start, _selected_columns(source, n_columns).shape[0]


def _source_binding(
    source_step_idx: int,
    steps: tuple[MCStep, ...],
    plan: BufferPlan,
    source: SourceArgs,
    *,
    target_offset: int,
    target_row_stride: int,
) -> FloatInputBinding:
    producer = steps[source_step_idx]
    layout = plan[producer.name].out_fields[source.field]
    if layout.dtype != np.dtype(np.float64) or len(layout.shape) != 2:
        raise ValueError(
            f"Native source {producer.name!r}.{source.field!r} must be a 2D float field."
        )
    n_rows, n_columns = layout.shape
    columns = _selected_columns(source, n_columns)
    n_selected_rows = n_rows - source.row_start
    if n_selected_rows < 0:
        raise ValueError("Native source row selection starts past the input.")
    return FloatInputBinding(
        source_step_idx=source_step_idx,
        source_offset=layout.offset,
        source_row_stride=n_columns,
        row_start=source.row_start,
        n_rows=n_selected_rows,
        columns=columns,
        target_offset=target_offset,
        target_row_stride=target_row_stride,
    )


def _fill_binding(
    n_rows: int,
    target_offset: int,
    target_row_stride: int,
    value: float,
) -> FloatInputBinding:
    return FloatInputBinding(
        source_step_idx=-1,
        source_offset=0,
        source_row_stride=0,
        row_start=0,
        n_rows=n_rows,
        columns=_FILL_COLUMNS,
        target_offset=target_offset,
        target_row_stride=target_row_stride,
        fill_value=value,
    )


def _static_binding(
    values: NDF,
    target_offset: int,
    *,
    batched: bool = False,
) -> FloatInputBinding:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if batched:
        if array.ndim < 2 or array.shape[0] == 0:
            raise ValueError("Batched native input requires a non-empty leading axis.")
        row_size = int(np.prod(array.shape[1:], dtype=np.intp))
    else:
        row_size = array.size
    if row_size == 0:
        raise ValueError("Native static input bindings cannot be empty.")
    flattened = np.ascontiguousarray(array.reshape(-1), dtype=np.float64)
    return FloatInputBinding(
        source_step_idx=_STATIC_SOURCE_STEP,
        source_offset=0,
        source_row_stride=row_size,
        row_start=0,
        n_rows=1,
        columns=np.arange(row_size, dtype=np.int64),
        target_offset=target_offset,
        target_row_stride=row_size,
        static_values=flattened,
        static_rep_stride=row_size if batched else 0,
        static_batched=batched,
    )


def _selected_columns(source: SourceArgs, n_columns: int) -> NDI:
    columns = source.column_selector
    if isinstance(columns, slice):
        resolved = np.arange(n_columns, dtype=np.int64)[columns]
    else:
        resolved = np.asarray(columns, dtype=np.int64)
    return np.ascontiguousarray(resolved, dtype=np.int64)


def _check_raw_model_data_layout(
    step: MCStep,
    fields: Mapping[str, FieldLayout],
) -> None:
    expected_offset = 0
    for field in ("states", "observables"):
        value = step.kwargs[field]
        if value is None:
            continue
        layout = fields[field]
        if layout.offset != expected_offset:
            raise ValueError(
                "Raw model data fields must occupy contiguous float output."
            )
        expected_offset += layout.flat_count
