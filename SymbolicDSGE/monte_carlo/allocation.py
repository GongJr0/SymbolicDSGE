"""Monte Carlo shape resolution and pipeline-owned buffer allocation."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Sequence, TypeAlias

import numpy as np

from ..core.solved_model import SolvedModel
from .mc_constructs import MCStep, OpType, SourceArgs

Shape: TypeAlias = tuple[int, ...]


class BufferSpec(NamedTuple):
    """Shape and dtype declaration for one array buffer."""

    shape: Shape
    dtype: Any


BufferPlan: TypeAlias = dict[str, dict[str, BufferSpec]]
AllocatedBuffers: TypeAlias = dict[str, dict[str, np.ndarray]]


def resolve_output_specs(
    steps: Sequence[MCStep],
    source_indices: Sequence[Sequence[int]],
    reference: SolvedModel,
    dgp: SolvedModel | None,
) -> BufferPlan:
    """Resolve per-replication output specs without allocating buffers."""
    specs: BufferPlan = {}
    for step_index, step in enumerate(steps):
        indices = source_indices[step_index]
        match step.op_type:
            case OpType.DATAGEN:
                specs[step.name] = _resolve_datagen_specs(step, reference, dgp)
            case OpType.TRANSFORM:
                if step.step_type == "payload":
                    specs[step.name] = _resolve_payload_specs(step)
                else:
                    specs[step.name] = _resolve_transform_specs(
                        step,
                        indices[0],
                        specs,
                        steps,
                    )
            case OpType.FILTER:
                datagen_step = steps[0]
                specs[step.name] = _resolve_filter_specs(
                    step,
                    datagen_step,
                    specs[datagen_step.name],
                    reference,
                )
            case OpType.REGRESSION:
                specs[step.name] = _resolve_regression_specs(
                    step,
                    indices,
                    specs,
                    steps,
                )
            case OpType.TEST:
                specs[step.name] = _resolve_test_specs()
            case _:
                raise NotImplementedError(
                    f"Shape resolution is not implemented for step {step.name!r} "
                    f"({step.step_type!r})."
                )
    return specs


def allocate_buffers(plan: BufferPlan) -> AllocatedBuffers:
    """Allocate a nested, pipeline-owned buffer plan.

    Each spec defines ``shape`` and ``dtype``. Algorithms that require
    cleared workspace initialize it locally before use.
    """
    allocated: AllocatedBuffers = {}
    for step_name, step_plan in plan.items():
        step_buffers: dict[str, np.ndarray] = {}
        for buffer_name, spec in step_plan.items():
            shape = spec.shape
            if any(size < 0 for size in shape):
                raise ValueError(
                    f"Buffer {step_name}.{buffer_name} has a negative dimension."
                )
            dtype = np.dtype(spec.dtype)
            step_buffers[buffer_name] = np.empty(shape, dtype=dtype)
        allocated[step_name] = step_buffers
    return allocated


def _raw_data_shape(field: str, value: object) -> Shape:
    array = np.asarray(value)
    if array.ndim == 1:
        return (array.shape[0], 1)
    if array.ndim == 2:
        return tuple(int(size) for size in array.shape)
    if array.ndim == 3:
        return tuple(int(size) for size in array.shape[1:])
    raise ValueError(f"Raw data for {field!r} must be 1D, 2D, or 3D.")


def _payload_shape(value: object) -> Shape:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        return array.shape[0], 1
    if array.ndim == 2:
        return tuple(int(size) for size in array.shape)
    raise ValueError(f"Payload must be 1-D, or 2-D; got {array.ndim}-D.")


def _float_specs(shapes: Mapping[str, Shape]) -> dict[str, BufferSpec]:
    return {
        name: BufferSpec(shape=shape, dtype=np.float64)
        for name, shape in shapes.items()
    }


def _selected_source_shape(
    specs: BufferPlan,
    steps: Sequence[MCStep],
    source_idx: int,
    selector: SourceArgs,
) -> Shape:
    producer = steps[source_idx]
    try:
        n_rows, n_columns = specs[producer.name][selector.field].shape
    except KeyError as exc:
        raise ValueError(
            f"Step {producer.name!r} does not produce source field {selector.field!r}."
        ) from exc

    n_rows = max(0, n_rows - selector.row_start)
    columns = selector.column_selector
    if isinstance(columns, slice):
        n_columns = len(range(*columns.indices(n_columns)))
    else:
        n_columns = len(columns)
    return n_rows, n_columns


def _transform_output_shape(
    step_type: str,
    input_shape: Shape,
    kwargs: Mapping[str, Any],
) -> Shape:
    n_rows, n_columns = input_shape
    match step_type:
        case "standardize" | "log":
            return n_rows, n_columns
        case "log_diff":
            return max(0, n_rows - 1), n_columns
        case "diff":
            return max(0, n_rows - int(kwargs["order"])), n_columns
        case "rolling_mean" | "rolling_std" | "rolling_var":
            window = int(kwargs["window"])
            return max(0, n_rows - window + 1), n_columns
        case _:
            raise NotImplementedError(
                f"Shape resolution is not implemented for transform step type {step_type!r}."
            )


def _resolve_datagen_specs(
    step: MCStep,
    reference: SolvedModel,
    dgp: SolvedModel | None,
) -> dict[str, BufferSpec]:
    match step.step_type:
        case "simulation":
            model = reference if step.kwargs["target"] == "reference" else dgp
            if model is None:
                raise ValueError(
                    "Simulation shape resolution requires its target model."
                )
            T = int(step.kwargs["T"])
            shapes: dict[str, Shape] = {"states": (T, len(model.compiled.var_names))}
            if step.kwargs["observables"]:
                shapes["observables"] = (T, len(model.compiled.observable_names))
            return _float_specs(shapes)
        case "raw_model_data":
            return _float_specs(
                {
                    field: _raw_data_shape(field, value)
                    for field in ("states", "observables")
                    if (value := step.kwargs[field]) is not None
                }
            )
        case _:
            raise NotImplementedError(
                f"Shape resolution is not implemented for datagen step type "
                f"{step.step_type!r}."
            )


def _resolve_filter_specs(
    step: MCStep,
    datagen_step: MCStep,
    datagen_specs: dict[str, BufferSpec],
    reference: SolvedModel,
) -> dict[str, BufferSpec]:
    try:
        T, datagen_n_obs = datagen_specs["observables"].shape
    except KeyError as exc:
        raise ValueError(
            "Filter shape resolution requires datagen observables."
        ) from exc

    selected_observables = step.kwargs["observables"]
    if selected_observables is not None:
        n_obs = len(selected_observables)
    elif (
        datagen_step.step_type == "raw_model_data"
        and not datagen_step.kwargs["observable_names"]
    ):
        n_obs = len(reference.compiled.observable_names)
    else:
        n_obs = datagen_n_obs
    n_var = len(reference.compiled.var_names)

    common: dict[str, Shape] = dict(
        x_pred=(T, n_var),
        x_filt=(T, n_var),
        y_pred=(T, n_obs),
        y_filt=(T, n_obs),
        S=(T, n_obs, n_obs),
        innov=(T, n_obs),
        std_innov=(T, n_obs),
        loglik=(),
    )
    match step.kwargs["filter_mode"]:
        case "linear" | "extended":
            linear_shapes = common | dict(
                P_pred=(T, n_var, n_var),
                P_filt=(T, n_var, n_var),
            )
            if step.kwargs["return_shocks"]:
                linear_shapes["eps_hat"] = (T, reference.compiled.n_exog)
            return _float_specs(linear_shapes)
        case "unscented":
            n_state = reference.compiled.n_state
            n_z = 2 * n_state
            return _float_specs(
                common
                | dict(
                    P_pred=(T, n_z, n_z),
                    P_filt=(T, n_z, n_z),
                    x1_pred=(T, n_state),
                    x1_filt=(T, n_state),
                    x2_pred=(T, n_state),
                    x2_filt=(T, n_state),
                )
            )
        case _:
            raise ValueError(
                f"Unrecognized filter mode {step.kwargs['filter_mode']!r}."
            )


def _resolve_transform_specs(
    step: MCStep,
    source_idx: int,
    specs: BufferPlan,
    steps: Sequence[MCStep],
) -> dict[str, BufferSpec]:
    if len(step.source_args) != 1:
        raise ValueError(f"Transform step {step.name!r} must have one source argument.")
    input_shape = _selected_source_shape(
        specs,
        steps,
        source_idx,
        step.source_args[0],
    )
    return _float_specs(
        {
            "payload": _transform_output_shape(
                step.step_type or "", input_shape, step.kwargs
            )
        }
    )


def _resolve_payload_specs(step: MCStep) -> dict[str, BufferSpec]:
    return _float_specs({"payload": _payload_shape(step.kwargs["value"])})


def _resolve_regression_specs(
    step: MCStep,
    source_indices: Sequence[int],
    specs: BufferPlan,
    steps: Sequence[MCStep],
) -> dict[str, BufferSpec]:
    if len(step.source_args) != 2 or len(source_indices) != 2:
        raise ValueError(
            f"Regression step {step.name!r} must have response and design sources."
        )
    y_rows, y_columns = _selected_source_shape(
        specs, steps, source_indices[0], step.source_args[0]
    )
    X_rows, X_columns = _selected_source_shape(
        specs, steps, source_indices[1], step.source_args[1]
    )
    if y_columns != 1:
        raise ValueError(
            f"Regression step {step.name!r} response must resolve to one column."
        )
    if y_rows != X_rows:
        raise ValueError(
            f"Regression step {step.name!r} response and design must have the "
            "same number of rows."
        )
    p = X_columns + int(step.kwargs["intercept"])
    if p == 0:
        raise ValueError(
            f"Regression step {step.name!r} requires a regressor or an intercept."
        )
    output_specs: dict[str, BufferSpec] = {
        "coef": BufferSpec((p,), np.float64),
        "ssr": BufferSpec((), np.float64),
        "sst": BufferSpec((), np.float64),
        "status": BufferSpec((), np.int64),
    }
    if step.kwargs["kind"] == "ols":
        output_specs["se"] = BufferSpec((p,), np.float64)
    return output_specs


def _resolve_test_specs() -> dict[str, BufferSpec]:
    """Return the scalar per-replication channels common to all tests.

    The allocation phase adds the replication axis, yielding one trace for each
    channel. Test-specific scratch buffers are planned separately from their
    resolved input shapes.
    """
    return {
        "statistic": BufferSpec((), np.float64),
        "pval": BufferSpec((), np.float64),
        "status": BufferSpec((), np.int64),
    }
