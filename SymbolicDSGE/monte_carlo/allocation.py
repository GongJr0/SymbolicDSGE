"""Native Monte Carlo output-layout planning.

The Python pipeline compiler resolves each built-in step's logical output fields
before execution.  Native kernels receive flat float and integer output lanes,
so this module also assigns the field offsets that map those lanes back to
named logical arrays.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Sequence, TypeAlias

import numpy as np

from .._ckernels.monte_carlo import _arena
from ..core.solved_model import SolvedModel
from .mc_constructs import MCStep, OpType, SourceArgs

Shape: TypeAlias = tuple[int, ...]


class ArenaSize(NamedTuple):
    """Element counts for the native float64 and int64 arena lanes."""

    n_float: int = 0
    n_int: int = 0


class FieldLayout(NamedTuple):
    """One named output field's logical shape and flat native-lane location."""

    shape: Shape
    flat_count: int
    dtype: np.dtype[Any]
    offset: int


class StepBufferPlan(NamedTuple):
    """Flat native arena plan for one compiled per-replication step.

    ``n_retain`` remains unresolved when it is ``-1``.  The run allocator
    replaces that sentinel with ``n_rep`` when it creates retained arenas.
    ``input_size`` includes the native kernel's staged inputs and workspace.
    Source bindings and static backing storage are planned in the next lowering
    phase; the existing Python ``SourceArgs`` flow is not altered here.
    """

    name: str
    input_size: ArenaSize
    output_size: ArenaSize
    out_fields: Mapping[str, FieldLayout]
    n_retain: int


class _FieldSpec(NamedTuple):
    """Logical field declaration used while compiling a flat layout."""

    shape: Shape
    dtype: Any


BufferPlan: TypeAlias = dict[str, StepBufferPlan]


def resolve_output_specs(
    steps: Sequence[MCStep],
    source_indices: Sequence[Sequence[int]],
    reference: SolvedModel,
    dgp: SolvedModel | None,
) -> BufferPlan:
    """Compile native output layouts for the pipeline's built-in steps.

    Shapes remain logical metadata.  Only ``output_size`` and the offsets in
    ``out_fields`` describe physical storage, which is allocated later as
    flat per-step arenas.
    """
    plans: BufferPlan = {}
    for step_index, step in enumerate(steps):
        indices = source_indices[step_index]
        match step.op_type:
            case OpType.DATAGEN:
                fields = _resolve_datagen_fields(step, reference, dgp)
            case OpType.TRANSFORM:
                if step.step_type == "payload":
                    fields = _resolve_payload_fields(step)
                else:
                    fields = _resolve_transform_fields(
                        step,
                        indices[0],
                        plans,
                        steps,
                    )
            case OpType.FILTER:
                fields = _resolve_filter_fields(
                    step,
                    steps[0],
                    plans[steps[0].name],
                    reference,
                )
            case OpType.REGRESSION:
                fields = _resolve_regression_fields(step, indices, plans, steps)
            case OpType.TEST:
                fields = _resolve_test_fields()
            case _:
                raise NotImplementedError(
                    f"Output-layout resolution is not implemented for step "
                    f"{step.name!r} ({step.step_type!r})."
                )

        output_size, out_fields = _compile_field_layout(fields)
        input_size = _resolve_input_asize(
            step,
            indices,
            plans,
            steps,
            reference,
            dgp,
        )
        plans[step.name] = StepBufferPlan(
            name=step.name,
            input_size=input_size,
            output_size=output_size,
            out_fields=out_fields,
            n_retain=step.n_retain,
        )
    return plans


def _compile_field_layout(
    fields: Mapping[str, _FieldSpec],
) -> tuple[ArenaSize, dict[str, FieldLayout]]:
    """Assign dtype-local offsets in the native float and integer lanes."""
    n_float = 0
    n_int = 0
    layouts: dict[str, FieldLayout] = {}
    for name, spec in fields.items():
        shape = tuple(int(size) for size in spec.shape)
        if any(size < 0 for size in shape):
            raise ValueError(f"Output field {name!r} has a negative dimension.")
        flat_count = int(np.prod(shape, dtype=np.intp)) if shape else 1
        dtype = np.dtype(spec.dtype)
        if dtype == np.dtype(np.float64):
            offset = n_float
            n_float += flat_count
        elif dtype == np.dtype(np.int64):
            offset = n_int
            n_int += flat_count
        else:
            raise TypeError(
                f"Native output field {name!r} has unsupported dtype {dtype}."
            )
        layouts[name] = FieldLayout(shape, flat_count, dtype, offset)
    return ArenaSize(n_float, n_int), layouts


def _field(shape: Shape, dtype: Any = np.float64) -> _FieldSpec:
    return _FieldSpec(shape, dtype)


def _asize(values: tuple[int, int]) -> ArenaSize:
    return ArenaSize(*values)


def _resolve_input_asize(
    step: MCStep,
    source_indices: Sequence[int],
    plans: BufferPlan,
    steps: Sequence[MCStep],
    reference: SolvedModel,
    dgp: SolvedModel | None,
) -> ArenaSize:
    """Resolve one native step's packed input and workspace requirement."""
    match step.op_type:
        case OpType.DATAGEN:
            return _resolve_datagen_input_asize(step, reference, dgp)
        case OpType.TRANSFORM:
            if step.step_type == "payload":
                return ArenaSize()
            n, p = _selected_source_shape(
                plans, steps, source_indices[0], step.source_args[0]
            )
            param = int(step.kwargs.get("order", step.kwargs.get("window", 0)))
            return _asize(
                _arena.transform_arena_size(step.step_type or "", n, p, param)
            )
        case OpType.FILTER:
            return _resolve_filter_input_asize(
                step, steps[0], plans[steps[0].name], reference
            )
        case OpType.REGRESSION:
            return _resolve_regression_input_asize(step, source_indices, plans, steps)
        case OpType.TEST:
            return _resolve_test_input_asize(step, source_indices, plans, steps)
        case _:
            raise NotImplementedError(
                f"Input arena resolution is not implemented for step {step.name!r} "
                f"({step.step_type!r})."
            )


def _resolve_datagen_input_asize(
    step: MCStep,
    reference: SolvedModel,
    dgp: SolvedModel | None,
) -> ArenaSize:
    if step.step_type == "raw_model_data":
        return ArenaSize()
    if step.step_type != "simulation":
        raise NotImplementedError(
            f"Input arena resolution is not implemented for datagen step type "
            f"{step.step_type!r}."
        )
    model = reference if step.kwargs["target"] == "reference" else dgp
    if model is None:
        raise ValueError("Simulation input planning requires its target model.")
    return _asize(
        _arena.simulation_arena_size(
            model.policy.order,
            model.compiled.n_state,
            len(model.compiled.var_names),
            model.compiled.n_exog,
            int(step.kwargs["T"]),
            len(model.compiled.calib_params),
        )
    )


def _resolve_filter_input_asize(
    step: MCStep,
    datagen_step: MCStep,
    datagen_plan: StepBufferPlan,
    reference: SolvedModel,
) -> ArenaSize:
    try:
        T, datagen_n_obs = datagen_plan.out_fields["observables"].shape
    except KeyError as exc:
        raise ValueError("Filter input planning requires datagen observables.") from exc
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
    n_state = reference.compiled.n_state
    n_ctrl = len(reference.compiled.var_names) - n_state
    return _asize(
        _arena.filter_arena_size(
            step.kwargs["filter_mode"],
            n_state,
            n_ctrl,
            reference.compiled.n_exog,
            n_obs,
            T,
            len(reference.compiled.calib_params),
        )
    )


def _resolve_regression_input_asize(
    step: MCStep,
    source_indices: Sequence[int],
    plans: BufferPlan,
    steps: Sequence[MCStep],
) -> ArenaSize:
    y_rows, y_columns = _selected_source_shape(
        plans, steps, source_indices[0], step.source_args[0]
    )
    X_rows, X_columns = _selected_source_shape(
        plans, steps, source_indices[1], step.source_args[1]
    )
    if y_columns != 1 or y_rows != X_rows:
        raise ValueError(
            f"Regression step {step.name!r} must resolve one response column and "
            "matching response and design row counts."
        )
    intercept = bool(step.kwargs["intercept"])
    p = X_columns + int(intercept)
    return _asize(
        _arena.regression_arena_size(
            step.kwargs["kind"],
            y_rows,
            p,
            intercept,
            int(step.kwargs.get("num", 0)),
            int(step.kwargs.get("max_iter", 1000)),
        )
    )


def _resolve_test_input_asize(
    step: MCStep,
    source_indices: Sequence[int],
    plans: BufferPlan,
    steps: Sequence[MCStep],
) -> ArenaSize:
    first_shape = _selected_source_shape(
        plans, steps, source_indices[0], step.source_args[0]
    )
    n, p = first_shape
    match step.step_type:
        case "wald":
            return _asize(
                _arena.diagnostic_arena_size(f"wald_{step.kwargs['kind']}", n, p)
            )
        case "ljung_box":
            _require_single_column(step, p)
            return _asize(
                _arena.diagnostic_arena_size(
                    "ljung_box", n, lags=int(step.kwargs["lags"])
                )
            )
        case "jarque_bera":
            _require_single_column(step, p)
            return _asize(_arena.diagnostic_arena_size("jarque_bera", n))
        case "breusch_pagan" | "breusch_godfrey" | "chow" | "cusum" | "cusumsq":
            if len(source_indices) != 2:
                raise ValueError(
                    f"Test step {step.name!r} requires two source arguments."
                )
            second_n, second_p = _selected_source_shape(
                plans, steps, source_indices[1], step.source_args[1]
            )
            _require_single_column(step, p)
            if n != second_n:
                raise ValueError(
                    f"Test step {step.name!r} source arguments must have matching row counts."
                )
            return _asize(
                _arena.diagnostic_arena_size(
                    step.step_type,
                    n,
                    second_p,
                    int(step.kwargs.get("lags", 0)),
                )
            )
        case _:
            raise NotImplementedError(
                f"Input arena resolution is not implemented for test step type "
                f"{step.step_type!r}."
            )


def _require_single_column(step: MCStep, n_columns: int) -> None:
    if n_columns != 1:
        raise ValueError(f"Test step {step.name!r} requires a single-column source.")


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
    if array.ndim == 3:
        return tuple(int(size) for size in array.shape[1:])
    raise ValueError(f"Payload must be 1D, 2D, or 3D, got {array.ndim}D.")


def _selected_source_shape(
    plans: BufferPlan,
    steps: Sequence[MCStep],
    source_idx: int,
    selector: SourceArgs,
) -> Shape:
    producer = steps[source_idx]
    try:
        shape = plans[producer.name].out_fields[selector.field].shape
    except KeyError as exc:
        raise ValueError(
            f"Step {producer.name!r} does not produce source field {selector.field!r}."
        ) from exc
    if len(shape) != 2:
        raise ValueError(
            f"Source field {producer.name!r}.{selector.field} must be 2D, "
            f"got shape {shape}."
        )

    n_rows, n_columns = shape
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
                f"Output-layout resolution is not implemented for transform "
                f"step type {step_type!r}."
            )


def _resolve_datagen_fields(
    step: MCStep,
    reference: SolvedModel,
    dgp: SolvedModel | None,
) -> dict[str, _FieldSpec]:
    match step.step_type:
        case "simulation":
            model = reference if step.kwargs["target"] == "reference" else dgp
            if model is None:
                raise ValueError(
                    "Simulation output planning requires its target model."
                )
            T = int(step.kwargs["T"])
            fields: dict[str, _FieldSpec] = {
                "states": _field((T, len(model.compiled.var_names)))
            }
            if step.kwargs["observables"]:
                fields["observables"] = _field(
                    (T, len(model.compiled.observable_names))
                )
            return fields
        case "raw_model_data":
            return {
                field: _field(_raw_data_shape(field, value))
                for field in ("states", "observables")
                if (value := step.kwargs[field]) is not None
            }
        case _:
            raise NotImplementedError(
                f"Output-layout resolution is not implemented for datagen "
                f"step type {step.step_type!r}."
            )


def _resolve_filter_fields(
    step: MCStep,
    datagen_step: MCStep,
    datagen_plan: StepBufferPlan,
    reference: SolvedModel,
) -> dict[str, _FieldSpec]:
    try:
        T, datagen_n_obs = datagen_plan.out_fields["observables"].shape
    except KeyError as exc:
        raise ValueError(
            "Filter output planning requires datagen observables."
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

    match step.kwargs["filter_mode"]:
        case "linear" | "extended":
            fields = {
                "x_pred": _field((T, n_var)),
                "x_filt": _field((T, n_var)),
                "P_pred": _field((T, n_var, n_var)),
                "P_filt": _field((T, n_var, n_var)),
                "y_pred": _field((T, n_obs)),
                "y_filt": _field((T, n_obs)),
                "innov": _field((T, n_obs)),
                "std_innov": _field((T, n_obs)),
                "S": _field((T, n_obs, n_obs)),
            }
            if step.kwargs["return_shocks"]:
                fields["eps_hat"] = _field((T, reference.compiled.n_exog))
            fields["loglik"] = _field(())
            return fields
        case "unscented":
            n_state = reference.compiled.n_state
            n_z = 2 * n_state
            return {
                "x_pred": _field((T, n_var)),
                "x_filt": _field((T, n_var)),
                "P_pred": _field((T, n_z, n_z)),
                "P_filt": _field((T, n_z, n_z)),
                "y_pred": _field((T, n_obs)),
                "y_filt": _field((T, n_obs)),
                "innov": _field((T, n_obs)),
                "std_innov": _field((T, n_obs)),
                "S": _field((T, n_obs, n_obs)),
                "loglik": _field(()),
                "x1_pred": _field((T, n_state)),
                "x2_pred": _field((T, n_state)),
                "x1_filt": _field((T, n_state)),
                "x2_filt": _field((T, n_state)),
            }
        case _:
            raise ValueError(
                f"Unrecognized filter mode {step.kwargs['filter_mode']!r}."
            )


def _resolve_transform_fields(
    step: MCStep,
    source_idx: int,
    plans: BufferPlan,
    steps: Sequence[MCStep],
) -> dict[str, _FieldSpec]:
    if len(step.source_args) != 1:
        raise ValueError(f"Transform step {step.name!r} must have one source argument.")
    input_shape = _selected_source_shape(plans, steps, source_idx, step.source_args[0])
    return {
        "payload": _field(
            _transform_output_shape(step.step_type or "", input_shape, step.kwargs)
        )
    }


def _resolve_payload_fields(step: MCStep) -> dict[str, _FieldSpec]:
    return {"payload": _field(_payload_shape(step.kwargs["value"]))}


def _resolve_regression_fields(
    step: MCStep,
    source_indices: Sequence[int],
    plans: BufferPlan,
    steps: Sequence[MCStep],
) -> dict[str, _FieldSpec]:
    if len(step.source_args) != 2 or len(source_indices) != 2:
        raise ValueError(
            f"Regression step {step.name!r} must have response and design sources."
        )
    y_rows, y_columns = _selected_source_shape(
        plans, steps, source_indices[0], step.source_args[0]
    )
    X_rows, X_columns = _selected_source_shape(
        plans, steps, source_indices[1], step.source_args[1]
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
    fields = {
        "coef": _field((p,)),
        "ssr": _field(()),
        "sst": _field(()),
        "status": _field((), np.int64),
    }
    if step.kwargs["kind"] == "ols":
        fields["se"] = _field((p,))
    return fields


def _resolve_test_fields() -> dict[str, _FieldSpec]:
    """Return native diagnostic outputs, excluding post-loop p-values."""
    return {
        "statistic": _field(()),
        "status": _field((), np.int64),
    }
