"""Native Monte Carlo output-layout planning.

The Python pipeline compiler resolves each built-in step's logical output fields
before execution.  Native kernels receive flat float and integer output lanes,
so this module also assigns the field offsets that map those lanes back to
named logical arrays.
"""

from __future__ import annotations

from math import prod
from typing import Any, Mapping, NamedTuple, Sequence, TypeAlias

import numpy as np
from numpy import float64, int64

from .._ckernels.monte_carlo import _arena as a, _offsets as o
from .._ckernels.monte_carlo._runner import (
    DEFAULT_INTERCEPT,
    DEFAULT_BREUSCH_GODFREY_LAGS,
    DEFAULT_LJUNG_BOX_LAGS,
    DEFAULT_MAX_ITER,
    DEFAULT_ORDER,
    DEFAULT_RETURN_SHOCKS,
    DEFAULT_WINDOW,
)
from ..core.solved_model import SolvedModel
from ..core.shock.spec import _normalized_spec
from .defaults import (
    DEFAULT_FILTER_MODE,
    DEFAULT_REGRESSION_KIND,
    DEFAULT_SIMULATION_OBSERVABLES,
    DEFAULT_WALD_KIND_NAME,
)
from .mc_constructs import MCStep, OpType, SourceArgs
from ..core.shock.native import native_shock_scratch

Shape: TypeAlias = tuple[int, ...]


class ArenaSize(NamedTuple):
    """Element counts for the native float64 and int64 arena lanes."""

    n_float: int = 0
    n_int: int = 0


class FieldLayout(NamedTuple):
    """One named output field's logical shape and flat native-lane location."""

    shape: Shape
    flat_count: int
    dtype: type[float64] | type[int64]
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
    models: Mapping[str, SolvedModel] | None,
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
                fields, offsets = _resolve_datagen_fields(step, models)
            case OpType.TRANSFORM:
                if step.step_type == "payload":
                    fields, offsets = _resolve_payload_fields(step)
                else:
                    fields, offsets = _resolve_transform_fields(
                        step,
                        indices[0],
                        plans,
                        steps,
                    )
            case OpType.FILTER:
                fields, offsets = _resolve_filter_fields(
                    step, indices, plans, steps, models
                )
            case OpType.REGRESSION:
                fields, offsets = _resolve_regression_fields(
                    step, indices, plans, steps
                )
            case OpType.TEST:
                fields, offsets = _resolve_test_fields()
            case _:
                raise NotImplementedError(
                    f"Output-layout resolution is not implemented for step "
                    f"{step.name!r} ({step.step_type!r})."
                )

        output_size, out_fields = _compile_field_layout(fields, offsets)
        input_size = _resolve_input_asize(
            step,
            indices,
            plans,
            steps,
            models,
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
    offsets: o.ArenaOffset,
) -> tuple[ArenaSize, dict[str, FieldLayout]]:
    """Place each field on the buffer the native layout opened for it.

    Fields arrive one per buffer, in the order that layout describes them, and a
    field's dtype selects which lane it is counted against.  Both the offset and
    the width come from the layout, so the only thing stated here is which name
    belongs to which buffer.
    """
    lanes = {
        float64: list(zip(offsets.foffset, offsets.fwidth)),
        int64: list(zip(offsets.ioffset, offsets.iwidth)),
    }
    taken = {float64: 0, int64: 0}
    layouts: dict[str, FieldLayout] = {}
    for name, spec in fields.items():
        lane = lanes[spec.dtype]
        if taken[spec.dtype] >= len(lane):
            raise ValueError(
                f"Field {name!r} of type {spec.dtype.__name__} has no native buffer "
                "to occupy."
            )
        offset, width = lane[taken[spec.dtype]]
        layouts[name] = FieldLayout(
            shape=_shaped(name, spec.shape, width),
            flat_count=width,
            dtype=spec.dtype,
            offset=offset,
        )
        taken[spec.dtype] += 1

    for dtype, lane in lanes.items():
        if taken[dtype] != len(lane):
            raise ValueError(
                f"The native layout describes {len(lane)} {dtype.__name__} buffers, "
                f"but {taken[dtype]} fields name them."
            )
    return ArenaSize(_lane_total(lanes[float64]), _lane_total(lanes[int64])), layouts


def _lane_total(lane: Sequence[tuple[int, int]]) -> int:
    """One lane's element count, which its last buffer closes."""
    return lane[-1][0] + lane[-1][1] if lane else 0


def _shaped(name: str, shape: Shape, width: int) -> Shape:
    """A field's declared shape, holding no rows when the layout gave it no room.

    Dropping the leading axis covers a field the configuration left out and one
    that resolved to nothing on its own, which are the same fact about the same
    lane.  Every trailing dimension survives, so a shape never claims a width
    the step did not resolve.
    """
    resolved = tuple(int(size) for size in shape)
    if any(size < 0 for size in resolved):
        raise ValueError(f"Output field {name!r} has a negative dimension.")
    return resolved if width else (0, *resolved[1:])


def is_empty(layout: FieldLayout) -> bool:
    """Whether the native layout gave this field no elements."""
    return not layout.flat_count


def _flat(shape: Shape) -> int:
    """Elements one logical shape occupies in its flat lane."""
    return int(np.prod(shape, dtype=np.intp)) if shape else 1


def _field(shape: Shape, dtype: Any = float64) -> _FieldSpec:
    return _FieldSpec(shape, dtype)


def _with_int_flags(fields: Mapping[str, _FieldSpec]) -> dict[str, _FieldSpec]:
    """Close a field set with the int lane every native output carries.

    The lane is unconditional and the same for every family: whether any source
    failed, then this step's own status.  Naming it once here leaves each
    resolver stating only the fields that belong to it.
    """
    return {
        **fields,
        "has_failed_sources": _field((), int64),
        "status": _field((), int64),
    }


def _asize(values: tuple[int, int]) -> ArenaSize:
    return ArenaSize(*values)


def _resolve_input_asize(
    step: MCStep,
    source_indices: Sequence[int],
    plans: BufferPlan,
    steps: Sequence[MCStep],
    models: Mapping[str, SolvedModel] | None,
) -> ArenaSize:
    """Resolve one native step's packed input and workspace requirement."""
    match step.op_type:
        case OpType.DATAGEN:
            return _resolve_datagen_input_asize(step, models)
        case OpType.TRANSFORM:
            if step.step_type == "payload":
                return ArenaSize()
            n, p = _selected_source_shape(
                plans, steps, source_indices[0], step.source_args[0]
            )
            if step.step_type == "transform:custom":
                return ArenaSize(n * p)
            return _asize(
                a.transform_arena_size(
                    step.step_type or "",
                    n,
                    p,
                    int(step.kwargs.get("order", DEFAULT_ORDER)),
                    int(step.kwargs.get("window", DEFAULT_WINDOW)),
                )
            )
        case OpType.FILTER:
            return _resolve_filter_input_asize(
                step, source_indices, plans, steps, models
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
    models: Mapping[str, SolvedModel] | None,
) -> ArenaSize:
    if step.step_type == "raw_model_data":
        return ArenaSize()
    model = get_target_model(step, models)
    T = int(step.kwargs["T"])
    size = _asize(
        a.simulation_arena_size(
            model.policy.order,
            model.compiled.n_state,
            model.compiled.n_var,
            model.compiled.n_exog,
            T,
            model.compiled.n_par,
        )
    )
    # A step that draws its own shocks needs scratch past the simulation arena.
    # Lowering decides the same way, off the same spec, so the two agree.
    shocks = _normalized_spec(step.kwargs.get("shocks"))
    scratch = native_shock_scratch(shocks, T)
    if scratch:
        size = ArenaSize(n_float=size.n_float + scratch, n_int=size.n_int)
    return size


def _resolve_filter_shape(
    step: MCStep,
    source_indices: Sequence[int],
    plans: BufferPlan,
    steps: Sequence[MCStep],
    model: SolvedModel,
) -> tuple[int, int]:
    """Resolve selected observations and check their width against model names."""
    if len(step.source_args) != 1 or len(source_indices) != 1:
        raise ValueError(f"Filter step {step.name!r} must have one source argument.")
    T, n_obs = _selected_source_shape(
        plans, steps, source_indices[0], step.source_args[0]
    )
    names = step.kwargs.get("observables")
    expected = len(names) if names is not None else model.compiled.n_obs
    if n_obs != expected:
        raise ValueError(
            f"Filter step {step.name!r} selects {n_obs} observation columns "
            f"but requires {expected} for its observable names."
        )
    return T, n_obs


def _resolve_filter_input_asize(
    step: MCStep,
    source_indices: Sequence[int],
    plans: BufferPlan,
    steps: Sequence[MCStep],
    models: Mapping[str, SolvedModel] | None,
) -> ArenaSize:
    model = get_target_model(step, models)
    T, n_obs = _resolve_filter_shape(step, source_indices, plans, steps, model)
    comp = model.compiled
    return _asize(
        a.filter_arena_size(
            _filter_mode(step),
            comp.n_state,
            comp.n_ctrl,
            comp.n_exog,
            n_obs,
            T,
            comp.n_par,
        )
    )


def _resolve_regression_input_asize(
    step: MCStep,
    source_indices: Sequence[int],
    plans: BufferPlan,
    steps: Sequence[MCStep],
) -> ArenaSize:
    y_rows, _ = _selected_source_shape(
        plans, steps, source_indices[0], step.source_args[0]
    )
    _, X_columns = _selected_source_shape(
        plans, steps, source_indices[1], step.source_args[1]
    )
    intercept = bool(step.kwargs.get("intercept", DEFAULT_INTERCEPT))
    p = X_columns + int(intercept)
    return _asize(
        a.regression_arena_size(
            step.kwargs.get("kind", DEFAULT_REGRESSION_KIND),
            y_rows,
            p,
            intercept,
            int(step.kwargs.get("num", 0)),
            int(step.kwargs.get("max_iter", DEFAULT_MAX_ITER)),
        )
    )


def _filter_mode(step: MCStep) -> str:
    """A filter's mode, which selects the kernel rather than configuring one."""
    return str(step.kwargs.get("filter_mode", DEFAULT_FILTER_MODE))


def _breusch_godfrey_lags(step: MCStep) -> int:
    """Lagged residual count, which only Breusch-Godfrey sizes against."""
    if step.step_type != "breusch_godfrey":
        return 0
    return int(step.kwargs.get("lags", DEFAULT_BREUSCH_GODFREY_LAGS))


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
                a.diagnostic_arena_size(
                    f"wald_{step.kwargs.get('kind', DEFAULT_WALD_KIND_NAME)}",
                    n,
                    p,
                )
            )
        case "ljung_box":
            _require_single_column(step, p)
            return _asize(
                a.diagnostic_arena_size(
                    "ljung_box",
                    n,
                    lags=int(step.kwargs.get("lags", DEFAULT_LJUNG_BOX_LAGS)),
                )
            )
        case "jarque_bera":
            _require_single_column(step, p)
            return _asize(a.diagnostic_arena_size("jarque_bera", n))
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
                a.diagnostic_arena_size(
                    step.step_type,
                    n,
                    second_p,
                    _breusch_godfrey_lags(step),
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
    array = np.asarray(value, dtype=float64)
    if array.ndim < 2:
        return (prod(array.shape), 1)
    if array.ndim == 2:
        return tuple(int(size) for size in array.shape)
    if array.ndim == 3:
        return tuple(int(size) for size in array.shape[1:])
    raise ValueError(f"Payload must be 0D (scalar) to 3D, got {array.ndim}D.")


def source_matrix_shape(
    shape: Shape, producer_name: str, field: str
) -> tuple[int, int]:
    """Interpret a scalar, vector, or matrix source as rows and columns."""
    if len(shape) > 2:
        raise ValueError(
            f"Source field {producer_name!r}.{field} must be 2D or lower, "
            f"got shape {shape} with {len(shape)} dimensions."
        )
    return (shape[0], shape[1]) if len(shape) == 2 else (prod(shape), 1)


def _selected_source_shape(
    plans: BufferPlan,
    steps: Sequence[MCStep],
    source_idx: int,
    selector: SourceArgs,
) -> Shape:
    producer = steps[source_idx]
    layout = plans[producer.name].out_fields.get(selector.field)
    if layout is None or is_empty(layout):
        raise ValueError(
            f"Step {producer.name!r} does not produce source field {selector.field!r}."
        )
    n_rows, n_columns = source_matrix_shape(layout.shape, producer.name, selector.field)
    n_rows = n_rows - selector.burn_in
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
    """A transform's payload shape, over the rows the native rule gives it.

    ``passthrough`` and ``transform:custom`` name no layout kind: one keeps its
    source's shape, the other carries the shape its caller declared.
    """
    n_rows, n_columns = input_shape
    if step_type == "transform:custom":
        shape = tuple(int(size) for size in kwargs["output_shape"])
        if len(shape) != 2 or any(size < 0 for size in shape):
            raise ValueError(
                "Custom transform output_shape must contain two non-negative dimensions."
            )
        return shape
    if step_type == "passthrough":
        return n_rows, n_columns
    try:
        rows = o.transform_output_rows(
            step_type,
            n_rows,
            int(kwargs.get("order", DEFAULT_ORDER)),
            int(kwargs.get("window", DEFAULT_WINDOW)),
        )
    except ValueError as unsupported:
        raise NotImplementedError(
            f"Output-layout resolution is not implemented for transform "
            f"step type {step_type!r}."
        ) from unsupported
    return rows, n_columns


def get_target_model(
    step: MCStep,
    models: Mapping[str, SolvedModel] | None,
) -> SolvedModel:
    target: str | None = step.kwargs.get("target")

    if target is None:
        raise ValueError(
            f"Step {step.name!r} requires a target model, but no target was specified. "
            "Use the 'target' keyword to specify a model name to use in this step."
        )

    if models is None:
        raise ValueError(
            f"Step {step.name!r} requires its target model {target!r}, but no "
            "models were provided."
        )

    if target not in models:
        raise ValueError(
            f"Step {step.name!r} has unrecognized target model {target!r}. "
            f"Valid targets are {list(models.keys())}."
        )

    return models[target]


def _resolve_datagen_fields(
    step: MCStep,
    models: Mapping[str, SolvedModel] | None,
) -> tuple[dict[str, _FieldSpec], o.ArenaOffset]:
    match step.step_type:
        case "simulation":
            model = get_target_model(step, models)
            T = int(step.kwargs["T"])
            n_obs = (
                model.compiled.n_obs
                if step.kwargs.get("observables", DEFAULT_SIMULATION_OBSERVABLES)
                else 0
            )
            fields: dict[str, _FieldSpec] = {
                "states": _field((T, model.compiled.n_var)),
                "shocks": _field((T, model.compiled.n_exog)),
                "observables": _field((T, n_obs)),
            }
            return _with_int_flags(fields), o.simulation_output_offsets(
                model.policy.order,
                model.compiled.n_var,
                model.compiled.n_exog,
                T,
                n_obs,
            )
        case "raw_model_data":
            shapes = {
                field: (
                    _raw_data_shape(field, value)
                    if (value := step.kwargs.get(field)) is not None
                    else (0, 0)
                )
                for field in ("states", "shocks", "observables")
            }
            fields = {field: _field(shape) for field, shape in shapes.items()}
            return _with_int_flags(fields), o.raw_model_data_output_offsets(
                _flat(shapes["states"]),
                _flat(shapes["shocks"]),
                _flat(shapes["observables"]),
            )
        case _:
            raise NotImplementedError(
                f"Output-layout resolution is not implemented for datagen "
                f"step type {step.step_type!r}."
            )


def _resolve_filter_fields(
    step: MCStep,
    source_indices: Sequence[int],
    plans: BufferPlan,
    steps: Sequence[MCStep],
    models: Mapping[str, SolvedModel] | None,
) -> tuple[dict[str, _FieldSpec], o.ArenaOffset]:
    model = get_target_model(step, models)
    T, n_obs = _resolve_filter_shape(step, source_indices, plans, steps, model)
    comp = model.compiled
    n_var = comp.n_var
    mode = _filter_mode(step)
    match mode:
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
                "eps_hat": _field((T, comp.n_exog)),
                "loglik": _field(()),
            }
            return _with_int_flags(fields), o.filter_output_offsets(
                mode,
                comp.n_state,
                comp.n_ctrl,
                comp.n_exog,
                n_obs,
                T,
                bool(step.kwargs.get("return_shocks", DEFAULT_RETURN_SHOCKS)),
            )
        case "unscented":
            n_state = comp.n_state
            n_z = 2 * n_state
            fields = {
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
            return _with_int_flags(fields), o.filter_output_offsets(
                mode, n_state, comp.n_ctrl, comp.n_exog, n_obs, T
            )
        case _:
            raise ValueError(f"Unrecognized filter mode {mode!r}.")


def _resolve_transform_fields(
    step: MCStep,
    source_idx: int,
    plans: BufferPlan,
    steps: Sequence[MCStep],
) -> tuple[dict[str, _FieldSpec], o.ArenaOffset]:
    if len(step.source_args) != 1:
        raise ValueError(f"Transform step {step.name!r} must have one source argument.")
    input_shape = _selected_source_shape(plans, steps, source_idx, step.source_args[0])
    shape = _transform_output_shape(step.step_type or "", input_shape, step.kwargs)
    return _with_int_flags({"payload": _field(shape)}), o.transform_output_offsets(
        shape[0], shape[1]
    )


def _resolve_payload_fields(
    step: MCStep,
) -> tuple[dict[str, _FieldSpec], o.ArenaOffset]:
    shape = _payload_shape(step.kwargs["value"])
    return _with_int_flags({"payload": _field(shape)}), o.transform_output_offsets(
        shape[0], shape[1]
    )


def _resolve_regression_fields(
    step: MCStep,
    source_indices: Sequence[int],
    plans: BufferPlan,
    steps: Sequence[MCStep],
) -> tuple[dict[str, _FieldSpec], o.ArenaOffset]:
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
    p = X_columns + int(step.kwargs.get("intercept", DEFAULT_INTERCEPT))
    if p == 0:
        raise ValueError(
            f"Regression step {step.name!r} requires a regressor or an intercept."
        )
    fields = {
        "coef": _field((p,)),
        "ssr": _field(()),
        "sst": _field(()),
        "se": _field((p,)),
    }
    return _with_int_flags(fields), o.regression_output_offsets(
        str(step.kwargs.get("kind", DEFAULT_REGRESSION_KIND)), p
    )


def _resolve_test_fields() -> tuple[dict[str, _FieldSpec], o.ArenaOffset]:
    """Return native diagnostic outputs, excluding post-loop p-values."""
    return _with_int_flags({"statistic": _field(())}), o.diagnostic_output_offsets()
