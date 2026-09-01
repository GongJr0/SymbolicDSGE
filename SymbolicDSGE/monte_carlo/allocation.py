"""Native Monte Carlo output-layout planning.

The Python pipeline compiler resolves each built-in step's logical output fields
before execution.  Native kernels receive flat float and integer output lanes,
so this module also assigns the field offsets that map those lanes back to
named logical arrays.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Sequence, TypeAlias, cast

import numpy as np

from .._ckernels.monte_carlo import _arena, _offsets
from .._ckernels.monte_carlo._runner import (
    DEFAULT_BREUSCH_GODFREY_LAGS,
    DEFAULT_INTERCEPT,
    DEFAULT_LJUNG_BOX_LAGS,
    DEFAULT_MAX_ITER,
    DEFAULT_ORDER,
    DEFAULT_RETURN_SHOCKS,
    DEFAULT_WINDOW,
)
from ..core.solved_model import SolvedModel
from .defaults import (
    DEFAULT_FILTER_MODE,
    DEFAULT_REGRESSION_KIND,
    DEFAULT_SIMULATION_OBSERVABLES,
    DEFAULT_SIMULATION_TARGET,
    DEFAULT_WALD_KIND_NAME,
)
from .mc_constructs import MCStep, OpType, SourceArgs
from .shock_native import native_shock_scratch

Shape: TypeAlias = tuple[int, ...]

_FLOAT = np.dtype(np.float64)
_INT = np.dtype(np.int64)

# What a field's shape becomes when the native layout reserved no buffer for
# it.  A field that is merely empty keeps its own shape; this is for one the
# configuration dropped, and it carries no dimension the step never resolved.
_ABSENT: Shape = (0,)


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
                fields, offsets = _resolve_datagen_fields(step, reference, dgp)
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
                    step,
                    steps[0],
                    plans[steps[0].name],
                    reference,
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
    offsets: _offsets.ArenaOffset,
) -> tuple[ArenaSize, dict[str, FieldLayout]]:
    """Place each field on the buffer the native layout opened for it.

    Fields arrive one per buffer, in the order that layout describes them, and a
    field's dtype selects which lane it is counted against.  Both the offset and
    the width come from the layout, so the only thing stated here is which name
    belongs to which buffer.
    """
    lanes: dict[np.dtype[Any], list[tuple[int, int]]] = {
        _FLOAT: list(zip(offsets.foffset, offsets.fwidth)),
        _INT: list(zip(offsets.ioffset, offsets.iwidth)),
    }
    taken: dict[np.dtype[Any], int] = {_FLOAT: 0, _INT: 0}
    layouts: dict[str, FieldLayout] = {}
    for name, spec in fields.items():
        dtype = np.dtype(spec.dtype)
        lane = lanes.get(dtype)
        if lane is None:
            raise TypeError(
                f"Native output field {name!r} has unsupported dtype {dtype}."
            )
        if taken[dtype] == len(lane):
            raise ValueError(
                f"Output field {name!r} has no buffer in the native layout."
            )
        offset, width = lane[taken[dtype]]
        taken[dtype] += 1
        layouts[name] = FieldLayout(
            _shaped(name, spec.shape, width), width, dtype, offset
        )
    for dtype, lane in lanes.items():
        if taken[dtype] != len(lane):
            raise ValueError(
                f"The native layout describes {len(lane)} {dtype} buffers, but "
                f"{taken[dtype]} fields name them."
            )
    return ArenaSize(_lane_total(lanes[_FLOAT]), _lane_total(lanes[_INT])), layouts


def _lane_total(lane: Sequence[tuple[int, int]]) -> int:
    """One lane's element count, which its last buffer closes."""
    return lane[-1][0] + lane[-1][1] if lane else 0


def _shaped(name: str, shape: Shape, width: int) -> Shape:
    """A field's declared shape, or the shape of absence when it has no buffer.

    A field can be legitimately empty, as a rolling window wider than its source
    is, and that shape is its own to keep.  Absence is the other case: the
    native layout reserved nothing, and the field says so rather than reporting
    a width the step never resolved.
    """
    resolved = tuple(int(size) for size in shape)
    if any(size < 0 for size in resolved):
        raise ValueError(f"Output field {name!r} has a negative dimension.")
    if width or not _flat(resolved):
        return resolved
    return _ABSENT


def is_absent(layout: FieldLayout) -> bool:
    """Whether the native layout reserved no buffer for this field."""
    return layout.shape == _ABSENT


def _flat(shape: Shape) -> int:
    """Elements one logical shape occupies in its flat lane."""
    return int(np.prod(shape, dtype=np.intp)) if shape else 1


def _single_buffer(flat_count: int) -> _offsets.ArenaOffset:
    """The one layout a lane holding a single float buffer can have.

    Transform and payload outputs are written to a lane of their own, so there
    is no interior boundary for the two sides to agree on.
    """
    return _offsets.ArenaOffset((0,), (flat_count,), (), ())


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
            if step.step_type == "transform:custom":
                return ArenaSize(n * p)
            return _asize(
                _arena.transform_arena_size(
                    step.step_type or "", n, p, _transform_arena_param(step)
                )
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
    model = cast(
        SolvedModel,
        (
            reference
            if step.kwargs.get("target", DEFAULT_SIMULATION_TARGET) == "reference"
            else dgp
        ),
    )
    T = int(step.kwargs["T"])
    size = _asize(
        _arena.simulation_arena_size(
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
    scratch = native_shock_scratch(step.kwargs.get("shocks"), T)
    if scratch:
        size = ArenaSize(n_float=size.n_float + scratch, n_int=size.n_int)
    return size


def _resolve_filter_input_asize(
    step: MCStep,
    datagen_step: MCStep,
    datagen_plan: StepBufferPlan,
    reference: SolvedModel,
) -> ArenaSize:
    observables = datagen_plan.out_fields.get("observables")
    if observables is None or is_absent(observables):
        raise ValueError("Filter input planning requires datagen observables.")
    T, datagen_n_obs = observables.shape
    selected_observables = step.kwargs.get("observables")
    if selected_observables is not None:
        n_obs = len(selected_observables)
    elif datagen_step.step_type == "raw_model_data" and not datagen_step.kwargs.get(
        "observable_names"
    ):
        n_obs = reference.compiled.n_obs
    else:
        n_obs = datagen_n_obs
    return _asize(
        _arena.filter_arena_size(
            _filter_mode(step),
            reference.compiled.n_state,
            reference.compiled.n_ctrl,
            reference.compiled.n_exog,
            n_obs,
            T,
            reference.compiled.n_par,
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
        _arena.regression_arena_size(
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


def _transform_arena_param(step: MCStep) -> int:
    """The order or window a transform's arena is sized against, else nothing."""
    if step.step_type == "diff":
        return int(step.kwargs.get("order", DEFAULT_ORDER))
    if step.step_type in {"rolling_mean", "rolling_std", "rolling_var"}:
        return int(step.kwargs.get("window", DEFAULT_WINDOW))
    return 0


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
                _arena.diagnostic_arena_size(
                    f"wald_{step.kwargs.get('kind', DEFAULT_WALD_KIND_NAME)}",
                    n,
                    p,
                )
            )
        case "ljung_box":
            _require_single_column(step, p)
            return _asize(
                _arena.diagnostic_arena_size(
                    "ljung_box",
                    n,
                    lags=int(step.kwargs.get("lags", DEFAULT_LJUNG_BOX_LAGS)),
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
    layout = plans[producer.name].out_fields.get(selector.field)
    if layout is None or is_absent(layout):
        raise ValueError(
            f"Step {producer.name!r} does not produce source field {selector.field!r}."
        )
    shape = layout.shape
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
        case "transform:custom":
            shape = tuple(int(size) for size in kwargs["output_shape"])
            if len(shape) != 2 or any(size < 0 for size in shape):
                raise ValueError(
                    "Custom transform output_shape must contain two non-negative dimensions."
                )
            return shape
        case "passthrough":
            return n_rows, n_columns
        case "standardize" | "log":
            return n_rows, n_columns
        case "log_diff":
            return max(0, n_rows - 1), n_columns
        case "diff":
            return max(0, n_rows - int(kwargs.get("order", DEFAULT_ORDER))), n_columns
        case "rolling_mean" | "rolling_std" | "rolling_var":
            window = int(kwargs.get("window", DEFAULT_WINDOW))
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
) -> tuple[dict[str, _FieldSpec], _offsets.ArenaOffset]:
    match step.step_type:
        case "simulation":
            target = step.kwargs.get("target", DEFAULT_SIMULATION_TARGET)
            model = reference if target == "reference" else dgp
            if model is None:
                raise ValueError(
                    "Simulation output planning requires its target model."
                )
            T = int(step.kwargs["T"])
            n_obs = (
                model.compiled.n_obs
                if step.kwargs.get("observables", DEFAULT_SIMULATION_OBSERVABLES)
                else 0
            )
            fields: dict[str, _FieldSpec] = {
                "states": _field((T, model.compiled.n_var)),
                "shocks": _field((T, model.compiled.n_exog)),
                "observables": _field((T, n_obs) if n_obs else _ABSENT),
            }
            return fields, _offsets.simulation_output_offsets(
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
                    else _ABSENT
                )
                for field in ("states", "shocks", "observables")
            }
            return {
                field: _field(shape) for field, shape in shapes.items()
            }, _offsets.raw_model_data_output_offsets(
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
    datagen_step: MCStep,
    datagen_plan: StepBufferPlan,
    reference: SolvedModel,
) -> tuple[dict[str, _FieldSpec], _offsets.ArenaOffset]:
    comp = reference.compiled
    observables = datagen_plan.out_fields.get("observables")
    if observables is None or is_absent(observables):
        raise ValueError("Filter output planning requires datagen observables.")
    T, datagen_n_obs = observables.shape

    selected_observables = step.kwargs.get("observables")
    if selected_observables is not None:
        n_obs = len(selected_observables)
    elif datagen_step.step_type == "raw_model_data" and not datagen_step.kwargs.get(
        "observable_names"
    ):
        n_obs = comp.n_obs
    else:
        n_obs = datagen_n_obs
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
            return fields, _offsets.filter_output_offsets(
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
            }, _offsets.filter_output_offsets(
                mode, n_state, comp.n_ctrl, comp.n_exog, n_obs, T
            )
        case _:
            raise ValueError(f"Unrecognized filter mode {mode!r}.")


def _resolve_transform_fields(
    step: MCStep,
    source_idx: int,
    plans: BufferPlan,
    steps: Sequence[MCStep],
) -> tuple[dict[str, _FieldSpec], _offsets.ArenaOffset]:
    if len(step.source_args) != 1:
        raise ValueError(f"Transform step {step.name!r} must have one source argument.")
    input_shape = _selected_source_shape(plans, steps, source_idx, step.source_args[0])
    shape = _transform_output_shape(step.step_type or "", input_shape, step.kwargs)
    return {"payload": _field(shape)}, _single_buffer(_flat(shape))


def _resolve_payload_fields(
    step: MCStep,
) -> tuple[dict[str, _FieldSpec], _offsets.ArenaOffset]:
    shape = _payload_shape(step.kwargs["value"])
    return {"payload": _field(shape)}, _single_buffer(_flat(shape))


def _resolve_regression_fields(
    step: MCStep,
    source_indices: Sequence[int],
    plans: BufferPlan,
    steps: Sequence[MCStep],
) -> tuple[dict[str, _FieldSpec], _offsets.ArenaOffset]:
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
        "status": _field((), np.int64),
        "se": _field((p,)),
    }
    return fields, _offsets.regression_output_offsets(
        str(step.kwargs.get("kind", DEFAULT_REGRESSION_KIND)), p
    )


def _resolve_test_fields() -> tuple[dict[str, _FieldSpec], _offsets.ArenaOffset]:
    """Return native diagnostic outputs, excluding post-loop p-values."""
    return {
        "statistic": _field(()),
        "status": _field((), np.int64),
    }, _offsets.diagnostic_output_offsets()
