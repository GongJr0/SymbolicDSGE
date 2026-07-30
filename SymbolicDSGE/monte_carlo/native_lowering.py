"""Lower resolved Monte Carlo pipelines into native runner inputs.

Python remains responsible for resolving fields, selections, and lifetimes.
This module turns that resolved information into one arena allocation, native
step contexts, and the source transfers the runner applies inside its
replication loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray

from SymbolicDSGE.core.solver_backend import PerturbationSolution

from .._ckernels.monte_carlo._arenas import ArenaAllocation, allocate_arenas
from .._ckernels.monte_carlo._runner import (
    NativeStep,
    simulate1_step,
    simulate2_step,
    jarque_bera_step,
    ols_step,
    payload_step,
    raw_model_data_step,
    transform_step,
)
from ..core.solved_model import SolvedModel
from .allocation import BufferPlan, FieldLayout, StepBufferPlan
from .mc_constructs import MCStep, OpType, SourceArgs
from .operations.utils import _clone_or_pass_shocks

if TYPE_CHECKING:
    from .core import MCPipeline


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
class LoweredMCRun:
    """Run-local native inputs produced from one resolved pipeline."""

    plan: BufferPlan
    allocation: ArenaAllocation
    steps: tuple[NativeStep, ...]
    input_bindings: tuple[tuple[FloatInputBinding, ...], ...]
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
    return LoweredMCRun(
        plan,
        allocation,
        tuple(steps),
        tuple(bindings),
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

        elif step.step_type == "simulation":
            return _lower_simulation_step(step, step_plan, reference, dgp, n_rep)
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

    if step.op_type is OpType.REGRESSION:
        if step.kwargs["kind"] != "ols":
            raise NotImplementedError(
                f"Native lowering is not implemented for {step.kwargs['kind']!r}."
            )
        n, x_columns = _selected_shape(
            source_indices[1], step.source_args[1], steps, plan
        )
        y_rows, y_columns = _selected_shape(
            source_indices[0], step.source_args[0], steps, plan
        )
        if y_rows != n or y_columns != 1:
            raise ValueError("Native OLS lowering requires a one-column response.")
        intercept = bool(step.kwargs["intercept"])
        p = x_columns + int(intercept)
        step_bindings: list[FloatInputBinding] = []
        if intercept:
            step_bindings.append(_fill_binding(n, 0, p, 1.0))
        step_bindings.append(
            _source_binding(
                source_indices[1],
                steps,
                plan,
                step.source_args[1],
                target_offset=int(intercept),
                target_row_stride=p,
            )
        )
        step_bindings.append(
            _source_binding(
                source_indices[0],
                steps,
                plan,
                step.source_args[0],
                target_offset=n * p,
                target_row_stride=1,
            )
        )
        return ols_step(step.name, n, p, intercept), tuple(step_bindings)

    if step.op_type is OpType.TEST:
        if step.step_type != "jarque_bera":
            raise NotImplementedError(
                f"Native lowering is not implemented for {step.step_type!r}."
            )
        n, p = _selected_shape(source_indices[0], step.source_args[0], steps, plan)
        if p != 1:
            raise ValueError("Native Jarque-Bera lowering requires one column.")
        return jarque_bera_step(step.name, n), (
            _source_binding(
                source_indices[0],
                steps,
                plan,
                step.source_args[0],
                target_offset=0,
                target_row_stride=1,
            ),
        )

    raise NotImplementedError(
        f"Native lowering is not implemented for {step.name!r} ({step.step_type!r})."
    )


def _lower_simulation_step(
    step: MCStep,
    step_plan: StepBufferPlan,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    n_rep: int,
) -> tuple[NativeStep, tuple[FloatInputBinding, ...]]:
    T = int(step.kwargs["T"])
    target = step.kwargs["target"]
    if target not in {"reference", "dgp"}:
        raise ValueError(f"Unsupported simulation target: {target!r}.")
    model = reference if target == "reference" else dgp
    if model is None:
        raise ValueError("Simulation step requires its target model.")

    comp = model.compiled
    n_var = len(comp.var_names)
    n_state = comp.n_state
    n_ctrl = n_var - n_state
    n_exog = comp.n_exog
    n_par = len(comp.calib_params)
    observable_names = (
        tuple(comp.observable_names) if step.kwargs["observables"] else ()
    )
    n_obs = len(observable_names)
    measurement_addr = (
        int(comp.construct_measurement_cfunc(observable_names).address)
        if observable_names
        else 0
    )
    params = _model_params(model)
    shocks, shocks_batched = _simulation_shocks(model, step, T, n_rep)
    order = model.policy.order

    if order == 1:
        _check_simulation_layout(step_plan, T, n_var, n_obs)
        native_step = simulate1_step(
            step.name,
            measurement_addr,
            T,
            n_var,
            n_exog,
            n_par,
            n_obs,
        )
        x0 = model._simulation_initial_state(step.kwargs["x0"])
        bindings = _simulation_order1_bindings(
            model,
            x0,
            shocks,
            shocks_batched,
            params,
        )
        return native_step, bindings

    if order == 2:
        _check_simulation_layout(step_plan, T, n_var, n_obs)
        native_step = simulate2_step(
            step.name,
            measurement_addr,
            T,
            n_state,
            n_ctrl,
            n_exog,
            n_par,
            n_obs,
        )
        steady_state = _f64(model.policy.steady_state)
        initial_state = model._simulation_initial_state(step.kwargs["x0"])
        x0_deviation = initial_state[:n_state] - steady_state[:n_state]
        bindings = _simulation_order2_bindings(
            model,
            steady_state,
            x0_deviation,
            shocks,
            shocks_batched,
            params,
        )
        return native_step, bindings

    raise ValueError(f"Unsupported native simulation order: {order}.")


def _simulation_order1_bindings(
    model: SolvedModel,
    x0: NDF,
    shocks: NDF,
    shocks_batched: bool,
    params: NDF,
) -> tuple[FloatInputBinding, ...]:
    n_var = model.A.shape[0]
    n_exog = model.compiled.n_exog
    T = shocks.shape[-2]
    bindings: list[FloatInputBinding] = []
    offset = 0
    for values in (_f64(model.A), _f64(model.B), _f64(x0)):
        if values.size:
            bindings.append(_static_binding(values, offset))
        offset += values.size
    if shocks.size:
        bindings.append(_static_binding(shocks, offset, batched=shocks_batched))
    offset += T * n_exog
    if params.size:
        bindings.append(_static_binding(params, offset))
    return tuple(bindings)


def _simulation_order2_bindings(
    model: SolvedModel,
    steady_state: NDF,
    x0_deviation: NDF,
    shocks: NDF,
    shocks_batched: bool,
    params: NDF,
) -> tuple[FloatInputBinding, ...]:
    policy = model.policy
    if not isinstance(policy, PerturbationSolution):
        raise ValueError("Native simulation order 2 requires a perturbation solution.")

    n_state = model.compiled.n_state
    n_exog = model.compiled.n_exog
    T = shocks.shape[-2]
    values_by_layout = (
        _f64(policy.p),
        _f64(policy.f),
        _f64(model.B[:n_state, :]),
        _f64(policy.hxx),
        _f64(policy.gxx),
        _f64(policy.hss),
        _f64(policy.gss),
        _f64(steady_state),
        _f64(x0_deviation),
    )
    bindings: list[FloatInputBinding] = []
    offset = 0
    for values in values_by_layout:
        if values.size:
            bindings.append(_static_binding(values, offset))
        offset += values.size
    if shocks.size:
        bindings.append(_static_binding(shocks, offset, batched=shocks_batched))
    offset += T * n_exog
    if params.size:
        bindings.append(_static_binding(params, offset))
    return tuple(bindings)


def _simulation_shocks(
    model: SolvedModel,
    step: MCStep,
    T: int,
    n_rep: int,
) -> tuple[NDF, bool]:
    shocks = step.kwargs["shocks"]
    shock_scale = float(step.kwargs["shock_scale"])
    if shocks is None:
        return (
            _array_f64(model._simulation_shock_matrix(T, shock_scale=shock_scale)),
            False,
        )

    values = np.empty((n_rep, T, model.compiled.n_exog), dtype=np.float64)
    for rep_idx in range(n_rep):
        per_rep_shocks = _clone_or_pass_shocks(
            shocks,
            T=T,
            rep_idx=rep_idx,
            seed_increment=step.kwargs["seed_increment"],
        )
        values[rep_idx] = model._simulation_shock_matrix(
            T,
            shocks=per_rep_shocks,
            shock_scale=shock_scale,
        )
    return values, True


def _model_params(model: SolvedModel) -> NDF:
    parameters = model.config.calibration.parameters
    return _f64(
        np.asarray([parameters[param] for param in model.compiled.calib_params])
    )


def _f64(values: NDF | NDC) -> NDF:
    return _array_f64(values).reshape(-1)


def _array_f64(values: NDF | NDC) -> NDF:
    return np.ascontiguousarray(np.real_if_close(values), dtype=np.float64)


def _check_simulation_layout(
    step_plan: StepBufferPlan,
    T: int,
    n_var: int,
    n_obs: int,
) -> None:
    fields = step_plan.out_fields
    states = fields["states"]
    if states.offset != 0 or states.shape != (T, n_var):
        raise ValueError("Native simulation states do not match their output layout.")
    if n_obs:
        observables = fields["observables"]
        if observables.offset != states.flat_count or observables.shape != (T, n_obs):
            raise ValueError(
                "Native simulation observables do not match their output layout."
            )


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
