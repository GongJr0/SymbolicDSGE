"""Native Kalman-filter step lowering."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from SymbolicDSGE.core.solver_backend import SecondOrderSolution
from SymbolicDSGE.kalman.resolvers import (
    _build_R,
    _build_P0,
    _build_unscented_z0,
)

from ..._ckernels.monte_carlo import _offsets
from ..._ckernels.monte_carlo._runner import (
    NativeStep,
    filter_extended_step,
    filter_linear_step,
    filter_unscented_step,
)
from ...core.solved_model import SolvedModel
from ...core.compiled_model import _shock_covariance
from ..allocation import BufferPlan, get_target_model
from ..defaults import DEFAULT_FILTER_MODE
from ..mc_constructs import MCStep
from .utils import (
    NDF,
    NDI,
    FloatInputBinding,
    _supplied,
    _flat_f64,
    _model_params,
    _static_binding,
    _source_binding,
)


def lower_filter_step(
    step: MCStep,
    source_indices: tuple[int, ...],
    steps: tuple[MCStep, ...],
    plan: BufferPlan,
    reference: SolvedModel | None,
    dgp: SolvedModel | None,
) -> tuple[NativeStep, tuple[FloatInputBinding, ...]]:
    """Compile a resolved filter configuration and its staged observations."""
    source_names, requested_names = _filter_observable_names(step, reference, dgp)
    if len(source_indices) != 1 or len(step.source_args) != 1:
        raise ValueError(f"Filter step {step.name!r} must have one source argument.")
    source_binding = _source_binding(
        source_indices[0],
        steps,
        plan,
        step.source_args[0],
        target_offset=0,
        target_row_stride=len(source_names),
    )
    T = source_binding.n_rows
    source_n_obs = source_binding.columns.size
    model = get_target_model(step, reference, dgp, "reference")
    mode = step.kwargs.get("filter_mode", DEFAULT_FILTER_MODE)

    canonical_names = _canonical_observables(model, requested_names)
    measurement_addr = 0
    jacobian_addr = 0
    if mode in {"extended", "unscented"}:
        measurement_addr = int(
            model.compiled.construct_measurement_cfunc(canonical_names).address
        )
        if mode == "extended":
            jacobian_addr = int(
                model.compiled.construct_measurement_jacobian_cfunc(
                    canonical_names
                ).address
            )

    if len(canonical_names) != source_n_obs:
        raise ValueError(
            "Selected source columns do not match the filter observable names."
        )
    source_columns = source_binding.columns[
        _filter_source_columns(source_names, canonical_names)
    ]
    n_state = model.compiled.n_state
    n_ctrl = model.compiled.n_ctrl
    n_var = model.compiled.n_var
    n_exog = model.compiled.n_exog
    n_par = model.compiled.n_par
    n_obs = len(canonical_names)
    before_y: tuple[NDF, ...]
    input_offsets = _offsets.filter_offsets(
        mode, n_state, n_ctrl, n_exog, n_obs, T, n_par
    ).foffset

    R = _build_R(model, step.kwargs.get("R"), canonical_names)
    Q = _shock_covariance(model.compiled)
    P0 = _build_P0(model, mode, step.kwargs.get("P0"))

    if mode == "linear":
        C, d = model._build_C_d_from_obs(canonical_names)
        before_y = (
            _flat_f64(model.policy.A),
            _flat_f64(model.policy.B),
            _flat_f64(C),
            _flat_f64(d),
            _flat_f64(Q),
            _flat_f64(R),
            _flat_f64(model.policy.steady_state),
        )
        binding = _filter_y_binding(
            source_binding, source_columns, input_offsets[len(before_y)], n_obs
        )
        return (
            filter_linear_step(
                step.name,
                T,
                n_var,
                n_obs,
                n_exog,
                **_supplied(
                    step.kwargs, "symmetrize", "joseph_cov", "jitter", "return_shocks"
                ),
            ),
            _filter_bindings(
                before_y,
                binding,
                (model._initial_state(step.kwargs.get("x0")), P0),
                input_offsets,
            ),
        )
    if mode == "extended":
        params = _model_params(model)
        before_y = (
            _flat_f64(model.policy.A),
            _flat_f64(model.policy.B),
            params,
            _flat_f64(Q),
            _flat_f64(R),
            _flat_f64(model.policy.steady_state),
        )
        binding = _filter_y_binding(
            source_binding, source_columns, input_offsets[len(before_y)], n_obs
        )
        return (
            filter_extended_step(
                step.name,
                measurement_addr,
                jacobian_addr,
                T,
                n_var,
                n_obs,
                n_exog,
                n_par,
                **_supplied(
                    step.kwargs, "symmetrize", "joseph_cov", "jitter", "return_shocks"
                ),
            ),
            _filter_bindings(
                before_y,
                binding,
                (model._initial_state(step.kwargs.get("x0")), P0),
                input_offsets,
            ),
        )

    else:  # mode == "unscented"
        if step.kwargs.get("return_shocks"):
            raise ValueError("Unscented filtering does not support return_shocks.")
        policy = model.policy
        if not isinstance(policy, SecondOrderSolution):
            raise ValueError(
                "Native unscented filtering requires a second order solution."
            )
        n_state = model.compiled.n_state
        n_ctrl = model.compiled.n_ctrl
        params = _model_params(model)
        z0 = _build_unscented_z0(model, step.kwargs.get("x0"))
        before_y = (
            _flat_f64(policy.hx),
            _flat_f64(policy.gx),
            _flat_f64(policy.B),
            _flat_f64(policy.hxx),
            _flat_f64(policy.gxx),
            _flat_f64(policy.hxu),
            _flat_f64(policy.gxu),
            _flat_f64(policy.huu),
            _flat_f64(policy.guu),
            _flat_f64(policy.hss),
            _flat_f64(policy.gss),
            _flat_f64(policy.steady_state),
            params,
            _flat_f64(Q),
            _flat_f64(R),
        )
        binding = _filter_y_binding(
            source_binding, source_columns, input_offsets[len(before_y)], n_obs
        )
        return (
            filter_unscented_step(
                step.name,
                measurement_addr,
                T,
                n_state,
                n_ctrl,
                n_exog,
                n_obs,
                n_par,
                **_supplied(
                    step.kwargs,
                    "alpha",
                    "beta",
                    "kappa",
                    "symmetrize",
                    "jitter",
                ),
            ),
            _filter_bindings(before_y, binding, (z0, P0), input_offsets),
        )


def _filter_observable_names(
    filter_step: MCStep,
    reference: SolvedModel | None,
    dgp: SolvedModel | None,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    """Name selected data columns explicitly or in the target model's order."""
    model = get_target_model(filter_step, reference, dgp, "reference")
    requested = filter_step.kwargs.get("observables")
    if requested is not None:
        names = tuple(requested)
        return names, names
    return tuple(model.compiled.observable_names), None


def _canonical_observables(
    reference: SolvedModel, requested: tuple[str, ...] | None
) -> tuple[str, ...]:
    all_names = tuple(reference.compiled.observable_names)
    selected = all_names if requested is None else requested
    index = {name: position for position, name in enumerate(all_names)}
    if not selected:
        raise ValueError("Filter observables must be non-empty.")
    if len(set(selected)) != len(selected):
        raise ValueError("Filter observables must be unique.")
    unknown = [name for name in selected if name not in index]
    if unknown:
        raise ValueError(f"Unknown reference observables: {unknown!r}.")
    return tuple(sorted(selected, key=index.__getitem__))


def _filter_source_columns(
    source_names: tuple[str, ...], canonical_names: tuple[str, ...]
) -> NDI:
    source_index = {name: position for position, name in enumerate(source_names)}
    missing = [name for name in canonical_names if name not in source_index]
    if missing:
        raise ValueError(f"DATAGEN output is missing filter observables: {missing!r}.")
    return np.asarray([source_index[name] for name in canonical_names], dtype=np.int64)


def _filter_y_binding(
    source_binding: FloatInputBinding,
    columns: NDI,
    target_offset: int,
    n_obs: int,
) -> FloatInputBinding:
    return replace(
        source_binding,
        columns=columns,
        target_offset=target_offset,
        target_row_stride=n_obs,
    )


def _filter_bindings(
    before_y: tuple[NDF, ...],
    y_binding: FloatInputBinding,
    after_y: tuple[NDF, ...],
    offsets: Sequence[int],
) -> tuple[FloatInputBinding, ...]:
    """Bind the staged constants around the observations they are packed with.

    The observation block is one buffer among them, so it takes its place in the
    sequence rather than being measured out from what precedes it.
    """
    bindings = [
        _static_binding(flattened, offset)
        for flattened, offset in zip(map(_flat_f64, before_y), offsets)
        if flattened.size
    ]
    bindings.append(y_binding)
    bindings.extend(
        _static_binding(flattened, offset)
        for flattened, offset in zip(
            map(_flat_f64, after_y), offsets[len(before_y) + 1 :]
        )
        if flattened.size
    )
    return tuple(bindings)
