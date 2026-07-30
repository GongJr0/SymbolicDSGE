"""Native Kalman-filter step lowering."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from SymbolicDSGE.core.solver_backend import PerturbationSolution
from SymbolicDSGE.kalman.interface import KalmanInterface

from SymbolicDSGE._ckernels.monte_carlo._runner import (
    NativeStep,
    filter_extended_step,
    filter_linear_step,
    filter_unscented_step,
)
from SymbolicDSGE.core.solved_model import SolvedModel
from ..allocation import BufferPlan, FieldLayout
from ..mc_constructs import MCStep
from .core import (
    NDF,
    NDI,
    FloatInputBinding,
    _f64,
    _model_params,
    _static_binding,
)


def lower_filter_step(
    step: MCStep,
    datagen_step: MCStep,
    plan: BufferPlan,
    reference: SolvedModel,
    dgp: SolvedModel | None,
) -> tuple[NativeStep, tuple[FloatInputBinding, ...]]:
    """Compile a resolved filter configuration and its staged observations."""
    source_names, requested_names = _filter_observable_names(
        datagen_step, step, reference, dgp
    )
    source_layout = plan[datagen_step.name].out_fields["observables"]
    T, source_n_obs = source_layout.shape
    mode = step.kwargs["filter_mode"]
    placeholder_y = np.zeros(
        (T, len(requested_names or source_names)), dtype=np.float64
    )
    canonical_names = _canonical_observables(reference, requested_names)
    measurement_addr = 0
    jacobian_addr = 0
    if mode in {"extended", "unscented"}:
        measurement_addr = int(
            reference.compiled.construct_measurement_cfunc(canonical_names).address
        )
        if mode == "extended":
            jacobian_addr = int(
                reference.compiled.construct_observable_jacobian_cfunc(
                    canonical_names
                ).address
            )
    interface = KalmanInterface(
        model=reference,
        observables=list(requested_names) if requested_names is not None else None,
        y=placeholder_y,
        filter_mode=mode,
        meas_addr=measurement_addr or None,
        jac_addr=jacobian_addr or None,
        calib_params=_model_params(reference),
        R=step.kwargs["R"],
        P0=step.kwargs["P0"],
        jitter=step.kwargs["jitter"],
        symmetrize=step.kwargs["symmetrize"],
        return_shocks=step.kwargs["return_shocks"],
    )
    canonical_names = tuple(interface.observables)
    if len(canonical_names) != source_n_obs and requested_names is None:
        raise ValueError("Filter observations do not match the DATAGEN output.")
    source_columns = _filter_source_columns(source_names, canonical_names)
    n_var = len(reference.compiled.var_names)
    n_exog = reference.compiled.n_exog
    n_par = len(reference.compiled.calib_params)
    n_obs = len(canonical_names)
    before_y: tuple[NDF, ...]

    if mode == "linear":
        C, d = interface._get_C_d()
        x0 = _filter_x0(step.kwargs["x0"], n_var)
        before_y = (
            _f64(interface.A),
            _f64(interface.B),
            _f64(C),
            _f64(d),
            _f64(interface.Q),
            _f64(interface.R),
        )
        binding = _filter_y_binding(
            source_layout, T, source_columns, sum(v.size for v in before_y), n_obs
        )
        return (
            filter_linear_step(
                step.name,
                T,
                n_var,
                n_obs,
                n_exog,
                interface.symmetrize,
                float(interface.jitter),
                interface.return_shocks,
            ),
            _filter_bindings(before_y, binding, (x0, interface.P0)),
        )
    if mode == "extended":
        x0 = _filter_x0(step.kwargs["x0"], n_var)
        params = _model_params(reference)
        before_y = (
            _f64(interface.A),
            _f64(interface.B),
            params,
            _f64(interface.Q),
            _f64(interface.R),
        )
        binding = _filter_y_binding(
            source_layout, T, source_columns, sum(v.size for v in before_y), n_obs
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
                interface.symmetrize,
                float(interface.jitter),
                interface.return_shocks,
            ),
            _filter_bindings(before_y, binding, (x0, interface.P0)),
        )
    if mode == "unscented":
        if interface.return_shocks:
            raise ValueError("Unscented filtering does not support return_shocks.")
        policy = reference.policy
        if not isinstance(policy, PerturbationSolution):
            raise ValueError(
                "Native unscented filtering requires a perturbation solution."
            )
        n_state = reference.compiled.n_state
        n_ctrl = n_var - n_state
        params = _model_params(reference)
        z0 = interface._build_unscented_z0(step.kwargs["x0"])
        before_y = (
            _f64(policy.p),
            _f64(policy.f),
            _f64(reference.B[:n_state, :]),
            _f64(policy.hxx),
            _f64(policy.gxx),
            _f64(policy.hss),
            _f64(policy.gss),
            _f64(policy.steady_state),
            params,
            _f64(interface.Q),
            _f64(interface.R),
        )
        binding = _filter_y_binding(
            source_layout, T, source_columns, sum(v.size for v in before_y), n_obs
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
                interface.ukf_alpha,
                interface.ukf_beta,
                interface.ukf_kappa,
                interface.symmetrize,
                float(interface.jitter),
            ),
            _filter_bindings(before_y, binding, (z0, interface.P0)),
        )
    raise ValueError(f"Unsupported native filter mode: {mode!r}.")


def _filter_observable_names(
    datagen_step: MCStep,
    filter_step: MCStep,
    reference: SolvedModel,
    dgp: SolvedModel | None,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    requested = filter_step.kwargs["observables"]
    if datagen_step.step_type == "raw_model_data":
        raw_names = tuple(datagen_step.kwargs["observable_names"])
        return (
            raw_names or tuple(reference.compiled.observable_names),
            tuple(requested) if requested is not None else (raw_names or None),
        )
    if datagen_step.step_type == "simulation":
        target = datagen_step.kwargs["target"]
        model = reference if target == "reference" else dgp
        if model is None:
            raise ValueError("Filter DATAGEN simulation requires its target model.")
        names = tuple(model.compiled.observable_names)
        return names, tuple(requested) if requested is not None else names
    raise NotImplementedError("Native filters require raw data or simulation DATAGEN.")


def _canonical_observables(
    reference: SolvedModel, requested: tuple[str, ...] | None
) -> tuple[str, ...]:
    all_names = tuple(reference.compiled.observable_names)
    selected = all_names if requested is None else requested
    index = {name: position for position, name in enumerate(all_names)}
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


def _filter_x0(value: ArrayLike | None, n_var: int) -> NDF:
    if value is None:
        return np.zeros(n_var, dtype=np.float64)
    x0 = _f64(np.asarray(value, dtype=np.float64))
    if x0.size != n_var:
        raise ValueError(f"Filter x0 must have length {n_var}.")
    return x0


def _filter_y_binding(
    source_layout: FieldLayout,
    T: int,
    columns: NDI,
    target_offset: int,
    n_obs: int,
) -> FloatInputBinding:
    source_T, source_n_obs = source_layout.shape
    if source_T != T or columns.size != n_obs:
        raise ValueError("Filter observations do not match their input layout.")
    return FloatInputBinding(
        source_step_idx=0,
        source_offset=source_layout.offset,
        source_row_stride=source_n_obs,
        row_start=0,
        n_rows=T,
        columns=columns,
        target_offset=target_offset,
        target_row_stride=n_obs,
    )


def _filter_bindings(
    before_y: tuple[NDF, ...],
    y_binding: FloatInputBinding,
    after_y: tuple[NDF, ...],
) -> tuple[FloatInputBinding, ...]:
    bindings: list[FloatInputBinding] = []
    offset = 0
    for values in before_y:
        flattened = _f64(values)
        if flattened.size:
            bindings.append(_static_binding(flattened, offset))
        offset += flattened.size
    if y_binding.target_offset != offset:
        raise ValueError("Native filter observation offset does not match its layout.")
    bindings.append(y_binding)
    offset += y_binding.n_rows * y_binding.target_row_stride
    for values in after_y:
        flattened = _f64(values)
        if flattened.size:
            bindings.append(_static_binding(flattened, offset))
        offset += flattened.size
    return tuple(bindings)
