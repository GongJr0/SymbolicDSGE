"""Native first and second-order simulation step lowering."""

from __future__ import annotations

from typing import Sequence, Mapping, cast

from ..._ckernels.monte_carlo import _offsets
from ..._ckernels.monte_carlo._runner import (
    NativeStep,
    simulate1_step,
    simulate2_step,
)
from ...core.solved_model import SolvedModel
from ...core.solver_backend import SecondOrderSolution
from ...core.shock.plan import get_native_shock_plan
from ...core.shock.spec import (
    ShockSpec,
    resolve_shock_plan,
)
from ..defaults import (
    DEFAULT_SHOCK_SCALE,
    DEFAULT_SIMULATION_OBSERVABLES,
)
from ..allocation import get_target_model
from ..mc_constructs import MCStep
from .utils import (
    NDF,
    FloatInputBinding,
    _flat_f64,
    _model_params,
    _static_binding,
)


def lower_simulation_step(
    step: MCStep,
    models: Mapping[str, SolvedModel] | None,
    n_rep: int,
) -> tuple[NativeStep, tuple[FloatInputBinding, ...]]:
    """Compile one model simulation into the native simulation ABI."""
    T = int(step.kwargs["T"])
    model = get_target_model(step, models)

    comp = model.compiled
    n_var = comp.n_var
    n_state = comp.n_state
    n_ctrl = comp.n_ctrl
    n_exog = comp.n_exog
    n_par = comp.n_par
    observable_names = (
        tuple(comp.observable_names)
        if step.kwargs.get("observables", DEFAULT_SIMULATION_OBSERVABLES)
        else ()
    )
    n_obs = len(observable_names)
    measurement_addr = (
        int(comp.construct_measurement_cfunc(observable_names).address)
        if observable_names
        else 0
    )
    params = _model_params(model)
    shock_scale = float(step.kwargs.get("shock_scale", DEFAULT_SHOCK_SCALE))
    plan = resolve_shock_plan(
        model.compiled,
        cast(ShockSpec, step.kwargs.get("shocks")),
        T,
    )
    native_plan = get_native_shock_plan(plan, T, shock_scale)
    order = model.policy.order

    if order == 1:
        native_step = simulate1_step(
            step.name, measurement_addr, T, n_var, n_exog, n_par, n_obs, native_plan
        )
        steady_state = _flat_f64(model.policy.steady_state)
        x0 = model._initial_state(step.kwargs.get("x0"))
        return native_step, _order1_bindings(model, steady_state, x0, params, T)
    if order == 2:
        native_step = simulate2_step(
            step.name,
            measurement_addr,
            T,
            n_state,
            n_ctrl,
            n_exog,
            n_par,
            n_obs,
            native_plan,
        )
        steady_state = _flat_f64(model.policy.steady_state)
        x0_arr = model._initial_state(step.kwargs.get("x0"))
        return native_step, _order2_bindings(
            model,
            steady_state,
            x0_arr[:n_state],
            params,
            T,
        )
    raise ValueError(f"Unsupported native simulation order: {order}.")


def _order1_bindings(
    model: SolvedModel,
    steady_state: NDF,
    x0: NDF,
    params: NDF,
    T: int,
) -> tuple[FloatInputBinding, ...]:
    comp = model.compiled
    offsets = _offsets.simulation_offsets(
        1, comp.n_state, comp.n_var, comp.n_exog, T, comp.n_par
    ).foffset
    constants = (
        _flat_f64(model.policy.A),
        _flat_f64(model.policy.B),
        _flat_f64(steady_state),
        _flat_f64(x0),
    )
    return _packed_bindings(constants, offsets, params)


def _order2_bindings(
    model: SolvedModel,
    steady_state: NDF,
    x0_deviation: NDF,
    params: NDF,
    T: int,
) -> tuple[FloatInputBinding, ...]:
    policy = model.policy
    if not isinstance(policy, SecondOrderSolution):
        raise ValueError(
            "Native simulation with order=2 requires a second order solution."
        )
    comp = model.compiled
    offsets = _offsets.simulation_offsets(
        2, comp.n_state, comp.n_var, comp.n_exog, T, comp.n_par
    ).foffset
    constants = (
        _flat_f64(policy.p),
        _flat_f64(policy.f),
        _flat_f64(policy.B),
        _flat_f64(policy.hxx),
        _flat_f64(policy.gxx),
        _flat_f64(policy.hxu),
        _flat_f64(policy.gxu),
        _flat_f64(policy.huu),
        _flat_f64(policy.guu),
        _flat_f64(policy.hss),
        _flat_f64(policy.gss),
        _flat_f64(steady_state),
        _flat_f64(x0_deviation),
    )
    return _packed_bindings(constants, offsets, params)


def _packed_bindings(
    constants: tuple[NDF, ...],
    offsets: Sequence[int],
    params: NDF,
) -> tuple[FloatInputBinding, ...]:
    """Bind each staged input onto the buffer the native layout opened for it.

    The shock block trails the constants and the parameters trail that, which is
    the order both orders share.  A value the model resolved to nothing is
    skipped, and its buffer is empty rather than missing, so nothing behind it
    moves.
    """
    bindings = [
        _static_binding(values, offset)
        for values, offset in zip(constants, offsets)
        if values.size
    ]
    if params.size:
        bindings.append(_static_binding(params, offsets[len(constants) + 1]))
    return tuple(bindings)
