"""Native first and second-order simulation step lowering."""

from __future__ import annotations

import numpy as np

from SymbolicDSGE.core.solver_backend import PerturbationSolution

from ..._ckernels.monte_carlo._runner import (
    NativeStep,
    simulate1_step,
    simulate2_step,
)
from ...core.solved_model import SolvedModel
from ..allocation import StepBufferPlan
from ..mc_constructs import MCStep
from ..shock_native import build_native_plan, validate_shock_specs
from .utils import (
    NDF,
    FloatInputBinding,
    _flat_f64,
    _model_params,
    _static_binding,
)


def lower_simulation_step(
    step: MCStep,
    step_plan: StepBufferPlan,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    n_rep: int,
) -> tuple[NativeStep, tuple[FloatInputBinding, ...]]:
    """Compile one model simulation into the native simulation ABI."""
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
    drawn = build_native_plan(model, step, T)
    if drawn is None:
        shocks, shocks_batched = _simulation_shocks(model, step, T, n_rep)
    else:
        # The step draws its own block per replication, so nothing is bound in.
        shocks, shocks_batched = np.zeros((0, n_exog), dtype=np.float64), False
    order = model.policy.order

    if order == 1:
        _check_simulation_layout(step_plan, T, n_var, n_obs)
        native_step = simulate1_step(
            step.name, measurement_addr, T, n_var, n_exog, n_par, n_obs, drawn
        )
        x0 = model._simulation_initial_state(step.kwargs["x0"])
        return native_step, _order1_bindings(
            model, x0, shocks, shocks_batched, params, T
        )
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
            drawn,
        )
        steady_state = _flat_f64(model.policy.steady_state)
        initial_state = model._simulation_initial_state(step.kwargs["x0"])
        x0_deviation = initial_state[:n_state] - steady_state[:n_state]
        return native_step, _order2_bindings(
            model,
            steady_state,
            x0_deviation,
            shocks,
            shocks_batched,
            params,
            T,
        )
    raise ValueError(f"Unsupported native simulation order: {order}.")


def _order1_bindings(
    model: SolvedModel,
    x0: NDF,
    shocks: NDF,
    shocks_batched: bool,
    params: NDF,
    T: int,
) -> tuple[FloatInputBinding, ...]:
    n_exog = model.compiled.n_exog
    bindings: list[FloatInputBinding] = []
    offset = 0
    for values in (_flat_f64(model.A), _flat_f64(model.B), _flat_f64(x0)):
        if values.size:
            bindings.append(_static_binding(values, offset))
        offset += values.size
    if shocks.size:
        bindings.append(_static_binding(shocks, offset, batched=shocks_batched))
    offset += T * n_exog
    if params.size:
        bindings.append(_static_binding(params, offset))
    return tuple(bindings)


def _order2_bindings(
    model: SolvedModel,
    steady_state: NDF,
    x0_deviation: NDF,
    shocks: NDF,
    shocks_batched: bool,
    params: NDF,
    T: int,
) -> tuple[FloatInputBinding, ...]:
    policy = model.policy
    if not isinstance(policy, PerturbationSolution):
        raise ValueError("Native simulation order 2 requires a perturbation solution.")
    n_exog = model.compiled.n_exog
    values_by_layout = (
        _flat_f64(policy.p),
        _flat_f64(policy.f),
        _flat_f64(model.B[: model.compiled.n_state, :]),
        _flat_f64(policy.hxx),
        _flat_f64(policy.gxx),
        _flat_f64(policy.hss),
        _flat_f64(policy.gss),
        _flat_f64(steady_state),
        _flat_f64(x0_deviation),
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
    """Materialize the ``(n_rep, T, n_exog)`` shock slab the native loop reads.

    The spec resolves against the model once. Only the seed varies per
    replication, so the loop below reseeds and draws straight into its own row
    of the slab; the calibration lookups, the covariance assembly, and its
    Cholesky are not repeated. Each replication shifts every base seed by the
    number of seeded entries, which keeps entries that share a run apart.
    """
    shocks = step.kwargs["shocks"]
    shock_scale = float(step.kwargs["shock_scale"])
    if shocks is None:
        return _array_shocks(model, T, shock_scale), False

    validate_shock_specs(shocks)
    plan = model._resolve_shock_plan(shocks, T)

    values = np.zeros((n_rep, T, model.compiled.n_exog), dtype=np.float64)
    for rep_idx in range(n_rep):
        plan.fill(values[rep_idx], T, shock_scale, rep_idx * plan.seeded_count)
    return values, True


def _array_shocks(model: SolvedModel, T: int, shock_scale: float) -> NDF:
    return np.ascontiguousarray(
        model._simulation_shock_matrix(T, shock_scale=shock_scale),
        dtype=np.float64,
    )


def _check_simulation_layout(
    step_plan: StepBufferPlan,
    T: int,
    n_var: int,
    n_obs: int,
) -> None:
    states = step_plan.out_fields["states"]
    if states.offset != 0 or states.shape != (T, n_var):
        raise ValueError("Native simulation states do not match their output layout.")
    if n_obs:
        observables = step_plan.out_fields["observables"]
        if observables.offset != states.flat_count or observables.shape != (T, n_obs):
            raise ValueError(
                "Native simulation observables do not match their output layout."
            )
