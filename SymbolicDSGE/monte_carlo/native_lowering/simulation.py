"""Native first and second-order simulation step lowering."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from SymbolicDSGE.core.solver_backend import SecondOrderSolution

from ..._ckernels.monte_carlo import _offsets
from ..._ckernels.monte_carlo._runner import (
    NativeStep,
    simulate1_step,
    simulate2_step,
)
from ...core.solved_model import SolvedModel
from ...core.solved_model.shocks import resolve_shock_plan, simulation_shock_matrix
from ..defaults import (
    DEFAULT_SHOCK_SCALE,
    DEFAULT_SIMULATION_OBSERVABLES,
    DEFAULT_SIMULATION_TARGET,
)
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
    reference: SolvedModel,
    dgp: SolvedModel | None,
    n_rep: int,
) -> tuple[NativeStep, tuple[FloatInputBinding, ...]]:
    """Compile one model simulation into the native simulation ABI."""
    T = int(step.kwargs["T"])
    target = step.kwargs.get("target", DEFAULT_SIMULATION_TARGET)
    if target not in {"reference", "dgp"}:
        raise ValueError(f"Unsupported simulation target: {target!r}.")
    model = reference if target == "reference" else dgp
    if model is None:
        raise ValueError("Simulation step requires its target model.")

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
    drawn = build_native_plan(model, step, T)
    if drawn is None:
        shocks, shocks_batched = _simulation_shocks(model, step, T, n_rep)
    else:
        # The step draws its own block per replication, so nothing is bound in.
        shocks, shocks_batched = np.zeros((0, n_exog), dtype=np.float64), False
    order = model.policy.order

    if order == 1:
        native_step = simulate1_step(
            step.name, measurement_addr, T, n_var, n_exog, n_par, n_obs, drawn
        )
        steady_state = _flat_f64(model.policy.steady_state)
        x0 = model._initial_state(step.kwargs.get("x0"))
        return native_step, _order1_bindings(
            model, steady_state, x0, shocks, shocks_batched, params, T
        )
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
            drawn,
        )
        steady_state = _flat_f64(model.policy.steady_state)
        x0_arr = model._initial_state(step.kwargs.get("x0"))
        return native_step, _order2_bindings(
            model,
            steady_state,
            x0_arr[:n_state],
            shocks,
            shocks_batched,
            params,
            T,
        )
    raise ValueError(f"Unsupported native simulation order: {order}.")


def _order1_bindings(
    model: SolvedModel,
    steady_state: NDF,
    x0: NDF,
    shocks: NDF,
    shocks_batched: bool,
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
    return _packed_bindings(constants, offsets, shocks, shocks_batched, params)


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
    return _packed_bindings(constants, offsets, shocks, shocks_batched, params)


def _packed_bindings(
    constants: tuple[NDF, ...],
    offsets: Sequence[int],
    shocks: NDF,
    shocks_batched: bool,
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
    if shocks.size:
        bindings.append(
            _static_binding(shocks, offsets[len(constants)], batched=shocks_batched)
        )
    if params.size:
        bindings.append(_static_binding(params, offsets[len(constants) + 1]))
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
    shocks = step.kwargs.get("shocks")
    shock_scale = float(step.kwargs.get("shock_scale", DEFAULT_SHOCK_SCALE))
    if shocks is None:
        return _array_shocks(model, T, shock_scale), False

    validate_shock_specs(shocks)
    plan = resolve_shock_plan(model.compiled, shocks, T)

    values = np.zeros((n_rep, T, model.compiled.n_exog), dtype=np.float64)
    for rep_idx in range(n_rep):
        plan.fill(values[rep_idx], T, shock_scale, rep_idx * plan.seeded_count)
    return values, True


def _array_shocks(model: SolvedModel, T: int, shock_scale: float) -> NDF:
    return np.ascontiguousarray(
        simulation_shock_matrix(model.compiled, T, shock_scale=shock_scale),
        dtype=np.float64,
    )
