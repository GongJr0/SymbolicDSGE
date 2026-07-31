"""Native first and second-order simulation step lowering."""

from __future__ import annotations

import numpy as np
from typing import Callable, Mapping

from SymbolicDSGE.core.solver_backend import PerturbationSolution

from ..._ckernels.monte_carlo._runner import NativeStep, simulate1_step, simulate2_step
from ...core.shock_generators import Shock
from ...core.solved_model import SolvedModel
from ..allocation import StepBufferPlan
from ..mc_constructs import MCStep, SeedIncrement, ShockMapping
from .utils import (
    NDF,
    FloatInputBinding,
    _f64,
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
    shocks, shocks_batched = _simulation_shocks(model, step, T, n_rep)
    order = model.policy.order

    if order == 1:
        _check_simulation_layout(step_plan, T, n_var, n_obs)
        native_step = simulate1_step(
            step.name, measurement_addr, T, n_var, n_exog, n_par, n_obs
        )
        x0 = model._simulation_initial_state(step.kwargs["x0"])
        return native_step, _order1_bindings(model, x0, shocks, shocks_batched, params)
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
        return native_step, _order2_bindings(
            model,
            steady_state,
            x0_deviation,
            shocks,
            shocks_batched,
            params,
        )
    raise ValueError(f"Unsupported native simulation order: {order}.")


def _order1_bindings(
    model: SolvedModel,
    x0: NDF,
    shocks: NDF,
    shocks_batched: bool,
    params: NDF,
) -> tuple[FloatInputBinding, ...]:
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


def _order2_bindings(
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
    n_exog = model.compiled.n_exog
    T = shocks.shape[-2]
    values_by_layout = (
        _f64(policy.p),
        _f64(policy.f),
        _f64(model.B[: model.compiled.n_state, :]),
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
        return _array_shocks(model, T, shock_scale), False
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


def _clone_or_pass_shocks(
    shocks: ShockMapping | None,
    *,
    T: int,
    rep_idx: int,
    seed_increment: SeedIncrement,
) -> Mapping[str, Callable[[float | NDF], NDF] | NDF] | None:
    if shocks is None:
        return None
    out: dict[str, Callable[[float | NDF], NDF] | NDF] = {}
    seed_offset = rep_idx * _resolve_seed_increment(shocks, seed_increment)
    for name, shock in shocks.items():
        if isinstance(shock, Shock):
            if shock.shock_arr is not None:
                raise ValueError(
                    "MC simulation requires generator-style Shock instances."
                )
            if ("," in name) != shock.multivar:
                raise ValueError(
                    f"Shock '{name}' must set multivar={',' in name} to match its specification."
                )
            seed = None if shock.seed is None else int(shock.seed) + seed_offset
            out[name] = Shock(
                dist=shock.dist,  # pyright: ignore
                multivar=shock.multivar,
                seed=seed,
                dist_args=shock.dist_args,
                dist_kwargs=shock.dist_kwargs.copy(),
            ).shock_generator(T)
        else:
            out[name] = shock
    return out


def _resolve_seed_increment(
    shocks: ShockMapping,
    seed_increment: SeedIncrement,
) -> int:
    if seed_increment == "auto":
        return sum(
            1
            for shock in shocks.values()
            if isinstance(shock, Shock) and shock.seed is not None
        )
    increment = int(seed_increment)
    if increment < 0:
        raise ValueError("seed_increment must be non-negative or 'auto'.")
    return increment
