from __future__ import annotations

from typing import Literal, Mapping, Sequence

import numpy as np
from numpy import float64, ndarray


from ....core.solved_model import SolvedModel
from ....kalman.filter import FilterRawResult, UnscentedFilterRawResult
from ...mc_constructs import MCContext, MCData, NDF, SeedIncrement, ShockMapping
from ..utils import _clone_or_pass_shocks, _select_raw_rep_array


def simulate(
    *,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    target: Literal["reference", "dgp"] = "dgp",
    T: int,
    shocks: ShockMapping | None = None,
    seed_increment: SeedIncrement = "auto",
    shock_scale: float | float64 = 1.0,
    x0: ndarray | None = None,
    observables: bool = True,
) -> MCData:
    target_valid = target in ("reference", "dgp")
    if not target_valid:
        raise ValueError(f"Invalid target '{target}'; must be 'reference' or 'dgp'.")
    model = reference if target == "reference" else dgp
    if model is None:
        raise ValueError(f"simulate requires a {target} SolvedModel.")

    sim_shocks = _clone_or_pass_shocks(
        shocks,
        T=T,
        rep_idx=rep_idx,
        seed_increment=seed_increment,
    )
    states = np.ascontiguousarray(
        model._simulate_state_matrix(
            T=T,
            shocks=sim_shocks,
            shock_scale=shock_scale,
            x0=x0,
        ),
        dtype=np.float64,
    )
    obs_names = (
        tuple(getattr(model.compiled, "observable_names", ())) if observables else ()
    )
    obs_mat = None
    raw: dict[str, np.ndarray] = {
        name: states[:, model.compiled.idx[name]] for name in model.compiled.var_names
    }
    raw["_X"] = states
    if obs_names:
        obs_full = model._simulate_observable_matrix(states, drop_initial=False)
        obs_mat = np.ascontiguousarray(obs_full[1:], dtype=np.float64)
        for i, name in enumerate(obs_names):
            raw[name] = obs_full[:, i]
    return MCData(
        states=states,
        observables=obs_mat,
        raw=raw,
        n_exog=model.compiled.n_exog,
        observable_names=obs_names,
    )


def raw_model_data_datagen(
    *,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    states: NDF | None = None,
    observables: NDF | None = None,
    raw: Mapping[str, NDF] | None = None,
    observable_names: Sequence[str] = (),
) -> MCData:
    del reference, dgp
    if states is None and observables is None:
        raise ValueError(
            "raw_model_data_datagen requires states, observables, or both."
        )

    state_mat = None
    if states is not None:
        state_mat = _select_raw_rep_array("states", states)
    obs_mat = None
    if observables is not None:
        obs_mat = _select_raw_rep_array("observables", observables)
    raw_payload: dict[str, NDF] = {}
    if state_mat is not None:
        raw_payload["_X"] = state_mat
    if raw is not None:
        raw_payload.update(
            {
                key: _select_raw_rep_array(f"raw['{key}']", value)
                for key, value in raw.items()
            }
        )
    return MCData(
        states=state_mat,
        observables=obs_mat,
        n_exog=-1,
        raw=raw_payload,
        observable_names=tuple(observable_names),
    )


def run_reference_filter(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    filter_mode: Literal["linear", "extended", "unscented"] = "linear",
    observables: list[str] | None = None,
    x0: NDF | None = None,
    p0_mode: Literal["diag", "eye"] | None = None,
    p0_scale: float | float64 | None = None,
    jitter: float | float64 | None = None,
    symmetrize: bool | None = None,
    return_shocks: bool = False,
    R: NDF | None = None,
    estimate_R_diag: bool = False,
    R_scale: float = 1.0,
) -> FilterRawResult | UnscentedFilterRawResult:
    del dgp, rep_idx
    data = context.require_data()
    if data.observables is None:
        raise ValueError("Reference filter step requires generated observables.")
    obs = observables
    if obs is None and data.observable_names:
        obs = list(data.observable_names)
    return reference._kalman_raw(
        y=data.observables,
        filter_mode=filter_mode,
        observables=obs,
        x0=x0,
        p0_mode=p0_mode,
        p0_scale=p0_scale,
        jitter=jitter,
        symmetrize=symmetrize,
        return_shocks=return_shocks,
        R=R,
        estimate_R_diag=estimate_R_diag,
        R_scale=R_scale,
    )


def add_payload(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    value: NDF | Sequence[float] | Sequence[Sequence[float]],
) -> NDF:
    del context, reference, dgp, rep_idx

    out = np.asarray(value, dtype=np.float64)
    if not out.ndim in (1, 2):
        raise ValueError(f"Payload must be 1-D, or 2-D; got {out.ndim}-D.")
    return out
