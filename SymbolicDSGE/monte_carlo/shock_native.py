"""Recover the shock paths used by Monte Carlo replications."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..core.shock.native import build_native_plan
from ..core.shock.spec import resolve_shock_plan, _normalized_spec
from .defaults import DEFAULT_SHOCK_SCALE

if TYPE_CHECKING:
    from ..core.solved_model import SolvedModel
    from .mc_constructs import MCStep

NDF = NDArray[float64]


def replication_shocks(
    model: SolvedModel,
    step: MCStep,
    rep_idx: int,
) -> dict[tuple[str, ...], NDF]:
    """The shock paths one Monte Carlo replication saw, keyed by entry target.

    Reproduces a specific replication regardless of whether the replication was
    retained in the output: an entry is seeded off its own spec member and the
    replication index, so a given index resolves to the draw the run made. Only
    seeded entries are reproducible.

    For retained replications, MCDataGenResult.replication(retained_idx) returns
    a live ``SimResult`` including the shocks.
    """
    T = int(step.kwargs["T"])
    shocks = _normalized_spec(step.kwargs.get("shocks"))
    if not shocks:
        raise ValueError("The simulation step draws no shocks.")

    resolved = resolve_shock_plan(model.compiled, shocks, T)
    plan = build_native_plan(
        model.compiled,
        shocks,
        T,
        float(step.kwargs.get("shock_scale", DEFAULT_SHOCK_SCALE)),
    )
    block = (
        resolved.matrix(
            T,
            float(step.kwargs.get("shock_scale", DEFAULT_SHOCK_SCALE)),
            rep_idx,
        )
        if plan is None
        else plan.draw(rep_idx)
    )

    out: dict[tuple[str, ...], NDF] = {}
    for entry in resolved.entries:
        columns = np.asarray(entry.indices, dtype=np.int64)
        out[entry.key] = block[:, columns]
    return out
