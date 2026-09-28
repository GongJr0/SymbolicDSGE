"""Recover the shock paths used by Monte Carlo replications."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import float64
from numpy.typing import NDArray

from ..core.shock.plan import draw_shock_matrix
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
    scale = float(step.kwargs.get("shock_scale", DEFAULT_SHOCK_SCALE))
    shocks = _normalized_spec(step.kwargs.get("shocks"))
    if not shocks:
        raise ValueError("The simulation step draws no shocks.")

    plan = resolve_shock_plan(model.compiled, shocks, T)
    drawn = draw_shock_matrix(plan, T, scale, rep_idx)

    out = {}
    for entry in plan.entries:
        out[entry.key] = drawn[:, entry.indices]
    return out
