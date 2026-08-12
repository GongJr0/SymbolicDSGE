"""Deciding whether a shock spec can be drawn inside the native loop (#374).

The Monte Carlo hot loop draws its own shocks from ``rep_idx``, which removes
the per-replication Python that used to materialize an ``(n_rep, T, n_exog)``
slab before the run. Only some specs can be reproduced in C: the native draw
covers the normal and uniform families over the counter-based engine, and
anything else (Student-t, arbitrary scipy distribution objects, user callables,
literal arrays) stays on the Python prematerialization route.

That decision is needed twice and must agree both times. Arena planning runs
before lowering and has to size the draw's scratch, while lowering builds the
plan the kernel reads. Both go through :func:`native_shock_families` here, which
reads the raw spec alone, so planning never has to resolve a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from .._ckernels.monte_carlo._runner import NativeShockPlan, shock_plan
from ..core.shock_generators import Shock
from ..core.shock_plan import ShockPlan, ShockPlanEntry
from ..core.solved_model.shocks import resolve_shock_plan

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime
    from ..core.solved_model import SolvedModel
    from .mc_constructs import MCStep

NDF = NDArray[float64]

# Mirrors the SDSGE_MC_SHOCK_* constants in _ckernels/monte_carlo/shocks.h.
SHOCK_NORMAL = 0
SHOCK_UNIFORM = 1

_NATIVE_FAMILIES = {"norm": SHOCK_NORMAL, "uni": SHOCK_UNIFORM}


@dataclass(frozen=True)
class NativeShockEntry:
    """One entry in the layout ``_runner.shock_plan`` consumes."""

    family: int
    columns: NDArray[np.int64]
    factor: NDF | None
    loc: NDF | None
    low: float
    span: float
    key: int

    def as_tuple(self) -> tuple:
        return (
            self.family,
            self.columns,
            self.factor,
            self.loc,
            self.low,
            self.span,
            self.key,
        )


def _spec_family(name: str, shock: Any) -> int | None:
    """The native family code for one raw spec entry, or None if C cannot draw it.

    A spec only qualifies when it names a known family as a string with no
    positional arguments, which is exactly the condition under which
    :meth:`Shock.draw_fn` itself bypasses scipy. Anything routed through a scipy
    distribution object draws through code we have not ported.
    """
    if not isinstance(shock, Shock):
        return None  # A literal array or a user callable.
    if shock.shock_arr is not None or shock.dist_args:
        return None
    if not isinstance(shock.dist, str):
        return None
    family = _NATIVE_FAMILIES.get(shock.dist)
    if family is None:
        return None  # Student-t and anything else unported.
    if family == SHOCK_UNIFORM and "," in name:
        return None
    return family


def native_shock_families(
    shocks: Mapping[str, Any] | None,
) -> dict[str, int] | None:
    """Family codes for a spec the native draw can take, else None.

    Eligibility is all-or-nothing: one entry the kernel cannot draw sends the
    whole spec back to the Python route, since a simulation step reads a single
    shock block.
    """
    if not shocks:
        return None
    families: dict[str, int] = {}
    for name, shock in shocks.items():
        family = _spec_family(name, shock)
        if family is None:
            return None
        families[name] = family
    return families


def native_shock_scratch(shocks: Mapping[str, Any] | None, T: int) -> int:
    """Float arena elements the native draw needs, or 0 when it does not run.

    Reads the raw spec so arena planning can size the scratch without resolving
    a plan against the model or drawing keys it would immediately discard. The
    widest entry sets the requirement, since entries are drawn one at a time.
    """
    if native_shock_families(shocks) is None:
        return 0
    assert shocks is not None
    return T * max(len(name.split(",")) for name in shocks)


def _entry_key(entry: ShockPlanEntry, rng: np.random.Generator) -> int:
    """The engine key for an entry.

    A seeded spec keys on its own seed, so its draws replay run to run. An
    unseeded spec has no reproducibility to preserve, and a fresh key per run
    reproduces what the Python route did by handing ``default_rng`` a None seed.
    """
    if entry.base_seed is not None:
        return int(entry.base_seed) & 0xFFFFFFFFFFFFFFFF
    return int(rng.integers(0, 2**64, dtype=np.uint64))


def _normal_entry(entry: ShockPlanEntry, key: int) -> NativeShockEntry:
    """Both widths take one code path in C, so give univariate a 1x1 factor."""
    kwargs = {} if entry.spec is None else entry.spec.dist_kwargs
    columns = np.asarray(entry.indices, dtype=np.int64)

    if entry.multivar:
        factor = np.ascontiguousarray(entry.factor, dtype=np.float64)
        mean = kwargs.get("mean")
        loc = None if mean is None else np.asarray(mean, dtype=np.float64)
    else:
        factor = np.asarray([[float(entry.scale)]], dtype=np.float64)  # type: ignore[arg-type]
        loc_value = float(kwargs.get("loc", 0.0))
        loc = None if loc_value == 0.0 else np.asarray([loc_value], dtype=np.float64)

    return NativeShockEntry(
        family=SHOCK_NORMAL,
        columns=columns,
        factor=factor,
        loc=loc,
        low=0.0,
        span=0.0,
        key=key,
    )


def _uniform_entry(entry: ShockPlanEntry, key: int) -> NativeShockEntry:
    """scipy's uniform is parameterized by ``loc`` and a width, not by bounds."""
    kwargs = {} if entry.spec is None else entry.spec.dist_kwargs
    return NativeShockEntry(
        family=SHOCK_UNIFORM,
        columns=np.asarray(entry.indices, dtype=np.int64),
        factor=None,
        loc=None,
        low=float(kwargs.get("loc", 0.0)),
        span=float(entry.scale),  # type: ignore[arg-type]
        key=key,
    )


def native_shock_entries(
    plan: ShockPlan,
    families: Mapping[str, int],
    rng: np.random.Generator | None = None,
) -> tuple[NativeShockEntry, ...]:
    """Lower a resolved plan into the entries the native draw reads.

    ``families`` comes from :func:`native_shock_families` on the same spec, so
    this never re-decides eligibility; it only builds what the kernel needs.
    """
    draws = np.random.default_rng() if rng is None else rng
    out: list[NativeShockEntry] = []
    for entry in plan.entries:
        family = families[entry.key]
        key = _entry_key(entry, draws)
        out.append(
            _normal_entry(entry, key)
            if family == SHOCK_NORMAL
            else _uniform_entry(entry, key)
        )
    return tuple(out)


def validate_shock_specs(shocks: Mapping[str, Any]) -> None:
    """Reject shock specs the per-replication redraw cannot honor."""
    for name, shock in shocks.items():
        if not isinstance(shock, Shock):
            continue
        if shock.shock_arr is not None:
            raise ValueError("MC simulation requires generator-style Shock instances.")
        if ("," in name) != shock.multivar:
            raise ValueError(
                f"Shock '{name}' must set multivar={',' in name} to match its "
                "specification."
            )


def build_native_plan(
    model: SolvedModel,
    step: MCStep,
    T: int,
) -> NativeShockPlan | None:
    """The plan a simulation step draws from, or None to prematerialize instead.

    Returning None is the fallback: the spec names a family the kernel cannot
    reproduce (Student-t, a scipy distribution object, a user callable, a
    literal array), so the caller draws every replication in Python up front.
    """
    shocks = step.kwargs["shocks"]
    families = native_shock_families(shocks)
    if families is None:
        return None

    validate_shock_specs(shocks)
    plan = resolve_shock_plan(model.compiled, shocks, T)
    entries = native_shock_entries(plan, families)
    return shock_plan(
        [entry.as_tuple() for entry in entries],
        T,
        model.compiled.n_exog,
        float(step.kwargs["shock_scale"]),
    )


def replication_shocks(
    model: SolvedModel,
    step: MCStep,
    rep_idx: int,
) -> dict[str, NDF]:
    """The shock paths one Monte Carlo replication saw, keyed by spec name.

    A Monte Carlo replication is not reproducible by rerunning the pipeline with
    a smaller ``n_rep``, because the loop addresses its own stream per
    replication rather than replaying a shared one. This is the way back to a
    single replication: the result feeds straight into
    ``model.sim(T, shocks=..., shock_scale=1.0)``, which reproduces exactly what
    replication ``rep_idx`` simulated. Scaling is already applied, hence the
    ``shock_scale=1.0``.

    ``step`` must be the same simulation step the run used, and ``model`` the
    role it targeted. Only specifications carrying a seed are reproducible: one
    with ``seed=None`` was drawn from a key the run discarded, so what comes
    back for it is a fresh path rather than the one that ran.
    """
    T = int(step.kwargs["T"])
    shocks = step.kwargs["shocks"]
    if not shocks:
        raise ValueError("The simulation step draws no shocks.")

    validate_shock_specs(shocks)
    resolved = resolve_shock_plan(model.compiled, shocks, T)
    plan = build_native_plan(model, step, T)
    block = (
        resolved.matrix(
            T, float(step.kwargs["shock_scale"]), rep_idx * resolved.seeded_count
        )
        if plan is None
        else plan.draw(rep_idx)
    )

    out: dict[str, NDF] = {}
    for entry in resolved.entries:
        columns = np.asarray(entry.indices, dtype=np.int64)
        out[entry.key] = block[:, columns] if entry.multivar else block[:, columns[0]]
    return out
