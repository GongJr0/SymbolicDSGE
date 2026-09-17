"""Deciding whether a shock spec can be drawn inside the native loop (#374).

Only some specs can be reproduced in C: the native draw covers the normal and
uniform families over the counter-based engine, and anything else (Student-t,
arbitrary scipy distribution objects, literal arrays) stays on the Python
prematerialization route.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple, Sequence

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from .._ckernels.monte_carlo._runner import NativeShockPlan, shock_plan
from ..core.shock.generators import Shock, ShockPath
from ..core.shock.plan import ShockPlan, ShockEntry
from ..core.shock.spec import resolve_shock_plan, _normalized_spec
from .defaults import DEFAULT_SHOCK_SCALE
from SymbolicDSGE._ckernels.monte_carlo import _runner

if TYPE_CHECKING:  # pragma: no cover; import cycle at runtime
    from ..core.solved_model import SolvedModel
    from .mc_constructs import MCStep

NDF = NDArray[float64]


class ShockCode(IntEnum):
    """Integer codes for the shock families the native draw implements.

    Mirrors the ``SDSGE_MC_SHOCK_*`` constants in
    ``_ckernels/monte_carlo/shocks.h`` -- the two MUST stay in lockstep (same
    names, same values). The code selects which standardized variate fills an
    entry's draw; every other field an entry carries is family-independent.
    """

    NORMAL = _runner.SHOCK_NORMAL
    UNIFORM = _runner.SHOCK_UNIFORM

    @classmethod
    def for_dist(cls, dist: Any) -> "ShockCode | None":
        """The code the kernel draws ``dist`` under, else ``None``.

        ``None`` is how a spec is found unported, which sends the whole spec to
        the Python draw: Student-t has no kernel, and neither has a live scipy
        distribution object, which compares equal to no family name.
        """
        if dist == "norm":
            return cls.NORMAL
        if dist == "uni":
            return cls.UNIFORM
        return None


class NativeShockEntry(NamedTuple):
    """One entry in the layout ``_runner.shock_plan`` consumes."""

    family: int
    columns: NDArray[np.int64]
    factor: NDF | None
    loc: NDF
    key: int


def _spec_family(shock: Shock | ShockPath) -> ShockCode | None:
    """The native family code for one raw spec entry, or None if C cannot draw it.

    A spec qualifies when it names a family the kernel implements, which is
    what :class:`ShockCode.for_dist` answers. A live scipy distribution object
    draws through code we have not ported, and so does Student-t.
    """
    if not isinstance(shock, Shock):
        # A supplied path is data the kernel could copy. The entry struct has no
        # family for one, and one ineligible entry sends the whole spec to the
        # Python draw.
        return None

    code = ShockCode.for_dist(shock.dist)
    if code is ShockCode.UNIFORM and len(shock.target) > 1:
        # A linear map of independent uniforms is not uniform in its margins.
        return None
    return code


def native_shock_families(
    shocks: Sequence[Shock | ShockPath],
) -> dict[tuple[str, ...], ShockCode]:
    """Family codes for a spec the native draw can take, else None.

    Eligibility is all-or-nothing: one entry the kernel cannot draw sends the
    whole spec back to the Python route, since a simulation step reads a single
    shock block.
    """
    families: dict[tuple[str, ...], ShockCode] = {}
    for shock in shocks:
        family = _spec_family(shock)
        if family is None:
            return {}
        families[shock.target] = family
    return families


def native_shock_scratch(shocks: Sequence[Shock | ShockPath], T: int) -> int:
    """Float arena elements the native draw needs.

    Reads the raw spec so arena planning can size the scratch without resolving
    a plan against the model or drawing keys it would immediately discard. The
    widest entry sets the requirement, since entries are drawn one at a time.
    """
    if not native_shock_families(shocks):
        return 0
    return T * max(len(s.target) for s in shocks)


def _entry_key(entry: ShockEntry, rng: np.random.Generator) -> int:
    """The engine key for an entry.

    A seeded spec keys on its own seed, so its draws replay run to run. An
    unseeded spec has no reproducibility to preserve, and a fresh key per run
    reproduces what the Python route did by handing ``default_rng`` a None seed.
    """
    if entry.base_seed is not None:
        return int(entry.base_seed) & 0xFFFFFFFFFFFFFFFF
    return int(rng.integers(0, 2**64, dtype=np.uint64))


def _native_entry(entry: ShockEntry, key: int, family: int) -> NativeShockEntry:
    """Both widths take one code path in C, so give univariate a 1x1 factor."""
    columns = np.asarray(entry.indices, dtype=np.int64)

    return NativeShockEntry(
        family=family,
        columns=columns,
        factor=entry.factor,
        loc=entry.loc,
        key=key,
    )


def native_shock_entries(
    plan: ShockPlan,
    families: Mapping[tuple[str, ...], int],
    rng: np.random.Generator | None = None,
) -> tuple[NativeShockEntry, ...]:
    """Lower a resolved plan into the entries the native draw reads.

    ``families`` comes from :func:`native_shock_families` on the same spec, so
    this never re-decides eligibility; it only builds what the kernel needs.
    """
    draws = np.random.default_rng() if rng is None else rng
    out: list[NativeShockEntry] = []
    for entry in plan.entries:
        if not isinstance(entry, ShockEntry):
            # Eligibility is all-or-nothing and a supplied path never qualifies,
            # so a plan that reaches here draws every one of its entries. Stated
            # rather than assumed: if eligibility ever goes per-entry, this is
            # the line that should fail.
            raise TypeError(
                f"Shock entry {entry.key!r} is a supplied path, which the native "
                "draw does not lower. This spec should have taken the Python route."
            )
        family = families[entry.key]
        key = _entry_key(entry, draws)
        out.append(_native_entry(entry, key, family))
    return tuple(out)


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
    shocks = _normalized_spec(step.kwargs.get("shocks"))
    families = native_shock_families(shocks)
    if not families:
        return None

    plan = resolve_shock_plan(model.compiled, shocks, T)
    entries = native_shock_entries(plan, families)
    return shock_plan(
        entries,
        T,
        model.compiled.n_exog,
        float(step.kwargs.get("shock_scale", DEFAULT_SHOCK_SCALE)),
    )


def replication_shocks(
    model: SolvedModel,
    step: MCStep,
    rep_idx: int,
) -> dict[tuple[str, ...], NDF]:
    """The shock paths one Monte Carlo replication saw, keyed by entry target.

    Reproduces a specific replication regardless of whether the replication was
    retained in the output. Index based seed incrementation allows resolving the
    exact ``Shock`` specification a simulation ``MCStep`` produced in-run for a
    given replication index. Only seeded entries are reproducible.

    For retained replications, MCDataGenResult.replication(retained_idx) returns
    a live ``SimResult`` including the shocks.
    """
    T = int(step.kwargs["T"])
    shocks = _normalized_spec(step.kwargs.get("shocks"))
    if not shocks:
        raise ValueError("The simulation step draws no shocks.")

    resolved = resolve_shock_plan(model.compiled, shocks, T)
    plan = build_native_plan(model, step, T)
    block = (
        resolved.matrix(
            T,
            float(step.kwargs.get("shock_scale", DEFAULT_SHOCK_SCALE)),
            rep_idx * resolved.seeded_count,
        )
        if plan is None
        else plan.draw(rep_idx)
    )

    out: dict[tuple[str, ...], NDF] = {}
    for entry in resolved.entries:
        columns = np.asarray(entry.indices, dtype=np.int64)
        out[entry.key] = block[:, columns]
    return out
