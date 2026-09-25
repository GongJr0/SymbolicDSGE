"""Native shock eligibility and plan construction shared by simulation callers."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from ..._ckernels.core._shocks import (
    NativeShockPlan,
    shock_plan,
    SHOCK_NORMAL,
    SHOCK_UNIFORM,
)
from .generators import Shock, ShockPath
from .plan import ShockPlan, ShockEntry, NativeShockEntry
from .spec import ShockSpec, resolve_shock_plan, _normalized_spec

if TYPE_CHECKING:
    from ..compiled_model import CompiledModel


class ShockCode(IntEnum):
    """Integer codes for the shock families the native draw implements.

    Uses the ``SDSGE_SHOCK_*`` enum values from ``_ckernels/core/shocks.h``,
    exported by the ``_shocks`` extension. The code selects which variate fills an
    entry's draw; every other field an entry carries is family-independent.
    """

    NORMAL = SHOCK_NORMAL
    UNIFORM = SHOCK_UNIFORM

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
    compiled: CompiledModel,
    shocks: ShockSpec | None,
    T: int,
    shock_scale: float = 1.0,
) -> NativeShockPlan | None:
    """Build a native plan from a compiled model and shock specification.

    Returning None is the fallback: the spec names a family the kernel cannot
    reproduce (Student-t, a scipy distribution object, a user callable, a
    literal array), so the caller draws every replication in Python up front.
    """
    shocks = _normalized_spec(shocks)
    families = native_shock_families(shocks)
    if not families:
        return None

    plan = resolve_shock_plan(compiled, shocks, T)
    entries = native_shock_entries(plan, families)
    return shock_plan(
        entries,
        T,
        compiled.n_exog,
        shock_scale,
    )
