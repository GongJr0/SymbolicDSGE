"""Resolved shock specifications, separated from the draws they produce.

A :class:`ShockPlan` is the first half of resolution. Callers that redraw the
same spec under many seeds (the Monte Carlo lowering materializes one path per
replication) resolve a plan and then call :meth:`ShockPlan.fill` per draw.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence
from functools import cached_property

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from .generators import Shock, ShockPath, ShockDrawFn
from ..._ckernels.core._shocks import (
    SHOCK_NORMAL,
    SHOCK_UNIFORM,
    NativeShockPlan,
    native_shock_plan,
)

NDF = NDArray[float64]


class ShockCode(IntEnum):
    """Integer codes for the shock families the native draw implements.

    Uses the ``SDSGE_SHOCK_*`` enum values from ``_ckernels/core/shocks.h``,
    exported by the ``_shocks`` extension. The code selects which variate fills an
    entry's draw; every other field an entry carries is family-independent.
    """

    NORMAL = SHOCK_NORMAL
    UNIFORM = SHOCK_UNIFORM

    @classmethod
    def for_dist(cls, dist: object, width: int) -> "ShockCode | None":
        """Return a code when the native kernel supports the family and width."""
        if not isinstance(dist, str) or width < 1:
            return None
        if dist == "norm":
            return cls.NORMAL
        if dist == "uni" and width == 1:
            return cls.UNIFORM
        return None


@dataclass(frozen=True)
class ShockEntry:
    """One drawn entry of a shock spec, resolved against a model.

    ``draw`` is the entry's family resolved for one horizon, and it computes
    ``loc + factor @ v`` over that family's standardized variate. ``factor`` is
    the scale at any width: the 1x1 holding a standard deviation, or the
    covariance block's factor. ``loc`` is the ``width``-long location.
    ``base_seed`` is the spec's own seed as declared, which :meth:`_seed` keys a
    per-replication draw from rather than using directly.

    ``kwargs`` carries what is neither location nor scale forward for the native
    lowering.
    """

    key: tuple[str, ...]
    indices: tuple[int, ...]
    family: ShockCode | None
    loc: NDF
    factor: NDF
    draw: ShockDrawFn
    base_seed: int | None = None
    kwargs: dict | None = None

    @cached_property
    def width(self) -> int:
        """Number of columns this entry targets."""
        return len(self.indices)

    def unpack(self, rep_idx: int = 0) -> list[tuple[int, NDF]]:
        """Draw this entry and pair each column with its exogenous index.

        ``rep_idx`` selects which replication of the spec to draw, which is what
        keeps the replications of one Monte Carlo run on different paths. An
        unseeded entry has no seed to key and redraws freshly whatever it is
        given.
        """
        drawn = self.draw(
            self.loc,
            self.factor,
            self._seed(rep_idx),
        )

        if drawn.ndim != 2 or drawn.shape[1] != self.width:
            raise ValueError(
                f"Draw for {self.key!r} must return shape (T, {self.width}); "
                f"got {tuple(drawn.shape)}."
            )
        return list(zip(self.indices, (drawn[:, i] for i in range(self.width))))

    def _seed(self, rep_idx: int = 0) -> int | None:
        """The seed for one replication of this entry, or None when unseeded.

        ``rep_idx`` takes 0 for any non-MC draw, where replications do not apply.
        ``indices[0]`` is the minimum canonical shock index in the entry, which
        separates entries uniquely because no two can share a shock index (#507).

        Mixing the three inputs is what makes a draw a function of the entry and
        the replication alone. Adding them cannot: a declared seed is unbounded
        and user-chosen, so any additive qualifier is congruent to some other
        entry's seed, which is how two entries used to land on one stream.

        ``SeedSequence`` serves as the mixing function rather than as an RNG
        state, so the result is a plain 64-bit integer.
        """
        if self.base_seed is None:
            return None
        return int(
            np.random.SeedSequence(
                entropy=self.base_seed, spawn_key=(self.indices[0], rep_idx)
            ).generate_state(1, dtype=np.uint64)[0]
        )

    @property
    def _native_seed_key(self) -> int:
        if self.base_seed is None:
            rng = np.random.default_rng()
            return int(rng.integers(0, 2**64, dtype=np.uint64))
        return int(self.base_seed) & 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True)
class ArrayEntry:
    """One literal path of a shock spec, resolved against a model.

    Nothing about a supplied path depends on the calibration: this carries no
    scale, no factor, and no seed. ``value`` is always ``(T, width)``. The
    resolution widens a single shock's one-dimensional path, which lets every
    entry unpack the same way.
    """

    key: tuple[str, ...]
    indices: tuple[int, ...]
    value: NDF

    @cached_property
    def width(self) -> int:
        """Number of columns this entry targets."""
        return len(self.indices)

    def unpack(self, rep_idx: int = 0) -> list[tuple[int, NDF]]:
        """Pair each column of the supplied path with its exogenous index.

        Takes ``rep_idx`` to match :meth:`ShockEntry.unpack`; a plan then
        unpacks its entries without asking which kind each one is. A supplied
        path has no seed to shift.
        """
        del rep_idx
        if self.value.ndim != 2 or self.value.shape[1] != self.width:
            raise ValueError(
                f"Array entry for {self.key!r} must have shape (T, {self.width}); "
                f"got {tuple(self.value.shape)}."
            )
        return list(zip(self.indices, (self.value[:, i] for i in range(self.width))))


@dataclass(frozen=True)
class ShockPlan:
    """A shock spec resolved against a model, ready to draw from repeatedly."""

    entries: tuple[ShockEntry | ArrayEntry, ...]
    n_exog: int

    def unpack(self, rep_idx: int = 0) -> list[tuple[int, NDF]]:
        """Draw every entry as ``(exogenous index, column)`` pairs."""
        out: list[tuple[int, NDF]] = []
        for entry in self.entries:
            out.extend(entry.unpack(rep_idx))
        return out

    def fill(
        self,
        out: NDF,
        T: int,
        shock_scale: float = 1.0,
        rep_idx: int = 0,
    ) -> None:
        """Draw into a preallocated ``(T, n_exog)`` view.

        Writing through a caller-owned view lets the Monte Carlo lowering target
        a row of its ``(n_rep, T, n_exog)`` slab directly, with no per-draw
        temporary. Columns no entry targets are left untouched, so the caller
        owns zeroing.
        """
        for idx, values in self.unpack(rep_idx):
            if values.shape[0] != T:
                raise ValueError(
                    f"Shock array for variable index {idx} must have length {T}."
                )
            out[:, idx] = shock_scale * values

    def matrix(self, T: int, shock_scale: float = 1.0, rep_idx: int = 0) -> NDF:
        """Draw a fresh ``(T, n_exog)`` shock matrix."""
        out = np.zeros((T, self.n_exog), dtype=float64)
        self.fill(out, T, shock_scale, rep_idx)
        return out


def validate_shock_targets(
    shocks: Sequence[Shock | ShockPath],
    shock_names: Sequence[str],
) -> None:
    """Check every entry names model shocks, each owned by one entry.

    One pass over the spec, which is what catches a shock shared across two
    grouped entries (``("e_g", "e_z")`` and ``("e_g", "e_r")``). An exact
    duplicate is caught by the same pass: the sequence shape can repeat an
    entry, unlike the mapping shape, whose keys deduplicate upstream.
    """
    shock_set = set(shock_names)
    owner: dict[str, str | Sequence[str]] = {}
    for shock in shocks:
        members = shock.target
        if not members:
            raise ValueError("Shock entries must name at least one shock.")
        for member in members:
            if member not in shock_set:
                where = f" in entry {','.join(members)!r}" if len(members) > 1 else ""
                raise ValueError(
                    f"Shock {member!r}{where} is not a model shock. "
                    f"Valid shocks: {list(shock_names)}."
                )
            if member in owner:
                raise ValueError(
                    f"Shock {member!r} is driven by more than one shock entry "
                    f"({owner[member]!r} and {','.join(members)!r}); each shock may appear "
                    "in at most one entry."
                )
            owner[member] = ",".join(members)


def is_native_spec_eligible(shocks: Sequence[Shock | ShockPath]) -> bool:
    """Check a normalized specification without resolving a plan or drawing keys.

    Empty specifications need no native draw. Any supplied path or unsupported
    family/width combination selects whole-spec Python materialization.
    """
    return bool(shocks) and all(
        isinstance(shock, Shock)
        and ShockCode.for_dist(shock.dist, len(shock.target)) is not None
        for shock in shocks
    )


def _spec_family(shock: ShockEntry | ArrayEntry) -> ShockCode | None:
    """The native family code for one raw spec entry, or None if C cannot draw it.

    A spec qualifies when it names a family the kernel implements, which is
    what :class:`ShockCode.for_dist` answers. A live scipy distribution object
    draws through code we have not ported, and so does Student-t.
    """
    if not isinstance(shock, ShockEntry):
        # A supplied path is data the kernel could copy. The entry struct has no
        # family for one, and one ineligible entry sends the whole spec to the
        # Python draw.
        return None

    return shock.family


def is_native_eligible(
    plan: ShockPlan,
) -> bool:
    """Family codes for a spec the native draw can take, else None.

    Eligibility is all-or-nothing: one entry the kernel cannot draw sends the
    whole spec back to the Python route, since a simulation step reads a single
    shock block.
    """
    return all(_spec_family(entry) is not None for entry in plan.entries)


def get_native_shock_plan(
    plan: ShockPlan,
    T: int,
    n_exog: int,
    shock_scale: float = 1.0,
) -> NativeShockPlan | None:
    if not is_native_eligible(plan):
        return None
    return native_shock_plan(plan, T, n_exog, shock_scale)
