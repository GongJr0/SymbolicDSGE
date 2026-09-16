"""Resolved shock specifications, separated from the draws they produce.

Turning a ``{name: Shock | ndarray}`` mapping into a shock matrix has two
halves. One half depends only on the model and the spec: which exogenous
columns an entry targets, the canonical order of a grouped (multivariate) entry,
the standard deviations and correlations pulled from the calibration, the
covariance assembled from them, and its factorization. The other half is the
draw itself, which is the only part that varies with the seed.

A :class:`ShockPlan` is the first half, resolved once. Callers that redraw the
same spec under many seeds (the Monte Carlo lowering materializes one path per
replication) resolve a plan and then call :meth:`ShockPlan.fill` per draw, so the
calibration lookups, the covariance assembly, and the Cholesky are paid once
rather than once per replication.
"""

from dataclasses import dataclass
from typing import Sequence
from functools import cached_property

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from .generators import ShockDrawFn

NDF = NDArray[float64]


@dataclass(frozen=True)
class ShockEntry:
    """One drawn entry of a shock spec, resolved against a model.

    ``draw`` is the entry's family resolved for one horizon, and it computes
    ``loc + factor @ v`` over that family's standardized variate. ``factor`` is
    the scale at any width: the 1x1 holding a standard deviation, or the
    covariance block's factor. ``loc`` is the ``width``-long location. Both are
    resolved once here rather than read from the spec at each boundary, which is
    what keeps the Python draw and the native lowering from interpreting one
    spec two ways. ``base_seed`` is the spec's own seed, which :meth:`unpack`
    shifts per draw.

    ``kwargs`` carries what is neither location nor scale, which today is the
    Student-t's ``df``, forward for the native lowering.
    """

    key: tuple[str, ...]
    indices: tuple[int, ...]
    loc: NDF
    factor: NDF
    draw: ShockDrawFn
    base_seed: int | None = None
    kwargs: dict | None = None

    @cached_property
    def width(self) -> int:
        """Number of columns this entry targets."""
        return len(self.indices)

    def unpack(self, seed_offset: int = 0) -> list[tuple[int, NDF]]:
        """Draw this entry and pair each column with its exogenous index.

        ``seed_offset`` shifts the base seed, which is what keeps the
        replications of one Monte Carlo run on different paths. An unseeded
        entry has nothing to shift and redraws freshly.
        """
        seed = None if self.base_seed is None else self.base_seed + seed_offset
        drawn = self.draw(self.loc, self.factor, seed)

        if drawn.ndim != 2 or drawn.shape[1] != self.width:
            raise ValueError(
                f"Draw for {self.key!r} must return shape (T, {self.width}); "
                f"got {tuple(drawn.shape)}."
            )
        return list(zip(self.indices, (drawn[:, i] for i in range(self.width))))


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

    def unpack(self, seed_offset: int = 0) -> list[tuple[int, NDF]]:
        """Pair each column of the supplied path with its exogenous index.

        Takes ``seed_offset`` to match :meth:`ShockEntry.unpack`; a plan then
        unpacks its entries without asking which kind each one is. A supplied
        path has no seed to shift.
        """
        del seed_offset
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
    seeded_count: int

    def unpack(self, seed_offset: int = 0) -> list[tuple[int, NDF]]:
        """Draw every entry as ``(exogenous index, column)`` pairs."""
        out: list[tuple[int, NDF]] = []
        for entry in self.entries:
            out.extend(entry.unpack(seed_offset))
        return out

    def fill(
        self,
        out: NDF,
        T: int,
        shock_scale: float = 1.0,
        seed_offset: int = 0,
    ) -> None:
        """Draw into a preallocated ``(T, n_exog)`` view.

        Writing through a caller-owned view lets the Monte Carlo lowering target
        a row of its ``(n_rep, T, n_exog)`` slab directly, with no per-draw
        temporary. Columns no entry targets are left untouched, so the caller
        owns zeroing.
        """
        for idx, values in self.unpack(seed_offset):
            if values.shape[0] != T:
                raise ValueError(
                    f"Shock array for variable index {idx} must have length {T}."
                )
            out[:, idx] = shock_scale * values

    def matrix(self, T: int, shock_scale: float = 1.0, seed_offset: int = 0) -> NDF:
        """Draw a fresh ``(T, n_exog)`` shock matrix."""
        out = np.zeros((T, self.n_exog), dtype=float64)
        self.fill(out, T, shock_scale, seed_offset)
        return out


def validate_shock_targets(
    keys: Sequence[tuple[str, ...]],
    shock_names: Sequence[str],
) -> None:
    """Check every entry names model shocks, each owned by one entry.

    Runs as one pass over the spec so a shock shared across two grouped keys
    (for example ``"e_g,e_z"`` and ``"e_g,e_r"``) is caught. An exact duplicate
    key cannot reach here because the mapping deduplicates it upstream.
    """
    shock_set = set(shock_names)
    owner: dict[str, str | Sequence[str]] = {}
    for members in keys:
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
