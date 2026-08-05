"""Resolved shock specifications, separated from the draws they produce.

Turning a ``{name: Shock | callable | ndarray}`` mapping into a shock matrix has
two halves. One half depends only on the model and the spec: which exogenous
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
from typing import Callable, Sequence

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from .shock_generators import Shock, ShockDrawFn

NDF = NDArray[float64]


@dataclass(frozen=True)
class ShockPlanEntry:
    """One resolved entry of a shock spec.

    Exactly one of ``draw``, ``func``, or ``values`` is set. ``draw`` is a
    family-resolved :class:`Shock` whose seed varies per call, ``func`` is a
    user-supplied callable with its own seed already baked in, and ``values`` is
    a literal array. ``scale`` is the standard deviation for a univariate entry
    and the covariance (or shape) matrix for a multivariate one. ``factor``
    holds its precomputed Cholesky factor when the entry can accept one.

    ``spec`` keeps the originating :class:`Shock` alongside its resolved draw.
    The draw closure has already absorbed the family and its keyword arguments,
    which is all the Python route needs, but the native Monte Carlo lowering has
    to inspect them to decide whether it can reproduce the entry in C, so the
    resolution keeps the spec rather than making that caller re-resolve it.
    """

    key: str
    indices: tuple[int, ...]
    multivar: bool
    scale: float | NDF | None = None
    factor: NDF | None = None
    draw: ShockDrawFn | None = None
    base_seed: int | None = None
    spec: Shock | None = None
    func: Callable[[float | NDF], NDF] | None = None
    values: NDF | None = None

    @property
    def width(self) -> int:
        return len(self.indices)

    def unpack(self, seed_offset: int = 0) -> list[tuple[int, NDF]]:
        """Draw this entry and pair each column with its exogenous index.

        ``seed_offset`` shifts the base seed of a :class:`Shock` entry. The
        fixed-callable and literal-array entries carry no seed to shift and
        ignore it.
        """
        if self.values is not None:
            drawn = self.values
        elif self.draw is not None:
            seed = None if self.base_seed is None else self.base_seed + seed_offset
            drawn = self.draw(self.scale, seed, self.factor)
        elif self.func is not None:
            drawn = self.func(self.scale)  # type: ignore[arg-type]
        else:  # pragma: no cover - construction guarantees one source
            raise ValueError(f"Shock entry {self.key!r} has no draw source.")

        if not self.multivar:
            return [(self.indices[0], np.asarray(drawn, dtype=float64))]

        mat = np.asarray(drawn, dtype=float64)
        if mat.ndim != 2 or mat.shape[1] != self.width:
            raise ValueError(
                f"Shock callable for {self.key} must return array with shape "
                f"(T, {self.width})"
            )
        return list(zip(self.indices, (mat[:, i] for i in range(self.width))))


@dataclass(frozen=True)
class ShockPlan:
    """A shock spec resolved against a model, ready to draw from repeatedly."""

    entries: tuple[ShockPlanEntry, ...]
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
    keys: Sequence[str],
    shock_names: Sequence[str],
) -> None:
    """Check every entry names model shocks, each owned by one entry.

    Runs as one pass over the spec so a shock shared across two grouped keys
    (for example ``"e_g,e_z"`` and ``"e_g,e_r"``) is caught. An exact duplicate
    key cannot reach here because the mapping deduplicates it upstream.
    """
    shock_set = set(shock_names)
    owner: dict[str, str] = {}
    for name in keys:
        members = [n.strip() for n in name.split(",")] if "," in name else [name]
        for member in members:
            if member not in shock_set:
                where = f" in entry {name!r}" if "," in name else ""
                raise ValueError(
                    f"Shock {member!r}{where} is not a model shock. "
                    f"Valid shocks: {list(shock_names)}."
                )
            if member in owner:
                raise ValueError(
                    f"Shock {member!r} is driven by more than one shock entry "
                    f"({owner[member]!r} and {name!r}); each shock may appear "
                    "in at most one entry."
                )
            owner[member] = name


__all__ = [
    "ShockPlan",
    "ShockPlanEntry",
    "validate_shock_targets",
]
