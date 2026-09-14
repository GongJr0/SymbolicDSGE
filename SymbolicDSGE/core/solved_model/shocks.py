"""Resolving a shock spec against a compiled model.

Nothing here reads the policy: target columns, the canonical order of a grouped
entry, the standard deviations and correlations off the calibration, the
assembled covariance and its factor all depend on the model and the spec alone.
"""

from __future__ import annotations

from typing import Mapping, Tuple, Union

import numpy as np
from numpy import asarray, float64, ndarray
from numpy.typing import NDArray

from ..compiled_model import CompiledModel
from ..config import make_Q
from ..shock_generators import Shock, _gaussian_factor
from ..shock_plan import (
    ArrayEntry,
    ShockEntry,
    ShockPlan,
    validate_shock_targets,
)

NDF = NDArray[float64]

ShockSpec = Mapping[str, Union[Shock, NDF]]


def _require_horizon(T: int | None, name: str) -> int:
    """A live ``Shock`` resolves its family against a horizon; demand one."""
    if T is None:
        raise ValueError(
            f"Shock spec {name!r} is a live Shock; resolving it needs a horizon "
            "T. Pass T, or draw the path yourself and pass the array."
        )
    return T


def _columns(key: str, shock_col: Mapping[str, int]) -> tuple[int, ...]:
    """The exogenous columns one spec key targets, in column order.

    Sorting here is what makes a grouped key's spelling irrelevant: ``"e_g,e_z"``
    and ``"e_z,e_g"`` resolve to the same columns, in the order the covariance
    block and its factor are built in. Nothing downstream re-sorts.
    """
    names = [n.strip() for n in key.split(",")] if "," in key else [key]
    return tuple(sorted(shock_col[name] for name in names))


def _array_entry(key: str, indices: tuple[int, ...], shock: ndarray) -> ArrayEntry:
    """A literal path, widened to ``(T, width)`` so every entry unpacks alike."""
    values = asarray(shock, dtype=float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] != len(indices):
        raise ValueError(
            f"Shock array for {key!r} must have shape (T, {len(indices)}); "
            f"got {tuple(values.shape)}."
        )
    return ArrayEntry(key=key, indices=indices, value=values)


def resolve_shock_plan(
    compiled: CompiledModel,
    shocks: ShockSpec,
    T: int | None = None,
) -> ShockPlan:
    """Resolve a shock spec against a model into a reusable plan.

    The plan holds everything the spec and the model fix between them: which
    exogenous columns each entry targets, the standard deviations and
    correlations read off the calibration, and the covariance factor a joint
    entry draws through. Drawing is separate, so one plan serves many draws.

    The calibration is plan-invariant: the covariance is assembled at most once
    here and each grouped entry indexes its own block out of it. Entry columns are
    positions in ``config.shocks`` (``shock_names`` is built from it, and
    ``shock_idx`` from that), which is the order the covariance is assembled in,
    letting an entry's indices slice it directly. A spec of single shocks reads
    its standard deviations straight off the calibration and never assembles one,
    which is also what keeps correlations out of a spec that declares no groups.

    ``T`` is required only when the mapping carries live :class:`Shock` specs,
    which resolve their distribution family against a horizon.
    """
    calib = compiled.config.calibration
    shock_col = compiled.shock_idx
    validate_shock_targets(list(shocks), list(compiled.shock_names))

    entries: list[ShockEntry | ArrayEntry] = []
    seeded_count = 0
    cov: NDF | None = None

    for key, shock in shocks.items():
        indices = _columns(key, shock_col)

        if isinstance(shock, ndarray):
            entries.append(_array_entry(key, indices, shock))
            continue

        if not isinstance(shock, Shock):
            raise TypeError(
                f"Shock for {key!r} must be a Shock or an ndarray path; got "
                f"{type(shock).__name__}."
            )

        if shock.seed is not None:
            seeded_count += 1

        # A width-1 entry draws against its own standard deviation; a grouped
        # one against its covariance block, which is why only the grouped case
        # assembles one. The assembly reads a calibration no entry can vary:
        # the first group pays for it and the rest index the same matrix.
        if len(indices) == 1:
            scale: float | NDF = calib.parameters[calib.shock_std[key]]
            factor = None
        else:
            if cov is None:
                cov = make_Q(
                    compiled.config.shocks,
                    calib.shock_std,
                    calib.shock_corr,
                    calib.parameters,
                )
            scale = cov[np.ix_(indices, indices)]
            factor = _gaussian_factor(scale)

        entries.append(
            ShockEntry(
                key=key,
                indices=indices,
                scale=scale,
                draw=shock.draw_fn(_require_horizon(T, key), len(indices) > 1),
                factor=factor,
                base_seed=None if shock.seed is None else int(shock.seed),
                kwargs=dict(shock.dist_kwargs),
            )
        )

    return ShockPlan(
        entries=tuple(entries),
        n_exog=compiled.n_exog,
        seeded_count=seeded_count,
    )


def shock_unpack(
    compiled: CompiledModel,
    shocks: ShockSpec,
) -> list[Tuple[int, NDF]]:
    """Resolve a spec and draw it once, as ``(exogenous index, column)``."""
    return resolve_shock_plan(compiled, shocks).unpack()


def simulation_shock_matrix(
    compiled: CompiledModel,
    T: int,
    shocks: ShockSpec | None = None,
    shock_scale: float = 1.0,
) -> NDF:
    """``(T, n_exog)`` innovations for a spec, or zeros when there is none."""
    if shocks is None:
        return np.zeros((T, compiled.n_exog), dtype=float64)
    return resolve_shock_plan(compiled, shocks, T).matrix(T, shock_scale)
