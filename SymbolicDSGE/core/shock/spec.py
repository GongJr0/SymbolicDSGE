"""Resolving a shock spec against a compiled model.

Nothing here reads the policy: target columns, the canonical order of a grouped
entry, the standard deviations and correlations off the calibration, the
assembled covariance and its factor all depend on the model and the spec alone.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple, Union, Any

import numpy as np
from numpy import asarray, float64, ndarray
from numpy.typing import NDArray

from ..compiled_model import CompiledModel
from ..config import make_Q
from .generators import Shock, _gaussian_factor, resolve_loc
from .plan import (
    ArrayEntry,
    ShockEntry,
    ShockPlan,
    validate_shock_targets,
)

NDF = NDArray[float64]

ShockSpec = Mapping[str | Sequence[str], Shock | NDF]


def _normalized_spec(
    shocks: Any,
) -> dict[tuple[str, ...], Shock | NDF]:
    """Normalize a shock spec to tuple keys, for internal use."""
    if shocks is None:
        return {}

    if not isinstance(shocks, Mapping):
        raise TypeError(f"Shock spec must be a mapping; got {type(shocks).__name__}.")
    if not all(isinstance(k, Sequence) for k in shocks.keys()):  # str is Sequence[str].
        raise TypeError(
            "Shock spec keys must be str or Sequence[str]; got "
            f"{[type(k).__name__ for k in shocks.keys()]}."
        )
    if not all(isinstance(v, (Shock, ndarray)) for v in shocks.values()):
        raise TypeError(
            "Shock spec values must be Shock or ndarray; got "
            f"{[type(v).__name__ for v in shocks.values()]}."
        )

    return {(k,) if isinstance(k, str) else tuple(k): v for k, v in shocks.items()}


def _require_horizon(T: int | None, key: tuple[str, ...]) -> int:
    """A live ``Shock`` resolves its family against a horizon; demand one."""
    if T is None:
        raise ValueError(
            f"Shock spec {','.join(key)!r} is a live Shock; resolving it needs a horizon "
            "T. Pass T, or draw the path yourself and pass the array."
        )
    if not isinstance(T, int) or T < 1:
        raise ValueError(f"T must be a positive integer; got {T!r}.")
    return T


def _columns(key: tuple[str, ...], shock_col: Mapping[str, int]) -> tuple[int, ...]:
    """The exogenous columns one spec key targets, in column order.

    Sorting here is what makes a grouped key's spelling irrelevant: ``"e_g,e_z"``
    and ``"e_z,e_g"`` resolve to the same columns, in the order the covariance
    block and its factor are built in. Nothing downstream re-sorts.
    """
    return tuple(sorted(shock_col[name] for name in key))


def _array_entry(
    key: tuple[str, ...], indices: tuple[int, ...], shock: ndarray
) -> ArrayEntry:
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
    shocks: Mapping[tuple[str, ...], Shock | NDF],
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

    validate_shock_targets(list(shocks.keys()), list(compiled.shock_names))

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
            # A 1x1 factor is the Cholesky of the 1x1 covariance, so the scalar
            # standard deviation needs no square root and every consumer sees
            # one shape.
            factor = np.array(
                [[calib.parameters[calib.shock_std[key[0]]]]], dtype=float64
            )
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
                draw=shock.draw_fn(_require_horizon(T, key), len(indices) > 1),
                loc=resolve_loc(shock.dist_kwargs, len(indices)),
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


def simulation_shock_matrix(
    compiled: CompiledModel,
    T: int,
    shocks: ShockSpec | None = None,
    shock_scale: float = 1.0,
) -> NDF:
    """``(T, n_exog)`` innovations for a spec, or zeros when there is none."""
    if shocks is None:
        return np.zeros((T, compiled.n_exog), dtype=float64)
    return resolve_shock_plan(
        compiled,
        _normalized_spec(shocks),
        T,
    ).matrix(T, shock_scale)


def shock_entry_to_json(key: tuple[str, ...], shock: Shock | NDF) -> dict[str, Any]:
    """One spec entry as a self-describing JSON object.

    A key names one or more shocks, which a JSON object cannot be keyed by, so
    an entry carries its own ``key`` and a spec travels as a list of entries. A
    :class:`Shock` flattens its constructor arguments into the entry and the
    receiver redraws it; a supplied path rides under ``path``. The two are told
    apart by what the entry declares rather than by the shape of a bare value.
    """
    names = [str(name) for name in key]
    if isinstance(shock, Shock):
        return {"key": names, **shock.to_dict()}
    if isinstance(shock, ndarray):
        return {"key": names, "path": shock.tolist()}
    raise TypeError(
        f"Shock {key!r} is a {type(shock).__name__}, which has no serialized "
        f"form. Pass a Shock for the receiver to redraw, or the path itself as "
        f"an array."
    )


def shock_entry_from_json(
    entry: Mapping[str, Any],
) -> tuple[tuple[str, ...], Shock | NDF]:
    """One serialized entry as the ``(key, spec)`` pair a mapping was keyed by."""
    key = tuple(str(name) for name in entry["key"])
    if "path" in entry:
        return key, asarray(entry["path"], dtype=float64)
    return key, Shock.from_dict(entry)
