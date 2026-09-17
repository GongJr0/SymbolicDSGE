"""Resolving a shock spec against a compiled model.

Nothing here reads the policy: target columns, the canonical order of a grouped
entry, the standard deviations and correlations off the calibration, the
assembled covariance and its factor all depend on the model and the spec alone.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Any

import numpy as np
from numpy import asarray, float64, ndarray
from numpy.typing import NDArray

from ..compiled_model import CompiledModel
from ..config import make_Q
from .generators import (
    Shock,
    ShockParameters,
    ShockPath,
    ShockPathParameters,
    _gaussian_factor,
    resolve_loc,
)
from .plan import (
    ArrayEntry,
    ShockEntry,
    ShockPlan,
    validate_shock_targets,
)

NDF = NDArray[float64]

#: A shock spec in either input shape, before :func:`_normalized_spec`.
#:
#: A mapping keys each entry from the outside, which is why its values are the
#: unbound forms: a :class:`Shock` the key binds, or a bare array the key names.
#: A :class:`ShockPath` is bound at construction and carries its own target, so
#: a key over one can only restate it or contradict it; it belongs to the
#: sequence shape, where every entry names itself.
ShockSpec = Mapping[str | Sequence[str], Shock | NDF] | Sequence[Shock | ShockPath]


def _normalized_spec(
    shocks: Any,
) -> Sequence[Shock | ShockPath]:
    """A shock spec in either authored shape, as the sequence of bound entries.

    The shapes differ in where an entry's targets come from. A mapping keys them
    from the outside, so its values are the unbound forms: a :class:`Shock` the
    key binds, or a bare array the key names. A sequence carries entries that
    name themselves, which is why a :class:`ShockPath` belongs only there.

    The names themselves are not checked here. Whether a target is a model shock
    is :func:`validate_shock_targets`, which runs where a model is in hand.
    """
    if shocks is None:
        return []
    normalized: list[Shock | ShockPath] = []
    if isinstance(shocks, Mapping):
        for k, s in shocks.items():
            keys = (k,) if isinstance(k, str) else tuple(k)
            if not isinstance(s, (Shock, ndarray)):
                raise TypeError(
                    f"Shock spec entry for {keys!r} must be a Shock to draw from "
                    f"or an array holding its path; got {type(s).__name__}."
                )
            normalized.append(
                s.joint(*keys) if isinstance(s, Shock) else ShockPath(s, *keys)
            )

    elif isinstance(shocks, Sequence):
        for s in shocks:
            if not isinstance(s, (Shock, ShockPath)):
                raise TypeError(
                    f"Shock spec entries must be a bound Shock or a ShockPath; "
                    f"got {type(s).__name__}."
                )
            if isinstance(s, Shock) and not s.is_bound:
                raise ValueError(
                    "Unbound Shock in sequence. "
                    "Call .joint(*keys) to bind a single instance "
                    "or .independent(*keys) to bind independent shocks per key."
                )
            normalized.append(s)
    else:
        raise TypeError(
            f"Shock spec must be a mapping or sequence; got {type(shocks).__name__}."
        )
    return normalized


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

    Sorting here is what makes a grouped key's spelling irrelevant to a drawn
    entry: ``"e_g,e_z"`` and ``"e_z,e_g"`` resolve to the same columns, in the
    order the covariance block and its factor are built in. Nothing downstream
    re-sorts. A supplied path has no such freedom, since its own columns say
    which shock each one drives; :func:`_array_entry` sorts that case itself.
    """
    return tuple(sorted(shock_col[name] for name in key))


def _array_entry(shock: ShockPath, shock_col: Mapping[str, int]) -> ArrayEntry:
    """A literal path, widened to ``(T, width)`` and put into column order.

    The key order is what says which column of the path drives which shock, and
    it is the user's to keep consistent. Entries are filed in ascending column
    order, so the permutation that sorts the key has to move the path's columns
    with it: ``ShockPath(arr, "e_v", "e_u")`` means ``arr[:, 0]`` is ``e_v``
    whichever order the model lists the two in. Sorting the key alone would hand
    each column to the other shock.
    """
    key = shock.target
    order = sorted(range(len(key)), key=lambda i: shock_col[key[i]])

    values = asarray(shock.path, dtype=float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] != len(key):
        raise ValueError(
            f"Shock array for {key!r} must have shape (T, {len(key)}); "
            f"got {tuple(values.shape)}."
        )
    # Fancy indexing copies, and a key already written in column order is the
    # common case, so only permute when the order actually differs.
    if order != list(range(len(order))):
        values = values[:, order]

    return ArrayEntry(
        key=tuple(key[i] for i in order),
        indices=tuple(shock_col[key[i]] for i in order),
        value=values,
    )


def resolve_shock_plan(
    compiled: CompiledModel,
    shocks: ShockSpec,
    T: int | None = None,
) -> ShockPlan:
    """Resolve a shock spec against a model into a reusable plan.

    The plan collects distributional info from shocks and model-dependent info
    from the ``CompiledModel`` to create a container + callable that materializes
    shock matrices.

    ``T`` is required only when the spec carries live :class:`Shock` entries,
    which resolve their distribution family against a horizon.
    """
    calib = compiled.config.calibration
    shock_col = compiled.shock_idx

    spec = _normalized_spec(shocks)
    validate_shock_targets(spec, list(compiled.shock_names))

    entries: list[ShockEntry | ArrayEntry] = []
    seeded_count = 0
    cov: NDF | None = None

    for shock in spec:
        key = shock.target

        if isinstance(shock, ShockPath):
            entries.append(_array_entry(shock, shock_col))
            continue

        if not isinstance(shock, Shock):
            raise TypeError(
                f"Shock for {key!r} must be a Shock or ShockPath; got "
                f"{type(shock).__name__}."
            )

        indices = _columns(key, shock_col)

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


def shock_from_json(
    entry: ShockParameters | ShockPathParameters,
) -> Shock | ShockPath:
    if "path" in entry:
        return ShockPath.from_dict(entry)
    else:
        return Shock.from_dict(entry)
