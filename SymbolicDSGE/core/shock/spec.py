"""Resolving a shock spec against a compiled model.

Nothing here reads the policy: target columns, the canonical order of a grouped
entry, the standard deviations and correlations off the calibration, the
assembled covariance and its factor all depend on the model and the spec alone.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Any, cast
from hashlib import blake2b

import numpy as np
from numpy import asarray, float64, ndarray, generic
from numpy.linalg import cholesky, LinAlgError, eigh
from numpy.typing import NDArray

from ..compiled_model import CompiledModel
from ..config import make_Q

from .plan import get_native_shock_plan, native_code
from .table import DistributionTable
from .generators import (
    Shock,
    ShockDistribution,
    ShockParameters,
    ShockPath,
    ShockPathParameters,
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


def _resolve_loc(kwargs: Mapping[str, Any], width: int) -> ndarray:
    """The location one spec declares, as a ``width``-long vector.

    The split draw paths read two names for one parameter, ``mean`` on the
    multivariate families and ``loc`` on the univariate ones, because each
    mirrored the scipy call it wrapped. One expression now covers both widths, so
    both spellings resolve here and ``mean`` wins when a spec carries both.

    The resolver calls this once per entry and the result travels on the entry,
    which is what keeps the Python draw and the native lowering from reading the
    same keyword arguments under two different rules.

    Any shape holding one value per variable is taken, since a location read out
    of a matrix calculation arrives as a row or a column as readily as flat.
    """
    name = "mean" if "mean" in kwargs else "loc"
    loc = asarray(kwargs.get(name, np.zeros(width)), dtype=float64)
    if loc.size != width:
        raise ValueError(
            f"`{name}` needs one value per variable in the entry. "
            f"Expected {width}, got {loc.size}. "
            "Omit the keyword for a zero default."
        )
    return loc.reshape(width)


def _gaussian_factor(cov: ndarray) -> ndarray:
    """A factor ``F`` with ``F @ F.T == cov``.

    Cholesky on the common positive-definite path, with an eigh-based fallback so
    positive-semidefinite (but not strictly PD) covariances still sample -- the
    robustness scipy's ``_PSD`` gives us, without scipy.
    """
    try:
        return cholesky(cov)
    except LinAlgError:
        w, V = eigh(cov)
        return cast(ndarray, V * np.sqrt(np.clip(w, 0.0, None)))


def resolve_shock_plan(
    compiled: CompiledModel,
    shocks: ShockSpec,
    T: int | None = None,
) -> ShockPlan:
    """Resolve a shock spec against a model into a reusable plan.

    The plan collects distributional info from shocks and model-dependent info
    from the ``CompiledModel`` to create a container + callable that materializes
    shock matrices.

    ``T`` is required only for an entry the kernel cannot draw, which resolves
    its family against a horizon to be drawn in Python.
    """
    calib = compiled.config.calibration
    shock_col = compiled.shock_idx

    spec = _normalized_spec(shocks)
    validate_shock_targets(spec, list(compiled.shock_names))

    entries: list[ShockEntry | ArrayEntry] = []
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
        family = native_code(shock)
        entries.append(
            ShockEntry(
                key=key,
                indices=indices,
                family=family,
                table=shock.dist if isinstance(shock.dist, DistributionTable) else None,
                loc=_resolve_loc(shock.dist_kwargs, len(indices)),
                factor=factor,
                base_seed=None if shock.seed is None else int(shock.seed),
                kwargs=dict(shock.dist_kwargs),
            )
        )

    return ShockPlan(
        entries=tuple(entries),
        n_exog=compiled.n_exog,
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
    plan = resolve_shock_plan(compiled, shocks, T)
    return get_native_shock_plan(plan, T, shock_scale).draw(0)


# --- Serialization/Deserialization ---


def _jsonable(value: Any) -> Any:
    """Coerce shock arg/kwarg values into JSON-serializable form."""
    if isinstance(value, ndarray):
        return value.tolist()
    if isinstance(value, generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _serialize_path(shock_path: ShockPath) -> tuple[NDF, ShockPathParameters]:
    path = shock_path.path.reshape(-1)
    path_id = blake2b(path.tobytes(), digest_size=8).hexdigest()
    return path, ShockPathParameters(target=shock_path.target, path_id=path_id)


def _serialize_table(shock: Shock) -> tuple[NDF, ShockParameters]:
    table = shock.dist if isinstance(shock.dist, DistributionTable) else None
    if table is None:
        raise TypeError(
            "Expected a tabulated distribution to serialize, but got "
            f"{type(shock.dist).__name__}."
        )

    return table.q, ShockParameters(
        target=shock.target,
        dist=None,
        table=table.to_params(),
        seed=shock.seed,
        dist_kwargs={str(k): _jsonable(v) for k, v in shock.dist_kwargs.items()},
    )


def serialize_shocks(
    shocks: Sequence[Shock | ShockPath],
) -> tuple[
    Sequence[ShockParameters | ShockPathParameters],
    Mapping[str, NDF],
    Mapping[str, NDF],
]:
    """Convert a shock specification to JSON-serializable parameters.

    Arrays ride separately through parquet/CSV bundle members.

    Parameters
    ----------
    shocks : Sequence[Shock | ShockPath]
        Normalized shock specification.

    Returns
    -------
    tuple[
        Sequence[ShockParameters | ShockPathParameters],
        Mapping[str, NDF],
        Mapping[str, NDF],
    ]
        {list(parameters), dict(path_id -> path), dict(table_id -> table)}.
    """
    specs: list[ShockParameters | ShockPathParameters] = []
    paths: dict[str, NDF] = {}
    tables: dict[str, NDF] = {}

    spec: ShockParameters | ShockPathParameters
    for shock in shocks:
        if isinstance(shock, ShockPath):
            path, spec = _serialize_path(shock)
            specs.append(spec)
            paths[spec["path_id"]] = path
            continue
        if isinstance(shock.dist, DistributionTable):
            table, spec = _serialize_table(shock)
            specs.append(spec)
            if spec["table"] is not None:
                tables[spec["table"]["table_id"]] = table
            continue

        specs.append(
            ShockParameters(
                target=shock.target,
                dist=shock.dist,  # pyright: ignore
                table=None,
                seed=int(shock.seed) if shock.seed is not None else None,
                dist_kwargs={
                    str(k): _jsonable(v) for k, v in shock.dist_kwargs.items()
                },
            )
        )
    return specs, paths, tables


def _deserialize_path(path: NDF, params: ShockPathParameters) -> ShockPath:
    return ShockPath(path, *params["target"])


def _deserialize_table(table: NDF, params: ShockParameters) -> DistributionTable:
    if params["table"] is None:
        raise ValueError(
            "Cannot deserialize a tabulated distribution without table parameters."
        )
    return DistributionTable(
        table, params["table"]["zmax"], params["table"]["discrete"]
    )


def deserialize_shocks(
    specs: Sequence[ShockParameters | ShockPathParameters],
    paths: Mapping[str, NDF],
    tables: Mapping[str, NDF],
) -> list[Shock | ShockPath]:
    """Deserialize an incoming bundled shock specification.

    Parameters
    ----------
    specs : Sequence[ShockParameters | ShockPathParameters]
        Parameter spec from the JSON manifest.
    paths : Mapping[str, NDF]
        Arrays feeding `ShockPath` entries.
    tables : Mapping[str, NDF]
        Arrays feeding `DistributionTable` entries that approximate
        a custom distribution.

    Returns
    -------
    list[Shock | ShockPath]
        Normalized shock specification with bound entries,
        ready to pass into any shock-accepting library function.

    """
    shocks: list[Shock | ShockPath] = []
    for spec in specs:
        if "path_id" in spec:
            spec = cast(ShockPathParameters, spec)
            path_id = spec["path_id"]
            if path_id not in paths:
                raise ValueError(
                    f"Missing shock path for path_id {path_id!r}. "
                    f"Available paths: {list(paths.keys())}."
                )

            # Two paths with the same data but different shapes hash identically
            # to save disk space. A row-count permitting (-1, n_targets) reshape
            # restores the original shape. Bytes of the arrays are interpreted as
            # C-contiguous, therefore column order does change the hash, a reshaped
            # array cannot have swapped columns.
            p = len(spec["target"])
            arr = paths[path_id].copy().reshape(-1, p)

            shocks.append(_deserialize_path(arr, spec))

        else:
            dist: ShockDistribution | DistributionTable | None
            if "table" in spec and (tb := spec["table"]) is not None:
                table_id = tb["table_id"]
                if table_id not in tables:
                    raise ValueError(
                        f"Missing shock table for table_id {table_id!r}. "
                        f"Available tables: {list(tables.keys())}."
                    )
                dist = _deserialize_table(tables[table_id], spec)
            else:
                dist = spec["dist"]

            if dist is None:
                raise ValueError(
                    "Cannot deserialize a shock without a distribution or table."
                )

            shocks.append(
                Shock(
                    dist,
                    seed=spec["seed"],
                    **spec["dist_kwargs"],
                ).joint(*spec["target"])
            )
    return shocks
