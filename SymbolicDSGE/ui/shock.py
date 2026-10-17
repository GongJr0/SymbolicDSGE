from __future__ import annotations

from typing import Any, Sequence, TypedDict, cast

from numpy import asarray, float64

from ..core.shock import Shock, ShockPath, DistributionTable
from ..core.shock.generators import ShockDistribution
from ..core.shock.spec import _jsonable


class InlineTable(TypedDict):
    """A :class:`DistributionTable`'s knots, carried rather than referenced.

    No ``table_id``: the id is a function of the knots, and a transmitted copy
    of it is one more thing that can disagree with them.
    """

    q: list[float]
    zmax: float
    discrete: bool


class InlineShock(TypedDict):
    """A drawn entry. Exactly one of ``dist`` and ``table`` is set."""

    target: list[str]
    dist: ShockDistribution | None
    table: InlineTable | None
    seed: int | None
    dist_kwargs: dict[str, Any]


class InlinePath(TypedDict):
    """A supplied path, holding its own ``(T, width)`` rows."""

    target: list[str]
    path: list[list[float]]


#: One entry of a shock spec that carries its arrays. The kinds are told apart
#: by ``path``, present on a supplied array and absent on a drawn family, which
#: is the test :func:`_normalized_spec` applies to the live objects too.
InlineShockEntry = InlineShock | InlinePath


def serialize_shocks_inline(
    shocks: Sequence[Shock | ShockPath],
) -> list[InlineShockEntry]:
    """Convert a normalized shock spec to self-contained JSON entries.

    :func:`serialize_shocks` for a transport that has nowhere to pool an array:
    every array rides in the entry that reads it, so a spec crosses alone and
    comes back without a pool to resolve against. That is what an editing client
    needs, since a content id is a function of the bytes an editor exists to
    change.
    """
    entries: list[InlineShockEntry] = []
    for shock in shocks:
        if isinstance(shock, ShockPath):
            entries.append(
                InlinePath(target=list(shock.target), path=shock.path.tolist())
            )
            continue

        dist: ShockDistribution | None
        table: InlineTable | None
        if isinstance(shock.dist, DistributionTable):
            dist = None
            table = InlineTable(
                q=shock.dist.q.tolist(),
                zmax=float(shock.dist.zmax),
                discrete=bool(shock.dist.discrete),
            )
        else:
            dist, table = shock.dist, None  # pyright: ignore

        entries.append(
            InlineShock(
                target=list(shock.target),
                dist=dist,
                table=table,
                seed=None if shock.seed is None else int(shock.seed),
                dist_kwargs={
                    str(k): _jsonable(v) for k, v in shock.dist_kwargs.items()
                },
            )
        )
    return entries


def deserialize_shocks_inline(
    entries: Sequence[InlineShockEntry],
) -> list[Shock | ShockPath]:
    """Rebuild a shock spec from self-contained JSON entries.

    Read field by field, since the entries arrive as untrusted data: a key the
    sender invented has nowhere to land. Everything an entry does declare is
    validated by what it is handed to, so a family name, a knot run, and a path
    width are refused by :class:`Shock`, :class:`DistributionTable`, and
    :class:`ShockPath` respectively rather than checked twice.
    """
    shocks: list[Shock | ShockPath] = []
    for entry in entries:
        target = tuple(str(name) for name in entry["target"])
        if "path" in entry:
            path = asarray(cast(InlinePath, entry)["path"], dtype=float64)
            shocks.append(ShockPath(path, *target))
            continue

        dist: ShockDistribution | DistributionTable | None
        if (table := entry.get("table")) is not None:
            dist = DistributionTable(
                asarray(table["q"], dtype=float64),
                float(table["zmax"]),
                bool(table["discrete"]),
            )
        else:
            dist = entry.get("dist")
        if dist is None:
            raise ValueError(
                f"Shock entry for {','.join(target)!r} names neither a distribution "
                "family nor a table. A supplied array travels as an entry carrying "
                "'path' instead."
            )

        seed = entry.get("seed")
        shocks.append(
            Shock(
                dist,
                None if seed is None else int(seed),
                **dict(entry.get("dist_kwargs") or {}),
            ).joint(*target)
        )
    return shocks
