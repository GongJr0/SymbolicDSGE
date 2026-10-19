"""The JSON boundary for the Outputs tab's simulation spec.

Two conversions, one per direction, between the wire's self-contained
:class:`SimSpecWire` and the live ``sim`` keywords a run is made of. A shock
spec is the only field needing more than a coercion: it crosses with its arrays
inline, since the session holds no pool to reference them in.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .schemas import SimSpecWire
from .shock import deserialize_shocks_inline, serialize_shocks_inline
from ..bundle.loader import SimulationPrefill


def sim_run_kwargs(posted: Mapping[str, Any]) -> SimulationPrefill:
    """A posted simulation spec as the keywords ``sim`` takes.

    The fields are :class:`~SymbolicDSGE.bundle.loader.SimulationPrefill`'s, the
    same set the loader hands over for a stored run, but left untyped: that
    carrier's ``x0`` does not satisfy ``sim``'s own annotation for the parameter
    it feeds, and reconciling the two is a change to ``sim``.

    Read field by field rather than cast, so a key the client invented has
    nowhere to land. The shock entries are lifted to live objects here: nothing
    downstream of this takes a spec as data.
    """
    shocks = posted.get("shocks")
    return SimulationPrefill(
        T=int(posted.get("T", 0)),
        x0=posted.get("x0"),
        observables=bool(posted.get("observables", False)),
        shock_scale=float(posted.get("shock_scale", 1.0)),
        shocks=deserialize_shocks_inline(shocks) if shocks else None,
    )


def sim_spec_wire(kwargs: SimulationPrefill) -> SimSpecWire:
    """The keywords a run was made with, as the slot the tab repaints from.

    Emitted off the keywords rather than off the body that described them, so
    the stored spec is what actually ran.
    """
    shocks = kwargs["shocks"]
    return SimSpecWire(
        T=int(kwargs["T"]),
        x0=_x0_wire(kwargs["x0"]),
        observables=bool(kwargs["observables"]),
        shock_scale=float(kwargs["shock_scale"]),
        shocks=serialize_shocks_inline(shocks) if shocks else None,
    )


def _x0_wire(x0: Any) -> dict[str, float] | list[float] | None:
    """``x0`` as JSON, keeping the name-keyed and positional shapes apart.

    Every route into a session already carries it as JSON, so this only has to
    stay total for a prefill assembled in Python, where it may be an ndarray.
    """
    if x0 is None:
        return None
    if isinstance(x0, Mapping):
        return {str(name): float(value) for name, value in x0.items()}
    return [float(value) for value in x0]
