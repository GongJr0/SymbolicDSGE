"""Typed return artifacts for post-loop (``OpType.POSTPROC``) ops.

A POSTPROC op runs **once** after the replication loop, over the assembled
across-replication ``traces`` registry, and returns one or more *tagged*
artifacts that declare how each output is handled downstream:

- :class:`Summary` — a renderable result (scalar / table / small array) that
  belongs in the run's summary surface (its own tab in the GUI);
- :class:`Raw` — bulk numeric data kept as data (a parquet/trace member), not
  auto-rendered.

An op may return a single artifact, a bare value (wrapped by a default policy —
ndarray -> :class:`Raw`, anything else -> :class:`Summary`), or a ``Mapping`` of
named outputs to emit several at once (e.g. a raw indicator array *and* a summary
table from one op). The engine only normalizes and stores them; serialization
and the GUI dispatch on the artifact type.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Union

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Summary:
    """A renderable POSTPROC artifact — gets its own summary surface.

    ``value`` may be a scalar, a small ndarray, a mapping, or a (pandas)
    DataFrame; ``render`` is an optional hint, otherwise inferred from the value.
    """

    value: Any
    title: str | None = None
    render: Literal["auto", "table", "scalar", "array"] = "auto"


@dataclass(frozen=True)
class Raw:
    """A bulk POSTPROC artifact stored as data, not auto-rendered."""

    value: NDArray[Any]


Artifact = Union[Summary, Raw]


def as_artifact(value: Any) -> Artifact:
    """Wrap a bare value: ndarray -> :class:`Raw`, otherwise -> :class:`Summary`."""
    if isinstance(value, (Summary, Raw)):
        return value
    if isinstance(value, np.ndarray):
        return Raw(value=value)
    return Summary(value=value)


def normalize_artifacts(out: Any, step_name: str) -> dict[str, Artifact]:
    """Normalize a POSTPROC op's return into a flat ``{key: artifact}`` map.

    - a single :class:`Summary`/:class:`Raw` -> ``{step_name: artifact}``;
    - a ``Mapping`` of named outputs -> ``{f"{step_name}.{key}": artifact}`` (so
      one op can emit several artifacts without key collisions across steps);
    - a bare value -> wrapped via :func:`as_artifact` under ``step_name``.
    """
    if isinstance(out, (Summary, Raw)):
        return {step_name: out}
    if isinstance(out, Mapping):
        return {f"{step_name}.{key}": as_artifact(value) for key, value in out.items()}
    return {step_name: as_artifact(out)}
