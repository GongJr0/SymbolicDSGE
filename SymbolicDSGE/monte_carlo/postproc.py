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
from typing import Any, Literal, TypedDict, cast

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Summary:
    value: Any


@dataclass(frozen=True)
class Raw:
    value: NDArray[Any]


class Artifact(TypedDict):
    raw: Raw | None
    summary: Summary | None


def normalize_artifacts(inp: Any) -> Artifact:
    out = Artifact(raw=None, summary=None)

    if isinstance(inp, Summary):
        out["summary"] = inp
        return out
    elif isinstance(inp, Raw):
        out["raw"] = inp
        return out

    if not isinstance(inp, tuple):
        raise TypeError(
            "POSTPROC op must return a Summary, Raw, or a tuple of them, not "
            f"{type(inp).__name__}"
        )

    for e in inp:
        if isinstance(e, Summary):
            if out["summary"] is not None:
                raise ValueError("POSTPROC op returned multiple Summary artifacts")
            out["summary"] = e
        elif isinstance(e, Raw):
            if out["raw"] is not None:
                raise ValueError("POSTPROC op returned multiple Raw artifacts")
            out["raw"] = e
        else:
            raise TypeError(
                "POSTPROC op must return a Summary, Raw, or a tuple of them, not "
                f"{type(e).__name__}"
            )
    return out


def run_kde(
    *,
    traces: Mapping[str, NDArray[Any]],
    trace: str,
    bandwidth: str | float = "scott",
    grid_points: int = 200,
    kernel: str = "gaussian",
) -> tuple[Raw, Summary]:
    """Estimate a Gaussian KDE for one retained across-replication trace."""
    from scipy.stats import gaussian_kde

    if kernel != "gaussian":
        raise ValueError(
            f"KDE currently supports only the Gaussian kernel, got {kernel!r}."
        )
    if trace not in traces:
        raise KeyError(f"KDE trace {trace!r} is not available in the run's traces.")
    data = np.asarray(traces[trace], dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size < 2:
        raise ValueError(
            f"KDE needs at least two finite values in trace {trace!r}, got {data.size}."
        )
    estimator = gaussian_kde(cast(Any, data), bw_method=cast(Any, bandwidth))
    grid = np.linspace(float(data.min()), float(data.max()), int(grid_points))
    density = np.asarray(estimator(grid), dtype=np.float64)
    stats = {
        "count": float(data.size),
        "mean": float(data.mean()),
        "std": float(data.std(ddof=1)) if data.size > 1 else float("nan"),
        "min": float(data.min()),
        "q25": float(np.quantile(data, 0.25)),
        "median": float(np.median(data)),
        "q75": float(np.quantile(data, 0.75)),
        "max": float(data.max()),
    }
    import pandas as pd

    return (
        Raw(value=np.column_stack([grid, density])),
        Summary(
            value=pd.DataFrame(
                {"statistic": list(stats), "value": list(stats.values())}
            )
        ),
    )
