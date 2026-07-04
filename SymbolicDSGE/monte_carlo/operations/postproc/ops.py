"""Post-processing (``OpType.POSTPROC``) op implementations.

Post-loop ops run once after the replication loop, over the assembled
across-replication ``traces`` registry, and return :class:`Summary` / :class:`Raw`
artifacts. The op contract is ``op(*, traces, **kwargs)`` — post-loop ops see only
the assembled traces (and their step kwargs), never the per-rep ``reference`` /
``dgp`` models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ...postproc import Raw, Summary


def run_kde(
    *,
    traces: Mapping[str, NDArray[Any]],
    trace: str,
    bandwidth: str | float = "scott",
    grid_points: int = 200,
    kernel: str = "gaussian",
) -> dict[str, Raw | Summary]:
    """Gaussian kernel density estimate of an across-replication trace.

    Reads ``traces[trace]`` (e.g. ``"test.<name>.statistic"``), estimates its
    density on a uniform grid spanning the finite data range, and emits two
    artifacts: ``"curve"`` — the raw ``(x, density)`` ``N x 2`` array callers need
    for plotting — and ``"descriptives"`` — a small :class:`Summary` table of the
    trace's moments/quantiles. Keyed under the step name (``"<step>.curve"`` /
    ``"<step>.descriptives"``).
    """
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
    # scipy's stub over-narrows both args (Literal bandwidth, dataset dtype);
    # the runtime accepts our str|float bandwidth and float64 data fine.
    estimator = gaussian_kde(cast(Any, data), bw_method=cast(Any, bandwidth))
    grid = np.linspace(float(data.min()), float(data.max()), int(grid_points))
    density = np.asarray(estimator(grid), dtype=np.float64)
    return {
        "curve": Raw(value=np.column_stack([grid, density])),
        "descriptives": Summary(
            value=_describe(data, trace), title=f"{trace} descriptives", render="table"
        ),
    }


def _describe(data: NDArray[Any], trace: str) -> Any:
    """A tidy ``(statistic, value)`` DataFrame of a trace's descriptive moments."""
    import pandas as pd

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
    return pd.DataFrame({"statistic": list(stats), "value": list(stats.values())})
