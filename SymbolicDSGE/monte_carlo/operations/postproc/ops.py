"""Post-processing (``OpType.POSTPROC``) op implementations.

Post-loop ops run once after the replication loop, over the assembled
across-replication ``traces`` registry, and return :class:`Summary` / :class:`Raw`
artifacts. The op contract is ``op(*, traces, reference, dgp, **kwargs)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ....core.solved_model import SolvedModel
from ...postproc import Raw


def run_kde(
    *,
    traces: Mapping[str, NDArray[Any]],
    reference: SolvedModel,
    dgp: SolvedModel | None,
    trace: str,
    bandwidth: str | float = "scott",
    grid_points: int = 200,
    kernel: str = "gaussian",
) -> Raw:
    """Gaussian kernel density estimate of an across-replication trace.

    Reads ``traces[trace]`` (e.g. ``"test.<name>.statistic"``), estimates its
    density on a uniform grid spanning the finite data range, and returns the raw
    ``(x, density)`` curve as an ``N x 2`` array — the bulk data callers need for
    plotting. A descriptive :class:`Summary` (moments/quantiles as a table) is
    added once tabular artifacts land (#181).
    """
    from scipy.stats import gaussian_kde

    del reference, dgp
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
    return Raw(value=np.column_stack([grid, density]))
