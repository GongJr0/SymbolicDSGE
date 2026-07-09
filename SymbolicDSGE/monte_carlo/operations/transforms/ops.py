from __future__ import annotations

import numpy as np

from ....core.solved_model import SolvedModel
from ...mc_constructs import MCContext
from ..types import NDF

# Built-in transforms receive their selected input from the MC executor.


def run_standardize(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    sample: NDF,
    ddof: int = 0,
) -> NDF:
    """Per-column z-score: ``(x - mean) / std`` over each column.

    ``ddof`` selects sample (1) vs population (0) standard deviation. Columns
    whose ``std`` is zero are returned as zeros to avoid division-by-zero
    blowing up an entire MC replication.
    """
    del context, reference, dgp, rep_idx
    arr = sample
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, ddof=ddof, keepdims=True)
    safe_std = np.where(std == 0.0, 1.0, std)
    out = (arr - mean) / safe_std
    out = np.where(std == 0.0, 0.0, out)
    return np.ascontiguousarray(out, dtype=np.float64)


def run_log(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    sample: NDF,
    offset: float = 0.0,
) -> NDF:
    """``log(x + offset)`` per element. ``offset`` lets users handle zeros."""
    del context, reference, dgp, rep_idx
    arr = sample
    return np.ascontiguousarray(np.log(arr + float(offset)), dtype=np.float64)


def run_log_diff(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    sample: NDF,
    offset: float = 0.0,
) -> NDF:
    """One-period log differences along the time axis.

    Output has one fewer row than the input; ``offset`` is added before the log
    to handle inputs that touch zero.
    """
    del context, reference, dgp, rep_idx
    arr = sample
    logged = np.log(arr + float(offset))
    return np.ascontiguousarray(np.diff(logged, axis=0), dtype=np.float64)


def run_diff(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    sample: NDF,
    order: int = 1,
) -> NDF:
    """``np.diff`` along the time axis, repeated ``order`` times."""
    del context, reference, dgp, rep_idx
    if order < 1:
        raise ValueError("diff order must be at least 1.")
    arr = sample
    return np.ascontiguousarray(np.diff(arr, n=int(order), axis=0), dtype=np.float64)


def _rolling_window_view(arr: NDF, window: int) -> NDF:
    if window < 1:
        raise ValueError("rolling window must be at least 1.")
    if window > arr.shape[0]:
        raise ValueError(
            f"rolling window ({window}) exceeds input length ({arr.shape[0]})."
        )
    # ``sliding_window_view`` -> (n - w + 1, w, k); axis=0 over the time axis.
    return np.lib.stride_tricks.sliding_window_view(arr, window, axis=0)


def run_rolling_mean(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    sample: NDF,
    window: int = 10,
) -> NDF:
    """Centered-window-less trailing rolling mean over the time axis.

    Output shape is ``(n - window + 1, k)``. Each row is the average over the
    preceding ``window`` periods (inclusive of the current row).
    """
    del context, reference, dgp, rep_idx
    arr = sample
    view = _rolling_window_view(arr, int(window))
    return np.ascontiguousarray(view.mean(axis=-1), dtype=np.float64)


def run_rolling_std(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    sample: NDF,
    window: int = 10,
    ddof: int = 0,
) -> NDF:
    """Trailing rolling standard deviation over the time axis."""
    del context, reference, dgp, rep_idx
    arr = sample
    view = _rolling_window_view(arr, int(window))
    return np.ascontiguousarray(view.std(axis=-1, ddof=int(ddof)), dtype=np.float64)


def run_rolling_var(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    sample: NDF,
    window: int = 10,
    ddof: int = 0,
) -> NDF:
    """Trailing rolling variance over the time axis."""
    del context, reference, dgp, rep_idx
    arr = sample
    view = _rolling_window_view(arr, int(window))
    return np.ascontiguousarray(view.var(axis=-1, ddof=int(ddof)), dtype=np.float64)
