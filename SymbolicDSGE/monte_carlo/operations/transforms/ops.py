from __future__ import annotations

from typing import Sequence
import numpy as np

from ....core.solved_model import SolvedModel
from ...mc_constructs import MCContext
from ..types import InpSources, NDF
from ..utils import _resolve_context_array

# --------------------------------------------------------------------------- #
# Built-in transforms.                                                         #
#                                                                              #
# Each transform reads its input via the same ``_resolve_context_array``       #
# pipeline the tests use, applies a pure-numpy transformation, and returns the #
# resulting 2D ``float64`` array. The MC runner stashes the array at the       #
# step's ``output_key`` so downstream nodes consume it via ``source="payload"``#
# + ``payload_key=<this step's name>``. The graph validator auto-binds         #
# ``payload_key`` from the parent edge for these chains.                       #
# --------------------------------------------------------------------------- #


def _read_transform_input(
    *,
    context: MCContext,
    source: InpSources,
    filter_key: str,
    payload_key: str | None,
    columns: Sequence[int] | int | slice | None,
    burn_in: int,
    drop_initial: bool,
) -> NDF:
    col_idx: Sequence[int] | slice | None
    if isinstance(columns, int):
        col_idx = [columns]
    else:
        col_idx = columns
    return _resolve_context_array(
        context,
        source=source,
        filter_key=filter_key,
        payload_key=payload_key,
        columns=col_idx,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )


def run_standardize(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    source: InpSources,
    filter_key: str = "filter",
    payload_key: str | None = None,
    columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    ddof: int = 0,
) -> NDF:
    """Per-column z-score: ``(x - mean) / std`` over each column.

    ``ddof`` selects sample (1) vs population (0) standard deviation. Columns
    whose ``std`` is zero are returned as zeros to avoid division-by-zero
    blowing up an entire MC replication.
    """
    del reference, dgp, rep_idx
    arr = _read_transform_input(
        context=context,
        source=source,
        filter_key=filter_key,
        payload_key=payload_key,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
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
    source: InpSources,
    filter_key: str = "filter",
    payload_key: str | None = None,
    columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    offset: float = 0.0,
) -> NDF:
    """``log(x + offset)`` per element. ``offset`` lets users handle zeros."""
    del reference, dgp, rep_idx
    arr = _read_transform_input(
        context=context,
        source=source,
        filter_key=filter_key,
        payload_key=payload_key,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    return np.ascontiguousarray(np.log(arr + float(offset)), dtype=np.float64)


def run_log_diff(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    source: InpSources,
    filter_key: str = "filter",
    payload_key: str | None = None,
    columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    offset: float = 0.0,
) -> NDF:
    """One-period log differences along the time axis.

    Output has one fewer row than the input; ``offset`` is added before the log
    to handle inputs that touch zero.
    """
    del reference, dgp, rep_idx
    arr = _read_transform_input(
        context=context,
        source=source,
        filter_key=filter_key,
        payload_key=payload_key,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    logged = np.log(arr + float(offset))
    return np.ascontiguousarray(np.diff(logged, axis=0), dtype=np.float64)


def run_diff(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    source: InpSources,
    filter_key: str = "filter",
    payload_key: str | None = None,
    columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    order: int = 1,
) -> NDF:
    """``np.diff`` along the time axis, repeated ``order`` times."""
    del reference, dgp, rep_idx
    if order < 1:
        raise ValueError("diff order must be at least 1.")
    arr = _read_transform_input(
        context=context,
        source=source,
        filter_key=filter_key,
        payload_key=payload_key,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
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
    source: InpSources,
    filter_key: str = "filter",
    payload_key: str | None = None,
    columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    window: int = 10,
) -> NDF:
    """Centered-window-less trailing rolling mean over the time axis.

    Output shape is ``(n - window + 1, k)``. Each row is the average over the
    preceding ``window`` periods (inclusive of the current row).
    """
    del reference, dgp, rep_idx
    arr = _read_transform_input(
        context=context,
        source=source,
        filter_key=filter_key,
        payload_key=payload_key,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    view = _rolling_window_view(arr, int(window))
    return np.ascontiguousarray(view.mean(axis=-1), dtype=np.float64)


def run_rolling_std(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    source: InpSources,
    filter_key: str = "filter",
    payload_key: str | None = None,
    columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    window: int = 10,
    ddof: int = 0,
) -> NDF:
    """Trailing rolling standard deviation over the time axis."""
    del reference, dgp, rep_idx
    arr = _read_transform_input(
        context=context,
        source=source,
        filter_key=filter_key,
        payload_key=payload_key,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    view = _rolling_window_view(arr, int(window))
    return np.ascontiguousarray(view.std(axis=-1, ddof=int(ddof)), dtype=np.float64)


def run_rolling_var(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    source: InpSources,
    filter_key: str = "filter",
    payload_key: str | None = None,
    columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    window: int = 10,
    ddof: int = 0,
) -> NDF:
    """Trailing rolling variance over the time axis."""
    del reference, dgp, rep_idx
    arr = _read_transform_input(
        context=context,
        source=source,
        filter_key=filter_key,
        payload_key=payload_key,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    view = _rolling_window_view(arr, int(window))
    return np.ascontiguousarray(view.var(axis=-1, ddof=int(ddof)), dtype=np.float64)
