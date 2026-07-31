from __future__ import annotations

from SymbolicDSGE._ckernels.monte_carlo import _transforms as native

from SymbolicDSGE.core.solved_model import SolvedModel
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
    return native.standardize_ax0(sample, ddof)


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
    return native.log_transform(sample, offset)


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
    return native.log_diff_transform(sample, offset)


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
    return native.diff_transform(sample, order)


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
    return native.rolling_mean(sample, window)


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
    return native.rolling_std(sample, window, ddof)


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
    return native.rolling_var(sample, window, ddof)
