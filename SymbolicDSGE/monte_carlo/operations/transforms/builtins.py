from __future__ import annotations

from typing import Any, Callable
from ...mc_constructs import (
    ColumnSelector,
    MCStep,
    OpType,
    _compile_source_args,
)

from .ops import (
    run_standardize,
    run_log,
    run_log_diff,
    run_diff,
    run_rolling_mean,
    run_rolling_std,
    run_rolling_var,
)


def transform_step(
    name: str,
    func: Callable[..., Any],
    *,
    store_key: str | None = None,
    **step_kwargs: Any,
) -> MCStep:
    """Wrap a user callable as a per-replication ``OpType.TRANSFORM`` step.

    Signature: ``transform_step(name, func, *, store_key=None, **step_kwargs)``.

    Op contract: ``func(*, context, reference, dgp, rep_idx, **kwargs)``; all
    four are injected every replication (read this rep's data via
    ``context.require_data()`` / ``context.require_payload(...)``), and any
    ``step_kwargs`` passed here arrive as extra keywords. ``func`` returns a 2-D
    ndarray stored as the step's payload (at ``store_key`` or ``name``).
    Bundling requires ``func`` to be a
    :class:`~SymbolicDSGE.monte_carlo.custom_op.NumpyCustomFunc` (use
    ``@numpy_operation``); the bundle builder auto-wraps it at serialization.

    Example:
        >>> transform_step("z", my_op)
    """
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=func,
        kwargs=step_kwargs,
        store_key=store_key,
        step_type="transform:custom",
    )


def _single_source_transform_step(
    name: str,
    func: Callable[..., Any],
    step_type: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector,
    burn_in: int,
    drop_initial: bool,
    step_kwargs: dict[str, Any],
) -> MCStep:
    source_args = (
        _compile_source_args(
            arg="sample",
            source=source,
            field=field,
            columns=columns,
            burn_in=burn_in,
            drop_initial=drop_initial,
        ),
    )
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=func,
        kwargs=step_kwargs,
        source_args=source_args,
        step_type=step_type,
    )


def standardize_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """Per-column z-score ``(x - mean) / std`` over each column.

    Signature: ``standardize_step(name, *, source, field, columns=None, ddof=0)``.

    ``ddof`` picks population (0) vs sample (1) std; zero-variance columns
    return zeros.

    Example:
        >>> standardize_step("z", source="datagen", field="observables")

    See ``operations.transforms`` for the shared input / selection / output contract.
    """
    return _single_source_transform_step(
        name,
        run_standardize,
        "standardize",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        step_kwargs=step_kwargs,
    )


def log_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """Elementwise natural log ``log(x + offset)`` of the series.

    Signature: ``log_step(name, *, source, field, columns=None, offset=0.0)``.

    ``offset`` is added before the log so inputs that touch zero stay finite.

    Example:
        >>> log_step("lg", source="datagen", field="observables")

    See ``operations.transforms`` for the shared input / selection / output contract.
    """
    return _single_source_transform_step(
        name,
        run_log,
        "log",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        step_kwargs=step_kwargs,
    )


def log_diff_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """One-period log differences along the time axis (log growth rates).

    Signature: ``log_diff_step(name, *, source, field, columns=None, offset=0.0)``.

    Output has one fewer row than the input; ``offset`` is added before the log.

    Example:
        >>> log_diff_step("gr", source="datagen", field="observables")

    See ``operations.transforms`` for the shared input / selection / output contract.
    """
    return _single_source_transform_step(
        name,
        run_log_diff,
        "log_diff",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        step_kwargs=step_kwargs,
    )


def diff_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """Discrete difference along the time axis, applied ``order`` times.

    Signature: ``diff_step(name, *, source, field, columns=None, order=1)``.

    Output loses ``order`` rows; ``order`` must be at least 1.

    Example:
        >>> diff_step("d", source="datagen", field="observables")

    See ``operations.transforms`` for the shared input / selection / output contract.
    """
    return _single_source_transform_step(
        name,
        run_diff,
        "diff",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        step_kwargs=step_kwargs,
    )


def rolling_mean_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """Trailing rolling mean over a fixed ``window`` of the time axis.

    Signature: ``rolling_mean_step(name, *, source, field, columns=None, window=10)``.

    Output shape is ``(n - window + 1, k)``; ``window`` must not exceed the
    series length.

    Example:
        >>> rolling_mean_step("rm", source="datagen", field="observables", window=20)

    See ``operations.transforms`` for the shared input / selection / output contract.
    """
    return _single_source_transform_step(
        name,
        run_rolling_mean,
        "rolling_mean",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        step_kwargs=step_kwargs,
    )


def rolling_std_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """Trailing rolling standard deviation over a fixed ``window``.

    Signature: ``rolling_std_step(name, *, source, field, columns=None, window=10, ddof=0)``.

    Output shape is ``(n - window + 1, k)``; ``ddof`` picks population vs sample.

    Example:
        >>> rolling_std_step("rs", source="datagen", field="observables", window=20)

    See ``operations.transforms`` for the shared input / selection / output contract.
    """
    return _single_source_transform_step(
        name,
        run_rolling_std,
        "rolling_std",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        step_kwargs=step_kwargs,
    )


def rolling_var_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """Trailing rolling variance over a fixed ``window`` of the time axis.

    Signature: ``rolling_var_step(name, *, source, field, columns=None, window=10, ddof=0)``.

    Output shape is ``(n - window + 1, k)``; ``ddof`` picks population vs sample.

    Example:
        >>> rolling_var_step("rv", source="datagen", field="observables", window=20)

    See ``operations.transforms`` for the shared input / selection / output contract.
    """
    return _single_source_transform_step(
        name,
        run_rolling_var,
        "rolling_var",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        step_kwargs=step_kwargs,
    )
