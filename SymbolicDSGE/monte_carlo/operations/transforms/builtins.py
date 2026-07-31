from __future__ import annotations

from typing import Any, Callable

from ...custom_op import NumbaCustomFunc
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
    func: Callable[..., Any] | NumbaCustomFunc,
    *,
    source: str,
    field: str,
    output_shape: tuple[int, int],
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
) -> MCStep:
    """Build a native custom transform from a two-array Python function.

    ``func`` receives C-contiguous two-dimensional ``sample`` and ``output``
    arrays, fills ``output``, and returns a status integer. A plain function is
    wrapped in :class:`~SymbolicDSGE.monte_carlo.custom_op.NumbaCustomFunc`.
    ``output_shape`` declares the exact ``(n_rows, n_columns)`` result shape.

    The selected source is copied into the input array once per replication.
    The output is exposed as this step's ``payload`` field.

    Example:
        >>> transform_step(
        ...     "z",
        ...     my_op,
        ...     source="datagen",
        ...     field="observables",
        ...     output_shape=(100, 2),
        ... )
    """
    n_out, p_out = output_shape
    if n_out < 0 or p_out < 0:
        raise ValueError("output_shape dimensions must be non-negative.")
    wrapped = func if isinstance(func, NumbaCustomFunc) else NumbaCustomFunc(func)
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=wrapped,
        kwargs={"output_shape": (n_out, p_out)},
        source_args=(
            _compile_source_args(
                arg="sample",
                source=source,
                field=field,
                columns=columns,
                burn_in=burn_in,
                drop_initial=drop_initial,
            ),
        ),
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
    ddof: int = 0,
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
        step_kwargs={"ddof": ddof},
    )


def log_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    offset: float = 0.0,
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
        step_kwargs={"offset": offset},
    )


def log_diff_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    offset: float = 0.0,
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
        step_kwargs={"offset": offset},
    )


def diff_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    order: int = 1,
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
        step_kwargs={"order": order},
    )


def rolling_mean_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    window: int = 10,
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
        step_kwargs={"window": window},
    )


def rolling_std_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    window: int = 10,
    ddof: int = 0,
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
        step_kwargs={"window": window, "ddof": ddof},
    )


def rolling_var_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    window: int = 10,
    ddof: int = 0,
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
        step_kwargs={"window": window, "ddof": ddof},
    )
