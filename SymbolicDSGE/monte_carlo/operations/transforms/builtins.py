from __future__ import annotations

from typing import Any, Callable
from ...mc_constructs import MCStep, OpType

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
    **kwargs: Any,
) -> MCStep:
    """Wrap a user callable as a per-replication ``OpType.TRANSFORM`` step.

    Signature: ``transform_step(name, func, *, store_key=None, **kwargs)``.

    Any callable runs in-process; ``kwargs`` are forwarded to it each
    replication and the returned array is stored as the step's payload (at
    ``store_key`` or ``name``). Bundling requires ``func`` to be a
    :class:`~SymbolicDSGE.monte_carlo.custom_op.NumpyCustomFunc` (use
    ``@numpy_operation``); the bundle builder auto-wraps it at serialization.

    Example:
        >>> transform_step("z", my_op)
    """
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=func,
        kwargs=kwargs,
        store_key=store_key,
        step_type="transform:custom",
    )


def standardize_step(name: str, **kwargs: Any) -> MCStep:
    """Per-column z-score ``(x - mean) / std`` over each column.

    Signature: ``standardize_step(name, *, source, columns=None, ddof=0)``.

    ``ddof`` picks population (0) vs sample (1) std; zero-variance columns
    return zeros.

    Example:
        >>> standardize_step("z", source="observables")

    See ``operations.transforms`` for the shared input / selection / output contract.
    """
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=run_standardize,
        kwargs=kwargs,
        step_type="standardize",
    )


def log_step(name: str, **kwargs: Any) -> MCStep:
    """Elementwise natural log ``log(x + offset)`` of the series.

    Signature: ``log_step(name, *, source, columns=None, offset=0.0)``.

    ``offset`` is added before the log so inputs that touch zero stay finite.

    Example:
        >>> log_step("lg", source="observables")

    See ``operations.transforms`` for the shared input / selection / output contract.
    """
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=run_log,
        kwargs=kwargs,
        step_type="log",
    )


def log_diff_step(name: str, **kwargs: Any) -> MCStep:
    """One-period log differences along the time axis (log growth rates).

    Signature: ``log_diff_step(name, *, source, columns=None, offset=0.0)``.

    Output has one fewer row than the input; ``offset`` is added before the log.

    Example:
        >>> log_diff_step("gr", source="observables")

    See ``operations.transforms`` for the shared input / selection / output contract.
    """
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=run_log_diff,
        kwargs=kwargs,
        step_type="log_diff",
    )


def diff_step(name: str, **kwargs: Any) -> MCStep:
    """Discrete difference along the time axis, applied ``order`` times.

    Signature: ``diff_step(name, *, source, columns=None, order=1)``.

    Output loses ``order`` rows; ``order`` must be at least 1.

    Example:
        >>> diff_step("d", source="observables")

    See ``operations.transforms`` for the shared input / selection / output contract.
    """
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=run_diff,
        kwargs=kwargs,
        step_type="diff",
    )


def rolling_mean_step(name: str, **kwargs: Any) -> MCStep:
    """Trailing rolling mean over a fixed ``window`` of the time axis.

    Signature: ``rolling_mean_step(name, *, source, columns=None, window=10)``.

    Output shape is ``(n - window + 1, k)``; ``window`` must not exceed the
    series length.

    Example:
        >>> rolling_mean_step("rm", source="observables", window=20)

    See ``operations.transforms`` for the shared input / selection / output contract.
    """
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=run_rolling_mean,
        kwargs=kwargs,
        step_type="rolling_mean",
    )


def rolling_std_step(name: str, **kwargs: Any) -> MCStep:
    """Trailing rolling standard deviation over a fixed ``window``.

    Signature: ``rolling_std_step(name, *, source, columns=None, window=10, ddof=0)``.

    Output shape is ``(n - window + 1, k)``; ``ddof`` picks population vs sample.

    Example:
        >>> rolling_std_step("rs", source="observables", window=20)

    See ``operations.transforms`` for the shared input / selection / output contract.
    """
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=run_rolling_std,
        kwargs=kwargs,
        step_type="rolling_std",
    )


def rolling_var_step(name: str, **kwargs: Any) -> MCStep:
    """Trailing rolling variance over a fixed ``window`` of the time axis.

    Signature: ``rolling_var_step(name, *, source, columns=None, window=10, ddof=0)``.

    Output shape is ``(n - window + 1, k)``; ``ddof`` picks population vs sample.

    Example:
        >>> rolling_var_step("rv", source="observables", window=20)

    See ``operations.transforms`` for the shared input / selection / output contract.
    """
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=run_rolling_var,
        kwargs=kwargs,
        step_type="rolling_var",
    )
