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
    """Wrap a user-supplied callable as an ``OpType.TRANSFORM`` step.

    Permissive on purpose: any callable runs in-process (closures, helpers,
    ``MCData``-returning data ops). Bundling such a step additionally requires
    the callable to be a
    :class:`~SymbolicDSGE.monte_carlo.custom_op.NumpyCustomFunc` (use
    ``@custom_operation`` or pass one); the bundle builder enforces and
    auto-wraps that at serialization time.
    """
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=func,
        kwargs=kwargs,
        store_key=store_key,
        step_type="transform:custom",
    )


# --------------------------------------------------------------------------- #
# Built-in transforms.                                                         #
#                                                                              #
# Each factory binds a transform op into an :class:`MCStep` of op_type         #
# TRANSFORM. The runner stores the per-rep output ndarray at ``step.name``;    #
# downstream consumers reference it via ``source="payload"`` plus              #
# ``payload_key=step.name``, which the graph validator auto-binds when the     #
# parent edge is a transform.                                                  #
# --------------------------------------------------------------------------- #


def standardize_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=run_standardize,
        kwargs=kwargs,
        step_type="standardize",
    )


def log_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=run_log,
        kwargs=kwargs,
        step_type="log",
    )


def log_diff_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=run_log_diff,
        kwargs=kwargs,
        step_type="log_diff",
    )


def diff_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=run_diff,
        kwargs=kwargs,
        step_type="diff",
    )


def rolling_mean_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=run_rolling_mean,
        kwargs=kwargs,
        step_type="rolling_mean",
    )


def rolling_std_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=run_rolling_std,
        kwargs=kwargs,
        step_type="rolling_std",
    )


def rolling_var_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=run_rolling_var,
        kwargs=kwargs,
        step_type="rolling_var",
    )
