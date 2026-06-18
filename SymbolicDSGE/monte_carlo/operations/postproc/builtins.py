"""Post-processing step factories (``OpType.POSTPROC``)."""

from __future__ import annotations

from typing import Any, Callable

from ...mc_constructs import MCStep, OpType
from .ops import run_kde


def postproc_step(
    name: str,
    func: Callable[..., Any],
    *,
    store_key: str | None = None,
    **kwargs: Any,
) -> MCStep:
    """Wrap a user callable as an ``OpType.POSTPROC`` step.

    Post-loop op contract: ``func(*, traces, reference, dgp, **kwargs)`` returning
    :class:`~SymbolicDSGE.monte_carlo.postproc.Summary` / ``Raw`` artifacts (or a
    mapping of them). Bundling additionally requires the callable to be a
    :class:`~SymbolicDSGE.monte_carlo.custom_op.CustomFunc`; the bundle builder
    enforces/auto-wraps post-loop ops under the pandas namespace (so a returned
    DataFrame's builder code may reference ``pd``) at serialization time.
    """
    return MCStep(
        name=name,
        op_type=OpType.POSTPROC,
        func=func,
        kwargs=kwargs,
        store_key=store_key,
        step_type="postproc:custom",
    )


def kde_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.POSTPROC,
        func=run_kde,
        kwargs=kwargs,
        step_type="kde",
    )
