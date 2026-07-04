"""Post-processing step factories (``OpType.POSTPROC``)."""

from __future__ import annotations

from typing import Any, Callable

from ...mc_constructs import MCStep, OpType
from .._docs import with_base_doc
from .ops import run_kde

_BASE_DOC = """
Post-loop Monte Carlo summaries: run once over the assembled traces.

- Input: the across-replication ``traces`` registry (length-n_rep arrays keyed
  like "test.<name>.pval", "regression.<name>.coef", "payload.<name>").
- Output location: ``result.postproc[name]`` holds the op's return verbatim;
  a returned mapping fans out to "<name>.<key>" entries on serialize.
"""


def postproc_step(
    name: str,
    func: Callable[..., Any],
    *,
    store_key: str | None = None,
    **kwargs: Any,
) -> MCStep:
    """Wrap a user callable as a post-loop ``OpType.POSTPROC`` step.

    Signature: ``postproc_step(name, func, *, store_key=None, **kwargs)``.

    The op contract is ``func(*, traces, **kwargs)``; it runs once and its
    return is stored at ``result.postproc[name]`` (a
    :class:`~SymbolicDSGE.monte_carlo.postproc.Summary` / ``Raw``, a mapping of
    them, or a plain scalar / array / DataFrame). Bundling requires ``func`` to
    be a :class:`~SymbolicDSGE.monte_carlo.custom_op.CustomFunc` under the
    pandas namespace (so a returned DataFrame's builder may reference ``pd``).

    Example:
        >>> postproc_step("summary", my_postproc_op)
    """
    return MCStep(
        name=name,
        op_type=OpType.POSTPROC,
        func=func,
        kwargs=kwargs,
        store_key=store_key,
        step_type="postproc:custom",
    )


@with_base_doc(_BASE_DOC)
def kde_step(name: str, **kwargs: Any) -> MCStep:
    """Gaussian kernel density estimate of one across-replication trace.

    Signature: ``kde_step(name, *, trace, bandwidth="scott", grid_points=200,
    kernel="gaussian")``.

    ``trace`` names the source series (e.g. "test.<name>.statistic"). Returns a
    mapping of "curve" (an (N, 2) (x, density) array) and "descriptives" (a
    summary table of moments/quantiles).

    Example:
        >>> kde_step("density", trace="test.jb.statistic")
    """
    return MCStep(
        name=name,
        op_type=OpType.POSTPROC,
        func=run_kde,
        kwargs=kwargs,
        step_type="kde",
    )
