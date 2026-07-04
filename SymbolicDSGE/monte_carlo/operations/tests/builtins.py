from __future__ import annotations

from typing import Any
from ...mc_constructs import MCStep, OpType
from .._docs import with_base_doc

from .ops import (
    run_wald_test,
    run_ljung_box_test,
    run_jarque_bera_test,
    run_breusch_pagan_test,
    run_breusch_godfrey_test,
    run_cusum_test,
    run_cusumsq_test,
    run_chow_test,
)

_BASE_DOC = """
Per-replication Monte Carlo statistical tests.

- name: unique step name; also the trace-key prefix (see output below).
- Possible inputs (``source``): "observables", "states", "raw", "filter", "payload".
- Select/trim: ``column``/``columns`` to pick columns, ``burn_in`` to drop leading
  rows, ``drop_initial`` to drop the initial x0 row.
- Shared kwargs: ``filter_key="filter"``, ``payload_key=None``, ``alpha=0.05``.
- Output location: ``traces["test.<name>.statistic" | ".pval" | ".status"]`` (length n_rep).
"""


@with_base_doc(_BASE_DOC)
def wald_test_step(name: str, **kwargs: Any) -> MCStep:
    """Wald test that a sample moment equals a hypothesized target value.

    Signature: ``wald_test_step(name, *, source, target, kind="mean",
    kernel="bartlett", bandwidth="auto", columns=None)``.

    ``target`` is the hypothesized moment; ``kind`` picks which moment
    ("mean"/"covariance"/"second_moment") and ``kernel``/``bandwidth`` set the
    HAC long-run-variance estimator.

    Example:
        >>> wald_test_step("mean0", source="observables", target=np.zeros(2))
    """
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_wald_test,
        kwargs=kwargs,
        step_type="wald",
    )


@with_base_doc(_BASE_DOC)
def ljung_box_test_step(name: str, **kwargs: Any) -> MCStep:
    """Ljung-Box test for autocorrelation up to ``lags`` in one series.

    Signature: ``ljung_box_test_step(name, *, source, column=None, lags=10)``.

    ``column`` must resolve to a single series.

    Example:
        >>> ljung_box_test_step("lb", source="observables", column=0)
    """
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_ljung_box_test,
        kwargs=kwargs,
        step_type="ljung_box",
    )


@with_base_doc(_BASE_DOC)
def jarque_bera_test_step(name: str, **kwargs: Any) -> MCStep:
    """Jarque-Bera normality test on a single per-replication series.

    Signature: ``jarque_bera_test_step(name, *, source, column)``.

    ``column`` must resolve to exactly one column.

    Example:
        >>> jarque_bera_test_step("jb", source="observables", column=0)
    """
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_jarque_bera_test,
        kwargs=kwargs,
        step_type="jarque_bera",
    )


@with_base_doc(_BASE_DOC)
def breusch_pagan_test_step(name: str, **kwargs: Any) -> MCStep:
    """Breusch-Pagan test for heteroskedasticity of regression residuals.

    Signature: ``breusch_pagan_test_step(name, *, residual_source, X_source,
    residual_col=None, X_columns=None, residual_payload_key=None,
    x_payload_key=None, robust=False)``.

    Residuals and regressors are selected independently (so it does not use the
    shared ``source``); ``robust`` switches to the studentized Koenker variant.

    Example:
        >>> breusch_pagan_test_step("bp", residual_source="payload", X_source="states")
    """
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_breusch_pagan_test,
        kwargs=kwargs,
        step_type="breusch_pagan",
    )


@with_base_doc(_BASE_DOC)
def breusch_godfrey_test_step(name: str, **kwargs: Any) -> MCStep:
    """Breusch-Godfrey test for serial correlation of regression residuals.

    Signature: ``breusch_godfrey_test_step(name, *, residual_source, X_source,
    residual_col=None, X_columns=None, residual_payload_key=None,
    x_payload_key=None, lags=1)``.

    Residuals and regressors are selected independently (not via the shared
    ``source``); ``lags`` sets the auxiliary-regression lag order.

    Example:
        >>> breusch_godfrey_test_step("bg", residual_source="payload", X_source="states")
    """
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_breusch_godfrey_test,
        kwargs=kwargs,
        step_type="breusch_godfrey",
    )


@with_base_doc(_BASE_DOC)
def cusum_test_step(name: str, **kwargs: Any) -> MCStep:
    """CUSUM test for parameter stability of a recursive ``y ~ X`` regression.

    Signature: ``cusum_test_step(name, *, y_source, x_source, y_column=None,
    X_columns=None, y_payload_key=None, x_payload_key=None)``.

    The dependent series and regressor matrix are selected independently; ``y``
    must resolve to one column.

    Example:
        >>> cusum_test_step("cs", y_source="observables", x_source="states")
    """
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_cusum_test,
        kwargs=kwargs,
        step_type="cusum",
    )


@with_base_doc(_BASE_DOC)
def cusumsq_test_step(name: str, **kwargs: Any) -> MCStep:
    """CUSUM-of-squares test for variance stability of a ``y ~ X`` regression.

    Signature: ``cusumsq_test_step(name, *, y_source, x_source, y_column=None,
    X_columns=None, y_payload_key=None, x_payload_key=None)``.

    Same selection as :func:`cusum_test_step`; ``y`` must resolve to one column.

    Example:
        >>> cusumsq_test_step("csq", y_source="observables", x_source="states")
    """
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_cusumsq_test,
        kwargs=kwargs,
        step_type="cusumsq",
    )


@with_base_doc(_BASE_DOC)
def chow_test_step(name: str, **kwargs: Any) -> MCStep:
    """Chow test for a structural break in a ``y ~ X`` regression at ``t_break``.

    Signature: ``chow_test_step(name, *, y_source, x_source, t_break=10,
    y_column=None, X_columns=None, y_payload_key=None, x_payload_key=None)``.

    Selection matches :func:`cusum_test_step`; ``t_break`` is the row index that
    splits the sample into the two compared regimes.

    Example:
        >>> chow_test_step("chow", y_source="observables", x_source="states", t_break=50)
    """
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_chow_test,
        kwargs=kwargs,
        step_type="chow",
    )
