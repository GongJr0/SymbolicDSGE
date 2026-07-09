from __future__ import annotations

from typing import Any
from ...mc_constructs import ColumnSelector, MCStep, OpType, _compile_source_args

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


def _single_source_test_step(
    name: str,
    func: Any,
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
        op_type=OpType.TEST,
        func=func,
        kwargs=step_kwargs,
        source_args=source_args,
        step_type=step_type,
    )


def _two_source_test_step(
    name: str,
    func: Any,
    step_type: str,
    *,
    first_source: str,
    first_field: str,
    first_arg: str,
    first_columns: ColumnSelector,
    second_source: str,
    second_field: str,
    second_arg: str,
    second_columns: ColumnSelector,
    burn_in: int,
    drop_initial: bool,
    step_kwargs: dict[str, Any],
) -> MCStep:
    source_args = (
        _compile_source_args(
            arg=first_arg,
            source=first_source,
            field=first_field,
            columns=first_columns,
            burn_in=burn_in,
            drop_initial=drop_initial,
        ),
        _compile_source_args(
            arg=second_arg,
            source=second_source,
            field=second_field,
            columns=second_columns,
            burn_in=burn_in,
            drop_initial=drop_initial,
        ),
    )
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=func,
        kwargs=step_kwargs,
        source_args=source_args,
        step_type=step_type,
    )


def wald_test_step(
    name: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """Wald test that a sample moment equals a hypothesized target value.

    Signature: ``wald_test_step(name, *, source, field, target, kind="mean",
    kernel="bartlett", bandwidth="auto", columns=None)``.

    ``target`` is the hypothesized moment; ``kind`` picks which moment
    ("mean"/"covariance"/"second_moment") and ``kernel``/``bandwidth`` set the
    HAC long-run-variance estimator.

    Example:
        >>> wald_test_step("mean0", source="filter", field="std_innov", target=np.zeros(2))

    See ``operations.tests`` for the shared input / selection / output contract.
    """
    return _single_source_test_step(
        name,
        run_wald_test,
        "wald",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        step_kwargs=step_kwargs,
    )


def ljung_box_test_step(
    name: str,
    *,
    source: str,
    field: str,
    column: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """Ljung-Box test for autocorrelation up to ``lags`` in one series.

    Signature: ``ljung_box_test_step(name, *, source, field, column=None, lags=10)``.

    ``column`` must resolve to a single series.

    Example:
        >>> ljung_box_test_step("lb", source="datagen", field="observables", column=0)

    See ``operations.tests`` for the shared input / selection / output contract.
    """
    return _single_source_test_step(
        name,
        run_ljung_box_test,
        "ljung_box",
        source=source,
        field=field,
        columns=column,
        burn_in=burn_in,
        drop_initial=drop_initial,
        step_kwargs=step_kwargs,
    )


def jarque_bera_test_step(
    name: str,
    *,
    source: str,
    field: str,
    column: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """Jarque-Bera normality test on a single per-replication series.

    Signature: ``jarque_bera_test_step(name, *, source, field, column)``.

    ``column`` must resolve to exactly one column.

    Example:
        >>> jarque_bera_test_step("jb", source="datagen", field="observables", column=0)

    See ``operations.tests`` for the shared input / selection / output contract.
    """
    return _single_source_test_step(
        name,
        run_jarque_bera_test,
        "jarque_bera",
        source=source,
        field=field,
        columns=column,
        burn_in=burn_in,
        drop_initial=drop_initial,
        step_kwargs=step_kwargs,
    )


def breusch_pagan_test_step(
    name: str,
    *,
    residuals_source: str,
    residuals_field: str,
    X_source: str,
    X_field: str,
    residual_col: ColumnSelector = None,
    X_columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """Breusch-Pagan test for heteroskedasticity of regression residuals.

    Signature: ``breusch_pagan_test_step(name, *, residuals_source,
    residuals_field, X_source, X_field,
    residual_col=None, X_columns=None, robust=False)``.

    Residuals and regressors are selected independently (so it does not use the
    shared ``source``); ``robust`` switches to the studentized Koenker variant.

    Example:
        >>> breusch_pagan_test_step("bp", residuals_source="u", residuals_field="payload", X_source="datagen", X_field="states")

    See ``operations.tests`` for the shared selection / output contract.
    """
    return _two_source_test_step(
        name,
        run_breusch_pagan_test,
        "breusch_pagan",
        first_source=residuals_source,
        first_field=residuals_field,
        first_arg="residuals",
        first_columns=residual_col,
        second_source=X_source,
        second_field=X_field,
        second_arg="X",
        second_columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        step_kwargs=step_kwargs,
    )


def breusch_godfrey_test_step(
    name: str,
    *,
    residuals_source: str,
    residuals_field: str,
    X_source: str,
    X_field: str,
    residual_col: ColumnSelector = None,
    X_columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """Breusch-Godfrey test for serial correlation of regression residuals.

    Signature: ``breusch_godfrey_test_step(name, *, residuals_source,
    residuals_field, X_source, X_field,
    residual_col=None, X_columns=None, lags=1)``.

    Residuals and regressors are selected independently (not via the shared
    ``source``); ``lags`` sets the auxiliary-regression lag order.

    Example:
        >>> breusch_godfrey_test_step("bg", residuals_source="u", residuals_field="payload", X_source="datagen", X_field="states")

    See ``operations.tests`` for the shared selection / output contract.
    """
    return _two_source_test_step(
        name,
        run_breusch_godfrey_test,
        "breusch_godfrey",
        first_source=residuals_source,
        first_field=residuals_field,
        first_arg="residuals",
        first_columns=residual_col,
        second_source=X_source,
        second_field=X_field,
        second_arg="X",
        second_columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        step_kwargs=step_kwargs,
    )


def cusum_test_step(
    name: str,
    *,
    y_source: str,
    y_field: str,
    X_source: str,
    X_field: str,
    y_column: ColumnSelector = None,
    X_columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """CUSUM test for parameter stability of a recursive ``y ~ X`` regression.

    Signature: ``cusum_test_step(name, *, y_source, y_field, X_source, X_field, y_column=None, X_columns=None)``.

    The dependent series and regressor matrix are selected independently; ``y``
    must resolve to one column.

    Example:
        >>> cusum_test_step("cs", y_source="datagen", y_field="observables", X_source="datagen", X_field="states")

    See ``operations.tests`` for the shared selection / output contract.
    """
    return _two_source_test_step(
        name,
        run_cusum_test,
        "cusum",
        first_source=y_source,
        first_field=y_field,
        first_arg="y",
        first_columns=y_column,
        second_source=X_source,
        second_field=X_field,
        second_arg="X",
        second_columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        step_kwargs=step_kwargs,
    )


def cusumsq_test_step(
    name: str,
    *,
    y_source: str,
    y_field: str,
    X_source: str,
    X_field: str,
    y_column: ColumnSelector = None,
    X_columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """CUSUM-of-squares test for variance stability of a ``y ~ X`` regression.

    Signature: ``cusumsq_test_step(name, *, y_source, y_field, X_source, X_field, y_column=None, X_columns=None)``.

    Same selection as :func:`cusum_test_step`; ``y`` must resolve to one column.

    Example:
        >>> cusumsq_test_step("csq", y_source="datagen", y_field="observables", X_source="datagen", X_field="states")

    See ``operations.tests`` for the shared selection / output contract.
    """
    return _two_source_test_step(
        name,
        run_cusumsq_test,
        "cusumsq",
        first_source=y_source,
        first_field=y_field,
        first_arg="y",
        first_columns=y_column,
        second_source=X_source,
        second_field=X_field,
        second_arg="X",
        second_columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        step_kwargs=step_kwargs,
    )


def chow_test_step(
    name: str,
    *,
    y_source: str,
    y_field: str,
    X_source: str,
    X_field: str,
    y_column: ColumnSelector = None,
    X_columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    **step_kwargs: Any,
) -> MCStep:
    """Chow test for a structural break in a ``y ~ X`` regression at ``t_break``.

    Signature: ``chow_test_step(name, *, y_source, y_field, X_source, X_field, t_break=10,
    y_column=None, X_columns=None)``.

    Selection matches :func:`cusum_test_step`; ``t_break`` is the row index that
    splits the sample into the two compared regimes.

    Example:
        >>> chow_test_step("chow", y_source="datagen", y_field="observables", X_source="datagen", X_field="states", t_break=50)

    See ``operations.tests`` for the shared selection / output contract.
    """
    return _two_source_test_step(
        name,
        run_chow_test,
        "chow",
        first_source=y_source,
        first_field=y_field,
        first_arg="y",
        first_columns=y_column,
        second_source=X_source,
        second_field=X_field,
        second_arg="X",
        second_columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        step_kwargs=step_kwargs,
    )
