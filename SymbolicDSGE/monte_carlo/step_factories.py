"""Public constructors for native Monte Carlo pipeline steps."""

from __future__ import annotations

from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.shock_generators import Shock
from .custom_op import NumbaCustomFunc
from .mc_constructs import ColumnSelector, MCStep, OpType, _compile_source_args
from .postproc import run_kde

NDF = NDArray[np.float64]


def simulation_step(
    name: str = "datagen",
    target: str = "dgp",
    *,
    T: int,
    shocks: Mapping[str, Shock | Callable[[float | NDF], NDF] | NDF] | None = None,
    seed_increment: int | Literal["auto"] = "auto",
    shock_scale: float = 1.0,
    x0: list[float] | NDF | None = None,
    observables: bool = True,
) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.DATAGEN,
        kwargs={
            "target": target,
            "T": T,
            "shocks": shocks,
            "seed_increment": seed_increment,
            "shock_scale": shock_scale,
            "x0": x0,
            "observables": observables,
        },
        step_type="simulation",
    )


def raw_model_data_step(
    name: str = "datagen",
    *,
    states: NDF | Sequence[float] | Sequence[Sequence[float]] | None = None,
    observables: NDF | Sequence[float] | Sequence[Sequence[float]] | None = None,
    observable_names: Sequence[str] = (),
) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.DATAGEN,
        kwargs={
            "states": states,
            "observables": observables,
            "observable_names": observable_names,
        },
        step_type="raw_model_data",
    )


def reference_filter_step(
    name: str = "filter",
    *,
    filter_mode: Literal["linear", "extended", "unscented"] = "linear",
    observables: list[str] | None = None,
    x0: list[float] | NDF | None = None,
    P0: NDF | None = None,
    R: NDF | None = None,
    jitter: float | np.float64 | None = None,
    symmetrize: bool | None = None,
    return_shocks: bool = False,
) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.FILTER,
        kwargs={
            "filter_mode": filter_mode,
            "observables": observables,
            "x0": x0,
            "P0": P0,
            "R": R,
            "jitter": jitter,
            "symmetrize": symmetrize,
            "return_shocks": return_shocks,
        },
        step_type="filter",
    )


def add_payload_step(
    name: str,
    payload: (
        NDF
        | Sequence[float]
        | Sequence[Sequence[float]]
        | Sequence[Sequence[Sequence[float]]]
    ),
) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        kwargs={"value": payload},
        step_type="payload",
    )


def _one_source_step(
    name: str,
    op_type: OpType,
    step_type: str,
    *,
    source: str,
    field: str,
    columns: ColumnSelector,
    burn_in: int,
    drop_initial: bool,
    kwargs: Mapping[str, Any],
) -> MCStep:
    return MCStep(
        name=name,
        op_type=op_type,
        kwargs=kwargs,
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
        step_type=step_type,
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
    return _one_source_step(
        name,
        OpType.TRANSFORM,
        "standardize",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"ddof": ddof},
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
    return _one_source_step(
        name,
        OpType.TRANSFORM,
        "log",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"offset": offset},
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
    return _one_source_step(
        name,
        OpType.TRANSFORM,
        "log_diff",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"offset": offset},
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
    return _one_source_step(
        name,
        OpType.TRANSFORM,
        "diff",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"order": order},
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
    return _one_source_step(
        name,
        OpType.TRANSFORM,
        "rolling_mean",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"window": window},
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
    return _one_source_step(
        name,
        OpType.TRANSFORM,
        "rolling_std",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"window": window, "ddof": ddof},
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
    return _one_source_step(
        name,
        OpType.TRANSFORM,
        "rolling_var",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"window": window, "ddof": ddof},
    )


def regression_step(
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
    kind: Literal[
        "ols", "ridge", "lasso", "elastic_net", "ridge_gs", "lasso_gs", "elastic_net_gs"
    ] = "ols",
    intercept: bool = True,
    variables: list[str] | None = None,
    **kind_kwargs: Any,
) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.REGRESSION,
        kwargs={
            "kind": kind,
            "intercept": intercept,
            "variables": variables,
            **kind_kwargs,
        },
        source_args=(
            _compile_source_args(
                arg="y",
                source=y_source,
                field=y_field,
                columns=y_column,
                burn_in=burn_in,
                drop_initial=drop_initial,
            ),
            _compile_source_args(
                arg="X",
                source=X_source,
                field=X_field,
                columns=X_columns,
                burn_in=burn_in,
                drop_initial=drop_initial,
            ),
        ),
        step_type="regression",
    )


def _two_source_test(
    name: str,
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
    kwargs: Mapping[str, Any],
) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        kwargs=kwargs,
        source_args=(
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
        ),
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
    kind: Literal["mean", "covariance", "second_moment"] = "mean",
    target: NDF,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["andrews", "wooldridge", "auto"] | None = "auto",
    alpha: float = 0.05,
) -> MCStep:
    return _one_source_step(
        name,
        OpType.TEST,
        "wald",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={
            "kind": kind,
            "target": target,
            "kernel": kernel,
            "bandwidth": bandwidth,
            "alpha": alpha,
        },
    )


def ljung_box_test_step(
    name: str,
    *,
    source: str,
    field: str,
    column: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    lags: int = 10,
    alpha: float = 0.05,
) -> MCStep:
    return _one_source_step(
        name,
        OpType.TEST,
        "ljung_box",
        source=source,
        field=field,
        columns=column,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"lags": lags, "alpha": alpha},
    )


def jarque_bera_test_step(
    name: str,
    *,
    source: str,
    field: str,
    column: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    alpha: float = 0.05,
) -> MCStep:
    return _one_source_step(
        name,
        OpType.TEST,
        "jarque_bera",
        source=source,
        field=field,
        columns=column,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"alpha": alpha},
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
    robust: bool = False,
    alpha: float = 0.05,
) -> MCStep:
    return _two_source_test(
        name,
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
        kwargs={"robust": robust, "alpha": alpha},
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
    lags: int = 1,
    alpha: float = 0.05,
) -> MCStep:
    return _two_source_test(
        name,
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
        kwargs={"lags": lags, "alpha": alpha},
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
    alpha: float = 0.05,
) -> MCStep:
    return _two_source_test(
        name,
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
        kwargs={"alpha": alpha},
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
    alpha: float = 0.05,
) -> MCStep:
    return _two_source_test(
        name,
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
        kwargs={"alpha": alpha},
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
    t_break: int = 10,
    alpha: float = 0.05,
) -> MCStep:
    return _two_source_test(
        name,
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
        kwargs={"t_break": t_break, "alpha": alpha},
    )


def postproc_step(name: str, func: Callable[..., Any], **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.POSTPROC,
        func=func,
        kwargs=kwargs,
        step_type="postproc:custom",
    )


def kde_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name, op_type=OpType.POSTPROC, func=run_kde, kwargs=kwargs, step_type="kde"
    )
