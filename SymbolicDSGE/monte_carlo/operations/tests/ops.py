from __future__ import annotations

import numpy as np

from typing import Sequence, Literal
from ..types import InpSources, NDF

from ...mc_constructs import MCContext
from ....core.solved_model import SolvedModel
from ..utils import _resolve_context_array


from ...._diag_tests.result import TestResult
from ...._diag_tests.ljung_box import ljung_box
from ...._diag_tests.jarque_bera import jarque_bera
from ...._diag_tests.breusch_pagan import breusch_pagan, robust_breusch_pagan
from ...._diag_tests.breusch_godfrey import breusch_godfrey
from ...._diag_tests.cusum import cusum
from ...._diag_tests.cusumsq import cusumsq_test
from ...._diag_tests.chow import chow
from ...._diag_tests.wald_test import (
    wald_mean_hac,
    wald_covariance_hac,
    wald_second_moment_hac,
)


def run_wald_test(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    source: InpSources,
    target: NDF,
    kind: Literal["mean", "covariance", "second_moment"] = "mean",
    filter_key: str = "filter",
    payload_key: str | None = None,
    columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["andrews", "wooldridge", "auto"] | None = "auto",
    alpha: float = 0.05,
) -> TestResult:
    del reference, dgp, rep_idx
    arr = _resolve_context_array(
        context,
        source=source,
        filter_key=filter_key,
        payload_key=payload_key,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    target_arr = np.asarray(target, dtype=np.float64)
    if kind == "mean":
        return wald_mean_hac(
            arr,
            target_arr,
            kernel=kernel,
            bandwidth=bandwidth,
            alpha=alpha,
            _auto_pval=False,
        )
    if kind == "covariance":
        return wald_covariance_hac(
            arr,
            target_arr,
            kernel=kernel,
            bandwidth=bandwidth,
            alpha=alpha,
            _auto_pval=False,
        )
    if kind == "second_moment":
        return wald_second_moment_hac(
            arr,
            target_arr,
            kernel=kernel,
            bandwidth=bandwidth,
            alpha=alpha,
            _auto_pval=False,
        )
    raise ValueError(f"Unsupported Wald test kind: {kind}")


def run_ljung_box_test(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    source: InpSources,
    filter_key: str = "filter",
    payload_key: str | None = None,
    column: Sequence[int] | int | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    lags: int = 10,
    alpha: float = 0.05,
) -> TestResult:
    del reference, dgp, rep_idx

    col_idx: Sequence[int] | None
    if isinstance(column, int):
        col_idx = [column]
    else:
        col_idx = column

    arr = _resolve_context_array(
        context,
        source=source,
        filter_key=filter_key,
        payload_key=payload_key,
        columns=col_idx,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    if arr.shape[1] != 1:
        raise ValueError("Ljung-Box test requires a single column of data.")

    return ljung_box(arr[:, 0], L=lags, alpha=alpha, _auto_pval=False)


def run_jarque_bera_test(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    source: InpSources,
    filter_key: str = "filter",
    payload_key: str | None = None,
    column: Sequence[int] | int | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    alpha: float = 0.05,
) -> TestResult:
    del reference, dgp, rep_idx

    col_idx: Sequence[int] | None
    if isinstance(column, int):
        col_idx = [column]
    else:
        col_idx = column

    arr = _resolve_context_array(
        context,
        source=source,
        filter_key=filter_key,
        payload_key=payload_key,
        columns=col_idx,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    if arr.shape[1] != 1:
        raise ValueError("Jarque-Bera test requires a single column of data.")

    return jarque_bera(arr[:, 0], alpha=alpha, _auto_pval=False)


def run_breusch_pagan_test(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    residual_source: InpSources,
    X_source: InpSources,
    filter_key: str = "filter",
    residual_payload_key: str | None = None,
    x_payload_key: str | None = None,
    residual_col: Sequence[int] | int | None = None,
    X_columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    alpha: float = 0.05,
    robust: bool = False,
) -> TestResult:
    del reference, dgp, rep_idx

    residual_col_idx: Sequence[int] | None
    if isinstance(residual_col, int):
        residual_col_idx = [residual_col]
    else:
        residual_col_idx = residual_col

    residuals = _resolve_context_array(
        context,
        source=residual_source,
        filter_key=filter_key,
        payload_key=residual_payload_key,
        columns=residual_col_idx,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    X = _resolve_context_array(
        context,
        source=X_source,
        filter_key=filter_key,
        payload_key=x_payload_key,
        columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )

    if residuals.shape[1] != 1:
        raise ValueError(
            "Breusch-Pagan residuals must resolve to exactly one column. "
            f"Got shape {residuals.shape}."
        )
    if residuals.shape[0] != X.shape[0]:
        raise ValueError(
            "Breusch-Pagan residuals and regressors must have the same number "
            f"of rows. Got residuals={residuals.shape[0]} and X={X.shape[0]}."
        )

    residual_vec = np.ascontiguousarray(residuals[:, 0], dtype=np.float64)
    if robust:
        return robust_breusch_pagan(residual_vec, X, alpha=alpha, _auto_pval=False)
    return breusch_pagan(residual_vec, X, alpha=alpha, _auto_pval=False)


def run_breusch_godfrey_test(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    residual_source: InpSources,
    X_source: InpSources,
    filter_key: str = "filter",
    residual_payload_key: str | None = None,
    x_payload_key: str | None = None,
    residual_col: Sequence[int] | int | None = None,
    X_columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    lags: int = 1,
    alpha: float = 0.05,
) -> TestResult:
    del reference, dgp, rep_idx

    residual_col_idx: Sequence[int] | None
    if isinstance(residual_col, int):
        residual_col_idx = [residual_col]
    else:
        residual_col_idx = residual_col

    residuals = _resolve_context_array(
        context,
        source=residual_source,
        filter_key=filter_key,
        payload_key=residual_payload_key,
        columns=residual_col_idx,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    X = _resolve_context_array(
        context,
        source=X_source,
        filter_key=filter_key,
        payload_key=x_payload_key,
        columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )

    if residuals.shape[1] != 1:
        raise ValueError(
            "Breusch-Godfrey residuals must resolve to exactly one column. "
            f"Got shape {residuals.shape}."
        )
    if residuals.shape[0] != X.shape[0]:
        raise ValueError(
            "Breusch-Godfrey residuals and regressors must have the same number "
            f"of rows. Got residuals={residuals.shape[0]} and X={X.shape[0]}."
        )

    residual_vec = np.ascontiguousarray(residuals[:, 0], dtype=np.float64)
    return breusch_godfrey(residual_vec, X, lags=lags, alpha=alpha, _auto_pval=False)


def run_cusum_test(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    x_source: InpSources,
    y_source: InpSources,
    filter_key: str = "filter",
    y_payload_key: str | None = None,
    x_payload_key: str | None = None,
    y_column: Sequence[int] | int | None = None,
    X_columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    alpha: float = 0.05,
) -> TestResult:
    del reference, dgp, rep_idx
    y_col_idx: Sequence[int] | None
    if isinstance(y_column, int):
        y_col_idx = [y_column]
    else:
        y_col_idx = y_column
    y = _resolve_context_array(
        context,
        source=y_source,
        filter_key=filter_key,
        payload_key=y_payload_key,
        columns=y_col_idx,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    X = _resolve_context_array(
        context,
        source=x_source,
        filter_key=filter_key,
        payload_key=x_payload_key,
        columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    if y.shape[1] != 1:
        raise ValueError(
            "CUSUM test requires the dependent variable to resolve to exactly "
            f"one column. Got shape {y.shape}."
        )
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            "CUSUM test dependent variable and regressors must have the same "
            f"number of rows. Got y={y.shape[0]} and X={X.shape[0]}."
        )
    return cusum(y[:, 0], X, alpha=alpha, _auto_pval=False)


def run_cusumsq_test(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    x_source: InpSources,
    y_source: InpSources,
    filter_key: str = "filter",
    y_payload_key: str | None = None,
    x_payload_key: str | None = None,
    y_column: Sequence[int] | int | None = None,
    X_columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    alpha: float = 0.05,
) -> TestResult:
    del reference, dgp, rep_idx
    y_col_idx: Sequence[int] | None
    if isinstance(y_column, int):
        y_col_idx = [y_column]
    else:
        y_col_idx = y_column
    y = _resolve_context_array(
        context,
        source=y_source,
        filter_key=filter_key,
        payload_key=y_payload_key,
        columns=y_col_idx,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    X = _resolve_context_array(
        context,
        source=x_source,
        filter_key=filter_key,
        payload_key=x_payload_key,
        columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    if y.shape[1] != 1:
        raise ValueError(
            "CUSUMSQ test requires the dependent variable to resolve to exactly "
            f"one column. Got shape {y.shape}."
        )
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            "CUSUMSQ test dependent variable and regressors must have the same "
            f"number of rows. Got y={y.shape[0]} and X={X.shape[0]}."
        )
    return cusumsq_test(y[:, 0], X, alpha=alpha, _auto_pval=False)


def run_chow_test(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    x_source: InpSources,
    y_source: InpSources,
    filter_key: str = "filter",
    y_payload_key: str | None = None,
    x_payload_key: str | None = None,
    y_column: Sequence[int] | int | None = None,
    X_columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    t_break: int = 10,
    alpha: float = 0.05,
) -> TestResult:
    del reference, dgp, rep_idx
    y_col_idx: Sequence[int] | None
    if isinstance(y_column, int):
        y_col_idx = [y_column]
    else:
        y_col_idx = y_column
    y = _resolve_context_array(
        context,
        source=y_source,
        filter_key=filter_key,
        payload_key=y_payload_key,
        columns=y_col_idx,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    X = _resolve_context_array(
        context,
        source=x_source,
        filter_key=filter_key,
        payload_key=x_payload_key,
        columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    if y.shape[1] != 1:
        raise ValueError(
            "Chow test requires the dependent variable to resolve to exactly "
            f"one column. Got shape {y.shape}."
        )
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            "Chow test dependent variable and regressors must have the same "
            f"number of rows. Got y={y.shape[0]} and X={X.shape[0]}."
        )
    return chow(y[:, 0], X, t_break=t_break, alpha=alpha, _auto_pval=False)
