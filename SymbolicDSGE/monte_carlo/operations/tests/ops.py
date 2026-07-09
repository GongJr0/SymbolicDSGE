from __future__ import annotations

import numpy as np

from typing import Literal
from ..types import NDF

from ...mc_constructs import MCContext
from ....core.solved_model import SolvedModel


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
    sample: NDF,
    target: NDF,
    kind: Literal["mean", "covariance", "second_moment"] = "mean",
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["andrews", "wooldridge", "auto"] | None = "auto",
    alpha: float = 0.05,
) -> TestResult:
    del context, reference, dgp, rep_idx
    arr = sample
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
    sample: NDF,
    lags: int = 10,
    alpha: float = 0.05,
) -> TestResult:
    del context, reference, dgp, rep_idx
    arr = sample
    if arr.shape[1] != 1:
        raise ValueError("Ljung-Box test requires a single column of data.")

    return ljung_box(arr[:, 0], L=lags, alpha=alpha, _auto_pval=False)


def run_jarque_bera_test(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    sample: NDF,
    alpha: float = 0.05,
) -> TestResult:
    del context, reference, dgp, rep_idx
    arr = sample
    if arr.shape[1] != 1:
        raise ValueError("Jarque-Bera test requires a single column of data.")

    return jarque_bera(arr[:, 0], alpha=alpha, _auto_pval=False)


def run_breusch_pagan_test(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    residuals: NDF,
    X: NDF,
    alpha: float = 0.05,
    robust: bool = False,
) -> TestResult:
    del context, reference, dgp, rep_idx

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

    if robust:
        return robust_breusch_pagan(residuals[:, 0], X, alpha=alpha, _auto_pval=False)
    return breusch_pagan(residuals[:, 0], X, alpha=alpha, _auto_pval=False)


def run_breusch_godfrey_test(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    residuals: NDF,
    X: NDF,
    lags: int = 1,
    alpha: float = 0.05,
) -> TestResult:
    del context, reference, dgp, rep_idx

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

    return breusch_godfrey(residuals[:, 0], X, lags=lags, alpha=alpha, _auto_pval=False)


def run_cusum_test(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    y: NDF,
    X: NDF,
    alpha: float = 0.05,
) -> TestResult:
    del context, reference, dgp, rep_idx
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
    y: NDF,
    X: NDF,
    alpha: float = 0.05,
) -> TestResult:
    del context, reference, dgp, rep_idx
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
    y: NDF,
    X: NDF,
    t_break: int = 10,
    alpha: float = 0.05,
) -> TestResult:
    del context, reference, dgp, rep_idx
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
