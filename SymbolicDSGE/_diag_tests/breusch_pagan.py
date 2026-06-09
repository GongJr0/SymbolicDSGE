from __future__ import annotations

import numpy as np
from numba import njit
from numpy import float64
from numpy.typing import NDArray

from ..regression.enums import RegressionStatus
from ..regression.solvers import chol_solve, lstsq_solve
from .distributions import PvalMethod, ReferenceDistribution
from .result import TestResult
from .status import TestStatus

OK = int(TestStatus.OK)
LINALG = int(TestStatus.LINALG)
UDEF_VARIANCE = int(TestStatus.UDEF_VARIANCE)
INSUFFICIENT_SAMPLES = int(TestStatus.INSUFFICIENT_SAMPLES)
REGRESSION_OK = int(RegressionStatus.OK)

NDF = NDArray[float64]


@njit(cache=True)
def bp_aux(eps: NDF, X: NDF) -> tuple[int, float64, float64]:
    """Fit the auxiliary regression and return its RSS and centered TSS."""
    n = X.shape[0]
    if n == 0:
        return INSUFFICIENT_SAMPLES, float64(np.nan), float64(np.nan)

    eps_sq = eps**2
    sigma2 = np.sum(eps_sq) / n
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        return UDEF_VARIANCE, float64(np.nan), float64(np.nan)
    g = eps_sq / sigma2

    try:
        bhat, _, regression_status = chol_solve(X, g)
    except Exception:
        bhat, _, regression_status = lstsq_solve(X, g)
    if regression_status != REGRESSION_OK:
        return LINALG, float64(np.nan), float64(np.nan)

    residuals = g - X @ bhat
    rss = np.sum(residuals**2)
    g_mean = np.sum(g) / n
    tss = np.sum((g - g_mean) ** 2)
    return OK, rss, tss


@njit(cache=True)
def bp_stat(eps: NDF, X: NDF) -> tuple[int, float64]:
    status, rss, tss = bp_aux(eps, X)
    if status != OK:
        return status, float64(np.nan)

    return OK, max(float64(0.0), (tss - rss) * 0.5)


@njit(cache=True)
def robust_bp_stat(eps: NDF, X: NDF) -> tuple[int, float64]:
    status, rss, tss = bp_aux(eps, X)
    if status != OK:
        return status, float64(np.nan)
    if tss <= 0.0:
        return OK, float64(0.0)

    r2 = min(float64(1.0), max(float64(0.0), float64(1.0 - rss / tss)))
    return OK, r2 * len(eps)


def _prepare_bp_inputs(eps: NDF, X: NDF) -> tuple[NDF, NDF]:
    residuals = np.ascontiguousarray(eps, dtype=np.float64)
    regressors = np.ascontiguousarray(X, dtype=np.float64)

    if residuals.ndim != 1:
        raise ValueError("Breusch-Pagan residuals must be a 1D array.")
    if regressors.ndim != 2:
        raise ValueError("Breusch-Pagan regressors must be a 2D array.")
    if residuals.shape[0] != regressors.shape[0]:
        raise ValueError("Breusch-Pagan residual and regressor row counts differ.")
    if regressors.shape[1] == 0:
        raise ValueError("Breusch-Pagan requires at least one variance regressor.")
    if not np.isfinite(residuals).all() or not np.isfinite(regressors).all():
        raise ValueError("Breusch-Pagan inputs must contain only finite values.")
    if regressors.shape[0] > 0 and np.any(np.all(regressors == regressors[0], axis=0)):
        raise ValueError(
            "Breusch-Pagan regressors must not contain a constant column; "
            "an intercept is added automatically."
        )

    augmented = np.empty(
        (regressors.shape[0], regressors.shape[1] + 1), dtype=np.float64
    )
    augmented[:, 0] = 1.0
    augmented[:, 1:] = regressors
    return residuals, augmented


def breusch_pagan(
    eps: NDF, X: NDF, alpha: float = 0.05, _auto_pval: bool = True
) -> TestResult:
    residuals, regressors = _prepare_bp_inputs(eps, X)
    status, stat = bp_stat(residuals, regressors)
    return TestResult(
        test_name="breusch_pagan",
        status=TestStatus(status),
        statistic=stat,
        pval_method=PvalMethod.SF,
        dist=ReferenceDistribution.CHI2,
        df=regressors.shape[1] - 1,
        alpha=float64(alpha),
        _auto_pval=_auto_pval,
    )


def robust_breusch_pagan(
    eps: NDF, X: NDF, alpha: float = 0.05, _auto_pval: bool = True
) -> TestResult:
    residuals, regressors = _prepare_bp_inputs(eps, X)
    status, stat = robust_bp_stat(residuals, regressors)
    return TestResult(
        test_name="robust_breusch_pagan",
        status=TestStatus(status),
        statistic=stat,
        pval_method=PvalMethod.SF,
        dist=ReferenceDistribution.CHI2,
        df=regressors.shape[1] - 1,
        alpha=float64(alpha),
        _auto_pval=_auto_pval,
    )
