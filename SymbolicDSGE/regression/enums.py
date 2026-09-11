from __future__ import annotations
from enum import IntEnum
from enum import StrEnum


class RegressionStatus(IntEnum):
    """Mirrors the REGRESSION_* codes in _ckernels/regression/regression.h."""

    OK = 0
    RANK_DEFICIENT = -1501
    NON_CONVERGENT = -1502


class RegressionKind(StrEnum):
    """Regression method a fit or a Monte Carlo regression step ran."""

    OLS = "ols"
    RIDGE = "ridge"
    RIDGE_GS = "ridge_gs"
    LASSO = "lasso"
    LASSO_GS = "lasso_gs"
    ELASTIC_NET = "elastic_net"
    ELASTIC_NET_GS = "elastic_net_gs"
