from __future__ import annotations
from enum import IntEnum
from enum import StrEnum


class RegressionStatus(IntEnum):
    OK = 0
    RANK_DEFICIENT = -1
    NON_CONVERGENT = -2


class RegressionKind(StrEnum):
    OLS = "ols"
    RIDGE = "ridge"
    RIDGE_GS = "ridge_gs"
    LASSO = "lasso"
    LASSO_GS = "lasso_gs"
    ELASTIC_NET = "elastic_net"
    ELASTIC_NET_GS = "elastic_net_gs"
