from __future__ import annotations
from enum import IntEnum
from enum import StrEnum


class RegressionStatus(IntEnum):
    OK = 0
    RANK_DEFICIENT = -1


class RegressionKind(StrEnum):
    OLS = "ols"
    RIDGE = "ridge"
    RIDGE_GS = "ridge_gs"
