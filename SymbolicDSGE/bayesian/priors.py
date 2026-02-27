from .transforms.transform import TransformMethod

from enum import StrEnum
from typing import TypedDict, Any, Tuple
from abc import ABC, abstractmethod
from numpy import float64, exp
from numpy.typing import NDArray


class PriorFamily(StrEnum):
    NORMAL = "normal"
    LOG_NORMAL = "lognormal"
    TRUNC_NORMAL = "truncated_normal"
    HALF_NORMAL = "half_normal"
    HALF_CAUCHY = "half_cauchy"
    BETA = "beta"
    GAMMA = "gamma"
    INV_GAMMA = "invgamma"
    LKJ_CHOL = "lkj_chol"
    UNIFORM = "uniform"  # Bounded max entropy fallback


class PriorDispatch(TypedDict):
    family: PriorFamily
    parameters: dict[str, Any]  # TypedDicts for each family
    transform: TransformMethod
    transform_kwargs: dict[str, Any]  # TypedDicts for each transform
