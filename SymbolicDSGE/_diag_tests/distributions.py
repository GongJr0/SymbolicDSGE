from __future__ import annotations

from enum import Enum
from typing import Any, TypeAlias

import numpy as np
from numpy import float64, floating, integer
from numpy.typing import NDArray
from scipy.stats._distn_infrastructure import rv_frozen

FloatScalar: TypeAlias = float | floating[Any]
IntegerScalar: TypeAlias = int | integer[Any]
DistributionParameter: TypeAlias = FloatScalar | IntegerScalar
FrozenDistribution: TypeAlias = rv_frozen[Any, Any]


def cdf_pval(
    dist: FrozenDistribution,
    stat: FloatScalar | NDArray[float64],
) -> float64 | NDArray[float64]:
    return dist.cdf(stat)


def sf_pval(
    dist: FrozenDistribution,
    stat: FloatScalar | NDArray[float64],
) -> float64 | NDArray[float64]:
    return dist.sf(stat)


class PvalMethod(Enum):
    CDF = "cdf"
    SF = "sf"

    def __call__(
        self,
        dist: FrozenDistribution,
        stat: FloatScalar | NDArray[float64],
    ) -> float64 | NDArray[float64]:
        if self is PvalMethod.CDF:
            return cdf_pval(dist, stat)
        if self is PvalMethod.SF:
            return sf_pval(dist, stat)
        raise ValueError(f"Unsupported p-value method: {self}")


class ReferenceDistribution(Enum):
    CHI2 = "chi2"
    F = "f"
    t = "t"
    JB_LOOKUP = "jb_lookup"
    CUSUM = "cusum"
    CUSUMSQ = "cusumsq"

    def freeze(self, *df: DistributionParameter) -> FrozenDistribution:
        match self:
            case ReferenceDistribution.CHI2:
                from scipy.stats import chi2

                return chi2(*df)
            case ReferenceDistribution.F:
                from scipy.stats import f

                return f(*df)
            case ReferenceDistribution.t:
                from scipy.stats import t

                return t(*df)
            case ReferenceDistribution.JB_LOOKUP:
                from .jb_lookup import JarqueBeraDist

                if len(df) != 1:
                    raise TypeError(
                        "JB_LOOKUP requires exactly one sample-size argument"
                    )
                n = df[0]
                if isinstance(n, bool | np.bool_) or not isinstance(n, int | integer):
                    raise TypeError("JB_LOOKUP sample size must be an integer")
                return JarqueBeraDist(n)
            case ReferenceDistribution.CUSUM:
                from .cusum import CusumDist

                # CUSUM is parameter-free; any df forwarded by TestResult
                # (a NaN placeholder) is ignored.
                return CusumDist()
            case ReferenceDistribution.CUSUMSQ:
                from .cusumsq import CusumSq

                if len(df) != 1:
                    raise TypeError("CUSUMSQ requires exactly one sample-size argument")
                n = df[0]
                if isinstance(n, bool | np.bool_) or not isinstance(n, int | integer):
                    raise TypeError("CUSUMSQ sample size must be an integer")
                return CusumSq(n)
            case _:
                raise ValueError(f"Unsupported reference distribution: {self}")
