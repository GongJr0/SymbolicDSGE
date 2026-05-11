from __future__ import annotations

from enum import Enum
from typing import Any

from numpy import float64, floating
from numpy.typing import NDArray
from scipy.stats._distn_infrastructure import rv_frozen

FloatScalar = float | floating[Any]
FrozenDistribution = rv_frozen[Any, Any]


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

    def freeze(self, df: FloatScalar) -> FrozenDistribution:
        from scipy.stats import chi2

        if self is ReferenceDistribution.CHI2:
            return chi2(df=df)
        raise ValueError(f"Unsupported reference distribution: {self}")
