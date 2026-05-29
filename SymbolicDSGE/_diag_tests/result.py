from dataclasses import dataclass, field

import numpy as np
from numpy import float64, sqrt
from numpy.typing import NDArray

from .distributions import (
    FloatScalar,
    FrozenDistribution,
    PvalMethod,
    ReferenceDistribution,
)


def _compute_pvalues(
    dist: ReferenceDistribution,
    df: FloatScalar,
    pval_method: PvalMethod,
    statistic: FloatScalar | NDArray[float64],
) -> tuple[FrozenDistribution, NDArray[float64]]:
    if not isinstance(dist, ReferenceDistribution):
        raise ValueError(f"Unsupported reference distribution: {dist}")

    frozen_dist = dist.freeze(df)
    pvals = np.asarray(pval_method(frozen_dist, statistic), dtype=np.float64)
    return frozen_dist, pvals


@dataclass(frozen=True)
class TestResult:
    test_name: str
    dist: ReferenceDistribution
    df: float64
    pval_method: PvalMethod
    alpha: float64
    statistic: float64
    _auto_pval: bool = field(default=True, repr=False, compare=False)

    _frozen_dist: FrozenDistribution | None = field(
        init=False, default=None, repr=False, compare=False
    )
    _pval: float64 | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        statistic = float64(self.statistic)
        object.__setattr__(self, "statistic", statistic)

        if self._auto_pval:
            self.compute_pval()

    @property
    def frozen_dist(self) -> FrozenDistribution:
        if self._frozen_dist is None:
            self.compute_pval()
        frozen_dist = self._frozen_dist
        if frozen_dist is None:
            raise RuntimeError("p-value computation failed to freeze distribution")
        return frozen_dist

    @property
    def pval(self) -> float64:
        if self._pval is None:
            return self.compute_pval()
        return self._pval

    def compute_pval(self) -> float64:
        if self._pval is not None:
            return self._pval

        frozen_dist, pval = _compute_pvalues(
            self.dist,
            self.df,
            self.pval_method,
            self.statistic,
        )
        if pval.shape != ():
            raise ValueError("scalar statistic must produce a scalar p-value")

        object.__setattr__(
            self,
            "_frozen_dist",
            frozen_dist,
        )
        pval_scalar = float64(pval.item())
        object.__setattr__(
            self,
            "_pval",
            pval_scalar,
        )
        return pval_scalar

    def is_significant(self, threshold: float | float64 | None = None) -> bool:
        if threshold is None:
            threshold = self.alpha
        return bool(self.pval < threshold)

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "dist": self.dist.value,
            "df": self.df,
            "pval_method": self.pval_method.value,
            "alpha": self.alpha,
            "statistic": self.statistic,
            "pval": self.pval,
        }


@dataclass(frozen=True)
class MCResult:
    test_name: str
    dist: ReferenceDistribution
    df: float64
    pval_method: PvalMethod
    alpha: float64
    statistic_trace: NDArray[float64]

    frozen_dist: FrozenDistribution = field(init=False, repr=False)
    pval_trace: NDArray[float64] = field(init=False)
    n: int = field(init=False)
    mean_statistic: float64 = field(init=False)
    mean_pval: float64 = field(init=False)
    rejection_rate: float64 = field(init=False)
    pval_se: float64 = field(init=False)
    statistic_se: float64 = field(init=False)

    def __post_init__(self) -> None:
        statistic_trace = np.asarray(self.statistic_trace, dtype=np.float64)
        if statistic_trace.ndim != 1:
            raise ValueError("statistic_trace must be a 1D array")

        n = int(statistic_trace.size)
        if n == 0:
            raise ValueError("statistic_trace must be non-empty")

        frozen_dist, pval_trace = _compute_pvalues(
            self.dist,
            self.df,
            self.pval_method,
            statistic_trace,
        )
        if statistic_trace.shape != pval_trace.shape:
            raise ValueError("statistic_trace and pval_trace must have the same shape")

        object.__setattr__(
            self,
            "statistic_trace",
            statistic_trace,
        )
        object.__setattr__(
            self,
            "frozen_dist",
            frozen_dist,
        )
        object.__setattr__(
            self,
            "pval_trace",
            pval_trace,
        )

        object.__setattr__(
            self,
            "n",
            n,
        )
        object.__setattr__(
            self,
            "mean_statistic",
            self.statistic_trace.mean(),
        )

        object.__setattr__(
            self,
            "mean_pval",
            self.pval_trace.mean(),
        )
        object.__setattr__(
            self,
            "rejection_rate",
            (self.pval_trace < self.alpha).mean(),
        )
        object.__setattr__(
            self,
            "pval_se",
            ((self.rejection_rate * (1 - self.rejection_rate)) / self.n) ** 0.5,
        )
        object.__setattr__(
            self,
            "statistic_se",
            self.statistic_trace.std(ddof=1) / sqrt(self.n),
        )

    def pval_confidence_interval(
        self,
        confidence_level: float | float64 = 0.95,
        wilson: bool = True,
    ) -> tuple[float64, float64]:

        from scipy.stats import norm

        z = norm.ppf(1 - (1 - confidence_level) / 2)
        p = self.rejection_rate

        if wilson:
            q = 1 - p
            n = self.n

            center = (p + (z**2) / (2 * n)) / (1 + (z**2) / n)
            spread = z * sqrt((p * q) / n + (z**2) / (4 * n**2)) / (1 + (z**2) / n)
            return float64(max(0, center - spread)), float64(min(1, center + spread))
        else:
            se = self.pval_se
            return p - z * se, p + z * se

    def statistic_confidence_interval(
        self, confidence_level: float | float64 = 0.95, t_interval: bool = False
    ) -> tuple[float64, float64]:
        if t_interval:
            from scipy.stats import t

            df = self.n - 1
            z = t.ppf(1 - (1 - confidence_level) / 2, df)
        else:
            from scipy.stats import norm

            z = norm.ppf(1 - (1 - confidence_level) / 2)

        se = self.statistic_se
        return self.mean_statistic - z * se, self.mean_statistic + z * se
