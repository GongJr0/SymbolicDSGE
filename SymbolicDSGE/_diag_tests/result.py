from dataclasses import dataclass, asdict, field

import numpy as np
from numpy import float64, sqrt
from numpy.typing import NDArray

from .distributions import FrozenDistribution, PvalMethod, ReferenceDistribution


@dataclass(frozen=True)
class TestResult:
    test_name: str
    alpha: float64
    statistic: float64
    pval: float64

    def is_significant(self, threshold: float | float64 | None = None) -> bool:
        if threshold is None:
            threshold = self.alpha
        return bool(self.pval < threshold)

    def to_dict(self) -> dict:
        return asdict(self)


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

        if not isinstance(self.dist, ReferenceDistribution):
            raise ValueError(f"Unsupported reference distribution: {self.dist}")

        frozen_dist = self.dist.freeze(self.df)
        pval_trace = np.asarray(
            self.pval_method(frozen_dist, statistic_trace), dtype=np.float64
        )
        if statistic_trace.shape != pval_trace.shape:
            raise ValueError("statistic_trace and pval_trace must have the same shape")

        n = int(self.statistic_trace.size)
        if n == 0:
            raise ValueError("statistic_trace and pval_trace must be non-empty")

        object.__setattr__(
            self,
            "n",
            n,
        )
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
