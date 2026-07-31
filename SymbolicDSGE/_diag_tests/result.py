from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from numpy import float64, sqrt
from numpy.typing import NDArray

from .distributions import (
    DistributionParameter,
    FrozenDistribution,
    PvalMethod,
    ReferenceDistribution,
)
from .status import TestStatus


@dataclass(frozen=True)
class TestResult:
    test_name: str
    dist: ReferenceDistribution
    df: DistributionParameter | Sequence[DistributionParameter]
    pval_method: PvalMethod
    alpha: float64
    statistic: float64
    status: TestStatus
    _auto_pval: bool = field(default=True, repr=False, compare=False)

    _pval: float64 | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", TestStatus(self.status))

        if self._auto_pval:
            self.compute_pval()

    @cached_property
    def frozen_dist(self) -> FrozenDistribution:
        df_args = self.df if isinstance(self.df, Sequence) else (self.df,)
        return self.dist.freeze(*df_args)

    @cached_property
    def pval(self) -> float64:
        return float64(self.pval_method(self.frozen_dist, self.statistic))

    def compute_pval(self) -> float64:
        if self._pval is not None:
            return self._pval

        pval = np.asarray(
            self.pval_method(self.frozen_dist, self.statistic),
            dtype=np.float64,
        )
        if pval.shape != ():
            raise ValueError("scalar statistic must produce a scalar p-value")

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
            "status": self.status,
            "pval": self.pval,
        }


@dataclass(frozen=True)
class MCResult:
    test_name: str
    dist: ReferenceDistribution
    df: DistributionParameter | Sequence[DistributionParameter]
    pval_method: PvalMethod
    alpha: float64
    statistic_trace: NDArray[float64]

    n_retained: int
    retained_reps: NDArray[np.int_]
    n_rep: int

    _raw_status: NDArray[np.int_]

    @cached_property
    def frozen_dist(self) -> FrozenDistribution:
        df_args = self.df if isinstance(self.df, Sequence) else (self.df,)
        return self.dist.freeze(*df_args)

    @cached_property
    def pval_trace(self) -> NDArray[float64]:
        return np.asarray(
            [self.pval_method(self.frozen_dist, stat) for stat in self.statistic_trace],
            dtype=np.float64,
        )

    @cached_property
    def status_trace(self) -> tuple[TestStatus, ...]:
        return tuple(TestStatus(status) for status in self._raw_status)

    @cached_property
    def mean_statistic(self) -> float64:
        return float64(self.statistic_trace.mean())

    @cached_property
    def mean_pval(self) -> float64:
        return float64(self.pval_trace.mean())

    @cached_property
    def rejection_rate(self) -> float64:
        return float64((self.pval_trace < self.alpha).mean())

    @cached_property
    def pval_se(self) -> float64:
        return float64(
            ((self.rejection_rate * (1 - self.rejection_rate)) / self.n_retained) ** 0.5
        )

    @cached_property
    def statistic_se(self) -> float64:
        return float64(self.statistic_trace.std(ddof=1) / sqrt(self.n_retained))

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
            n = self.n_retained

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

            df = self.n_retained - 1
            z = t.ppf(1 - (1 - confidence_level) / 2, df)
        else:
            from scipy.stats import norm

            z = norm.ppf(1 - (1 - confidence_level) / 2)

        se = self.statistic_se
        return self.mean_statistic - z * se, self.mean_statistic + z * se
