from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy import float64, sqrt
from numpy.typing import NDArray
from scipy.stats import norm, t

from .distributions import (
    DistributionParameter,
    FrozenDistribution,
    PvalMethod,
    ReferenceDistribution,
)
from .status import TestStatus
from ..monte_carlo.spec import MCTestResultSpec, MCTestResultMeta

if TYPE_CHECKING:
    from pandas import DataFrame


@dataclass(frozen=True)
class TestResult:
    test_name: str
    dist: ReferenceDistribution
    df: DistributionParameter | Sequence[DistributionParameter]
    pval_method: PvalMethod
    alpha: float64 | float
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
class MCTestResult:
    test_name: str
    dist: ReferenceDistribution
    df: DistributionParameter | Sequence[DistributionParameter]
    pval_method: PvalMethod
    alpha: float64 | float
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

    def _mc_se(self, trace: NDArray[float64]) -> float64:
        """Monte Carlo standard error of a trace's mean."""
        if self.n_retained < 2:
            return float64(np.nan)
        return float64(trace.std(ddof=1) / sqrt(self.n_retained))

    @cached_property
    def pval_se(self) -> float64:
        """Spread of :attr:`mean_pval` across replications."""
        return self._mc_se(self.pval_trace)

    @cached_property
    def statistic_se(self) -> float64:
        return self._mc_se(self.statistic_trace)

    @cached_property
    def rejection_rate_se(self) -> float64:
        """Binomial standard error of :attr:`rejection_rate`."""
        p = self.rejection_rate
        return float64(((p * (1 - p)) / self.n_retained) ** 0.5)

    def _mean_interval(
        self,
        mean: float64,
        se: float64,
        confidence_level: float | float64,
        t_interval: bool,
    ) -> tuple[float64, float64]:
        if t_interval:
            crit = t.ppf(1 - (1 - confidence_level) / 2, self.n_retained - 1)
        else:
            crit = norm.ppf(1 - (1 - confidence_level) / 2)
        return float64(mean - crit * se), float64(mean + crit * se)

    def pval_confidence_interval(
        self,
        confidence_level: float | float64 = 0.95,
        t_interval: bool = False,
    ) -> tuple[float64, float64]:
        """Interval for :attr:`mean_pval`, off the spread of the p-value trace.

        Clamped to ``[0, 1]``. The rejection rate's own interval is
        :meth:`rejection_rate_confidence_interval`.
        """
        low, high = self._mean_interval(
            self.mean_pval, self.pval_se, confidence_level, t_interval
        )
        return float64(max(0.0, low)), float64(min(1.0, high))

    def rejection_rate_confidence_interval(
        self,
        confidence_level: float | float64 = 0.95,
        wilson: bool = True,
    ) -> tuple[float64, float64]:
        """Interval for :attr:`rejection_rate`, Wilson by default.

        The rate is a proportion, so the normal branch reads
        :attr:`rejection_rate_se`. Both are clamped to ``[0, 1]``.
        """
        z = norm.ppf(1 - (1 - confidence_level) / 2)
        p = self.rejection_rate

        if wilson:
            q = 1 - p
            n = self.n_retained

            center = (p + (z**2) / (2 * n)) / (1 + (z**2) / n)
            spread = z * sqrt((p * q) / n + (z**2) / (4 * n**2)) / (1 + (z**2) / n)
            return float64(max(0, center - spread)), float64(min(1, center + spread))

        se = self.rejection_rate_se
        return float64(max(0.0, p - z * se)), float64(min(1.0, p + z * se))

    def statistic_confidence_interval(
        self, confidence_level: float | float64 = 0.95, t_interval: bool = False
    ) -> tuple[float64, float64]:
        return self._mean_interval(
            self.mean_statistic, self.statistic_se, confidence_level, t_interval
        )

    def summary(self) -> DataFrame:
        """One row for this test, aggregated over the retained replications.

        ``reject_rate`` is the share of replications rejecting at :attr:`alpha`.
        Interval bounds are :meth:`intervals`.
        """
        import pandas as pd

        return pd.DataFrame(
            {
                "statistic": [self.mean_statistic],
                "statistic_se": [self.statistic_se],
                "pval": [self.mean_pval],
                "reject_rate": [self.rejection_rate],
            },
            index=pd.Index([self.test_name], name="test"),
        )

    def intervals(
        self,
        confidence_level: float | float64 = 0.95,
        t_interval: bool = False,
        wilson: bool = True,
    ) -> DataFrame:
        """Interval bounds for every quantity :meth:`summary` reports.

        ``statistic`` and ``pval`` come off the trace spread, normal by default
        and Student-t under ``t_interval``. ``reject_rate`` is a proportion, so
        it takes a Wilson interval unless ``wilson`` is off.
        """
        import pandas as pd

        blocks = {
            "statistic": self.statistic_confidence_interval(
                confidence_level, t_interval
            ),
            "pval": self.pval_confidence_interval(confidence_level, t_interval),
            "reject_rate": self.rejection_rate_confidence_interval(
                confidence_level, wilson
            ),
        }
        return pd.DataFrame(
            {
                "ci_low": [low for low, _ in blocks.values()],
                "ci_high": [high for _, high in blocks.values()],
            },
            index=pd.Index(list(blocks), name="quantity"),
        )

    def to_spec(self) -> MCTestResultSpec:
        meta = MCTestResultMeta(
            test_name=self.test_name,
            dist=self.dist.value,
            df=self.df,
            pval_method=self.pval_method.value,
            alpha=self.alpha,
            n_retained=self.n_retained,
            n_rep=self.n_rep,
        )
        return MCTestResultSpec(
            meta=meta,
            statistic_trace=self.statistic_trace,
            _raw_status=self._raw_status,
            retained_reps=self.retained_reps,
        )

    @classmethod
    def from_spec(cls, spec: MCTestResultSpec) -> "MCTestResult":
        meta = spec.meta
        return cls(
            test_name=meta["test_name"],
            dist=ReferenceDistribution(meta["dist"]),
            df=meta["df"],
            pval_method=PvalMethod(meta["pval_method"]),
            alpha=meta["alpha"],
            statistic_trace=spec.statistic_trace,
            n_retained=meta["n_retained"],
            retained_reps=spec.retained_reps,
            n_rep=meta["n_rep"],
            _raw_status=spec._raw_status,
        )
