from __future__ import annotations

from SymbolicDSGE._diag_tests.distributions import (
    FloatScalar,
    ReferenceDistribution,
    PvalMethod,
)
from SymbolicDSGE.regression.ols.diag_utils import r2, r2_adj, se
from ..._diag_tests.result import TestResult

from dataclasses import dataclass, field, asdict

from numpy import float64, asarray
from numpy.typing import NDArray
from scipy.stats import t
from functools import cached_property

from typing import TYPE_CHECKING
from enum import IntEnum

if TYPE_CHECKING:
    from pandas import DataFrame

NDF = NDArray[float64]


class Status(IntEnum):
    OK = 0
    RANK_DEFICIENT = -1


@dataclass(frozen=True)
class OLSResult:
    variables: list[str]
    coefficients: NDF

    # Raw Data
    y: NDF
    x: NDF
    n: int = field(init=False)
    k: int = field(init=False)

    # Meta
    status: Status
    _L: NDF = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n", self.y.shape[0])
        object.__setattr__(self, "k", self.x.shape[1])

    @cached_property
    def y_hat(self) -> NDF:
        return self.x @ self.coefficients

    @cached_property
    def se(self) -> NDF:
        return se(self._L, self.y, self.y_hat, self.x)

    @cached_property
    def t_stat(self) -> NDF:
        return self.coefficients / self.se

    @cached_property
    def r2(self) -> float64:
        return r2(self.y, self.y_hat)

    @cached_property
    def r2_adj(self) -> float64:
        return r2_adj(self.r2, self.n, self.k)

    @cached_property
    def partial_r2(self) -> NDF:
        return self.t_stat**2 / (self.t_stat**2 + self.n - self.k)

    @cached_property
    def p_values(self) -> NDF:
        df = self.n - self.k
        return 2 * (1 - t.cdf(abs(self.t_stat), df))

    def confidence_intervals(self, alpha: FloatScalar = 0.05) -> NDF:
        q = 1 - alpha / 2
        df = self.n - self.k
        t_crit = t.ppf(q, df)

        lower_bound = self.coefficients - t_crit * self.se
        upper_bound = self.coefficients + t_crit * self.se
        return asarray(list(zip(lower_bound, upper_bound)), dtype=float64)

    def summary(self, alpha: FloatScalar = 0.05) -> DataFrame:
        import pandas as pd

        coef_ci = self.confidence_intervals(alpha)
        pval = self.p_values
        partial_r2 = self.partial_r2

        summary_df = pd.DataFrame(
            {
                "coef": self.coefficients,
                "std_err": self.se,
                "coef_ci_low": coef_ci[:, 0],
                "coef_ci_high": coef_ci[:, 1],
                "t_stat": self.t_stat,
                "pval": pval,
                "partial_r2": partial_r2,
            },
            index=self.variables,
        )
        return summary_df

    def F_test(self, alpha: FloatScalar = 0.05) -> TestResult:
        r2 = self.r2
        n = self.n
        k = self.k

        dfn = k
        dfd = n - k - 1

        num = r2 / dfn
        denom = (1 - r2) / dfd
        F_stat = num / denom

        return TestResult(
            test_name="F-test",
            dist=ReferenceDistribution.F,
            df=(dfn, dfd),
            pval_method=PvalMethod.SF,
            alpha=float64(alpha),
            statistic=F_stat,
            _auto_pval=True,
        )

    def to_dict(self) -> dict:
        return asdict(self)
