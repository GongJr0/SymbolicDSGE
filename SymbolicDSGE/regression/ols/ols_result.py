from __future__ import annotations


from .diag_utils import se
from ..result import RegressionResult, _f_test_degrees_of_freedom
from ..._diag_tests.result import TestResult
from ..._diag_tests.status import TestStatus
from ..._diag_tests.distributions import (
    FloatScalar,
    ReferenceDistribution,
    PvalMethod,
)


from dataclasses import dataclass, field

from numpy import float64, asarray
from numpy.typing import NDArray
from scipy.stats import t
from functools import cached_property

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame

NDF = NDArray[float64]


@dataclass(frozen=True)
class OLSResult(RegressionResult):
    _L: NDF = field(repr=False)

    @cached_property
    def se(self) -> NDF:
        return se(self._L, self.y, self.y_hat, self.X)

    @cached_property
    def t_stat(self) -> NDF:
        return self.coefficients / self.se

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

        dfn, dfd = _f_test_degrees_of_freedom(n, k, self.variables)

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
            status=TestStatus.OK,
            _auto_pval=True,
        )
