from __future__ import annotations


from .diag_utils import se
from ..enums import RegressionStatus
from ..result import RegressionResult
from ..._diag_tests.result import MCTestResult, TestResult
from ..._diag_tests.status import TestStatus
from ..._diag_tests.distributions import (
    FloatScalar,
    ReferenceDistribution,
    PvalMethod,
)

from ...monte_carlo.spec import MCRegressionResultSpec, MCRegressionResultMeta


import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy import float64, asarray
from numpy.typing import NDArray
from scipy.stats import t
from functools import cached_property

from typing import TYPE_CHECKING, Sequence, cast

if TYPE_CHECKING:
    from pandas import DataFrame

NDF = NDArray[float64]


def _f_test_degrees_of_freedom(
    n: int,
    k: int,
    variables: Sequence[str],
) -> tuple[int, int]:
    has_intercept = len(variables) > 0 and variables[0] == "Intercept"
    if has_intercept:
        return k - 1, n - k
    return k, n - k - 1


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


@dataclass(frozen=True)
class MCRegressionResult:
    kind: str
    variables: Sequence[str]
    coef_trace: NDF

    ssr_trace: NDF
    sst_trace: NDF
    _se_trace: NDF | None

    n_retained: int
    retained_reps: NDArray[np.int_]
    n_rep: int
    n: int
    k: int

    _raw_status: NDArray[np.int_]

    @property
    def coefficients(self) -> NDF:
        return self.coef_trace

    @cached_property
    def se_trace(self) -> NDF:
        if self.kind == "ols" and self._se_trace is not None:
            return self._se_trace
        else:
            warnings.warn(
                "Non-OLS linear regressions don't have unbiased standard errors, `se_trace` is invalid.",
                UserWarning,
            )
            return np.full((self.n_retained, self.k), np.nan, dtype=np.float64)

    @cached_property
    def t_stat_trace(self) -> NDF:
        return np.asarray(self.coef_trace / self.se_trace, dtype=np.float64)

    @cached_property
    def mse_trace(self) -> NDF:
        return np.asarray(self.ssr_trace / self.n, dtype=np.float64)

    @cached_property
    def rmse_trace(self) -> NDF:
        return np.asarray(np.sqrt(self.mse_trace), dtype=np.float64)

    @cached_property
    def r2_trace(self) -> NDF:
        out = np.zeros(self.n_retained, dtype=np.float64)
        mask = self.sst_trace > 0
        out[mask] = 1 - self.ssr_trace[mask] / self.sst_trace[mask]
        return out

    @cached_property
    def r2_adj_trace(self) -> NDF:
        if self.n <= self.k + 1:
            return np.zeros(self.n_retained, dtype=np.float64)
        return np.asarray(
            1 - (1 - self.r2_trace) * (self.n - 1) / (self.n - self.k - 1),
            dtype=np.float64,
        )

    @cached_property
    def partial_r2_trace(self) -> NDF:
        t2 = self.t_stat_trace**2
        return np.asarray(t2 / (t2 + self.n - self.k), dtype=np.float64)

    @cached_property
    def pval_trace(self) -> NDF:
        df = self.n - self.k
        return np.asarray(2 * (1 - t.cdf(abs(self.t_stat_trace), df)), dtype=float64)

    @cached_property
    def F_stat_trace(self) -> NDF:
        if self.kind != "ols":
            return np.full(self.n_retained, np.nan, dtype=np.float64)
        dfn, dfd = _f_test_degrees_of_freedom(self.n, self.k, self.variables)
        num = self.r2_trace / dfn
        denom = (1 - self.r2_trace) / dfd
        return np.asarray(num / denom, dtype=np.float64)

    @cached_property
    def F_pval_trace(self) -> NDF:
        if self.kind != "ols":
            return np.full(self.n_retained, np.nan, dtype=np.float64)
        dfn, dfd = _f_test_degrees_of_freedom(self.n, self.k, self.variables)
        frozen = ReferenceDistribution.F.freeze(float64(dfn), float64(dfd))
        return np.asarray(PvalMethod.SF(frozen, self.F_stat_trace), dtype=np.float64)

    @cached_property
    def status_trace(self) -> tuple[RegressionStatus, ...]:
        return tuple(RegressionStatus(status) for status in self._raw_status)

    def confidence_intervals(self, alpha: FloatScalar = 0.05) -> NDF:
        q = 1 - alpha / 2
        df = self.n - self.k
        t_crit = t.ppf(q, df)
        lower_bound = self.coef_trace - t_crit * self.se_trace
        upper_bound = self.coef_trace + t_crit * self.se_trace
        return np.stack([lower_bound, upper_bound], axis=2).astype(np.float64)

    def summary(self, alpha: FloatScalar = 0.05) -> DataFrame:
        import pandas as pd

        index = pd.MultiIndex.from_product(
            [range(self.n_retained), self.variables], names=["retained_row", "variable"]
        )

        coef_ci = self.confidence_intervals(alpha)
        return pd.DataFrame(
            {
                "rep_idx": np.repeat(self.retained_reps, self.k),
                "coef": self.coef_trace.reshape(-1),
                "std_err": self.se_trace.reshape(-1),
                "coef_ci_low": coef_ci[:, :, 0].reshape(-1),
                "coef_ci_high": coef_ci[:, :, 1].reshape(-1),
                "t_stat": self.t_stat_trace.reshape(-1),
                "pval": self.pval_trace.reshape(-1),
                "partial_r2": self.partial_r2_trace.reshape(-1),
            },
            index=index,
        )

    def F_test(self, alpha: FloatScalar = 0.05) -> MCTestResult:
        dfn, dfd = _f_test_degrees_of_freedom(self.n, self.k, self.variables)
        return MCTestResult(
            test_name="F-test",
            dist=ReferenceDistribution.F,
            df=(float64(dfn), float64(dfd)),
            pval_method=PvalMethod.SF,
            alpha=float64(alpha),
            statistic_trace=self.F_stat_trace,
            n_retained=self.n_retained,
            retained_reps=self.retained_reps,
            n_rep=self.n_rep,
            _raw_status=np.asarray(
                [
                    (
                        TestStatus.OK
                        if status is RegressionStatus.OK
                        else TestStatus.LINALG
                    )
                    for status in self.status_trace
                ],
                dtype=np.int_,
            ),
        )

    def to_spec(self) -> MCRegressionResultSpec:
        meta = MCRegressionResultMeta(
            kind=self.kind,
            variables=list(self.variables),
            n_retained=self.n_retained,
            n_rep=self.n_rep,
            n=self.n,
            k=self.k,
        )
        return MCRegressionResultSpec(
            meta=meta,
            coef_trace=self.coef_trace,
            ssr_trace=self.ssr_trace,
            sst_trace=self.sst_trace,
            retained_reps=self.retained_reps,
            _raw_status=self._raw_status,
            _se_trace=self._se_trace,
        )

    @classmethod
    def from_spec(cls, spec: MCRegressionResultSpec) -> MCRegressionResult:
        meta = spec.meta
        return cls(
            kind=meta["kind"],
            variables=meta["variables"],
            n_retained=meta["n_retained"],
            n_rep=meta["n_rep"],
            n=meta["n"],
            k=meta["k"],
            coef_trace=spec.coef_trace,
            ssr_trace=spec.ssr_trace,
            sst_trace=spec.sst_trace,
            retained_reps=spec.retained_reps,
            _raw_status=spec._raw_status,
            _se_trace=spec._se_trace,
        )
