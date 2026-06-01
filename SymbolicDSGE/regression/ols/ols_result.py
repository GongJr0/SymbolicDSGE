from __future__ import annotations

from SymbolicDSGE._diag_tests.distributions import (
    FloatScalar,
    ReferenceDistribution,
    PvalMethod,
)
from .diag_utils import se
from ..enums import RegressionStatus
from ..result import RegressionResult
from ..._diag_tests.result import MCResult, TestResult
from ..._diag_tests.status import TestStatus

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
    variables: list[str]
    results: tuple[RegressionResult, ...] = field(repr=False)

    coef_trace: NDF = field(init=False)
    status_trace: tuple[RegressionStatus, ...] = field(init=False)
    n_rep: int = field(init=False)
    n: int = field(init=False)
    k: int = field(init=False)

    def __post_init__(self) -> None:
        results = tuple(self.results)
        if not results:
            raise ValueError("MCRegressionResult requires at least one result.")

        first = results[0]
        variables = list(self.variables)
        if variables != first.variables:
            raise ValueError("MC regression variables must match the first result.")

        n = first.n
        k = first.k
        if len(variables) != k:
            raise ValueError(
                "MC regression variables must match the number of coefficients."
            )

        coef_trace: list[NDF] = []
        status_trace: list[RegressionStatus] = []
        for result in results:
            if result.variables != variables:
                raise ValueError("MC regression results have incompatible variables.")
            if result.n != n or result.k != k:
                raise ValueError("MC regression results have incompatible dimensions.")
            if result.y.shape != (n,):
                raise ValueError("MC regression results require 1D response arrays.")
            if result.X.shape != (n, k):
                raise ValueError("MC regression results have incompatible designs.")
            if result.coefficients.shape != (k,):
                raise ValueError(
                    "MC regression results have incompatible coefficient shapes."
                )
            coef_trace.append(result.coefficients)
            status_trace.append(result.status)

        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "coef_trace", np.vstack(coef_trace))
        object.__setattr__(self, "status_trace", tuple(status_trace))
        object.__setattr__(self, "n_rep", len(self.results))
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "k", k)

    @classmethod
    def from_results(cls, results: Sequence[RegressionResult]) -> "MCRegressionResult":
        result_tuple = tuple(results)
        if not result_tuple:
            raise ValueError("MCRegressionResult requires at least one result.")
        return cls(variables=list(result_tuple[0].variables), results=result_tuple)

    def _require_ols_results(self) -> tuple[OLSResult, ...]:
        if not all(isinstance(result, OLSResult) for result in self.results):
            raise TypeError(
                "OLS-specific MC diagnostics require all results to be OLSResult."
            )
        return cast(tuple[OLSResult, ...], self.results)

    @property
    def coefficients(self) -> NDF:
        return self.coef_trace

    @cached_property
    def y_trace(self) -> NDF:
        return np.ascontiguousarray(
            np.stack([result.y for result in self.results]),
            dtype=np.float64,
        )

    @cached_property
    def x_trace(self) -> NDF:
        return np.ascontiguousarray(
            np.stack([result.X for result in self.results]),
            dtype=np.float64,
        )

    @cached_property
    def y_hat_trace(self) -> NDF:
        return np.asarray(
            np.einsum("rnk,rk->rn", self.x_trace, self.coef_trace),
            dtype=np.float64,
        )

    @cached_property
    def residual_trace(self) -> NDF:
        return np.asarray(self.y_trace - self.y_hat_trace, dtype=np.float64)

    @cached_property
    def ssr_trace(self) -> NDF:
        return np.asarray((self.residual_trace**2).sum(axis=1), dtype=np.float64)

    @cached_property
    def sst_trace(self) -> NDF:
        centered = self.y_trace - self.y_trace.mean(axis=1, keepdims=True)
        return np.asarray((centered**2).sum(axis=1), dtype=np.float64)

    @cached_property
    def mse_trace(self) -> NDF:
        return np.asarray(self.ssr_trace / self.n, dtype=np.float64)

    @cached_property
    def rmse_trace(self) -> NDF:
        return np.asarray(np.sqrt(self.mse_trace), dtype=np.float64)

    @cached_property
    def se_trace(self) -> NDF:
        results = self._require_ols_results()
        if all(result._L.shape == (self.k, self.k) for result in results):
            sigma2 = self.ssr_trace / (self.n - self.k)
            eye = np.eye(self.k, dtype=np.float64)
            rhs = np.broadcast_to(eye, (self.n_rep, self.k, self.k))
            L_trace = np.stack([result._L for result in results])
            L_inv = np.linalg.solve(L_trace, rhs)
            inv_diag = (L_inv * L_inv).sum(axis=1)
            return np.asarray(np.sqrt(inv_diag * sigma2[:, None]), dtype=np.float64)

        return np.asarray(
            np.vstack([result.se for result in results]), dtype=np.float64
        )

    @cached_property
    def t_stat_trace(self) -> NDF:
        return np.asarray(self.coef_trace / self.se_trace, dtype=np.float64)

    @cached_property
    def r2_trace(self) -> NDF:
        out = np.zeros(self.n_rep, dtype=np.float64)
        mask = self.sst_trace > 0
        out[mask] = 1 - self.ssr_trace[mask] / self.sst_trace[mask]
        return out

    @cached_property
    def r2_adj_trace(self) -> NDF:
        if self.n <= self.k + 1:
            return np.zeros(self.n_rep, dtype=np.float64)
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
        self._require_ols_results()
        dfn, dfd = _f_test_degrees_of_freedom(self.n, self.k, self.variables)
        num = self.r2_trace / dfn
        denom = (1 - self.r2_trace) / dfd
        return np.asarray(num / denom, dtype=np.float64)

    @cached_property
    def F_pval_trace(self) -> NDF:
        dfn, dfd = _f_test_degrees_of_freedom(self.n, self.k, self.variables)
        frozen = ReferenceDistribution.F.freeze(float64(dfn), float64(dfd))
        return np.asarray(PvalMethod.SF(frozen, self.F_stat_trace), dtype=np.float64)

    def confidence_intervals(self, alpha: FloatScalar = 0.05) -> NDF:
        self._require_ols_results()
        q = 1 - alpha / 2
        df = self.n - self.k
        t_crit = t.ppf(q, df)
        lower_bound = self.coef_trace - t_crit * self.se_trace
        upper_bound = self.coef_trace + t_crit * self.se_trace
        return np.stack([lower_bound, upper_bound], axis=2).astype(np.float64)

    def summary(self, alpha: FloatScalar = 0.05) -> DataFrame:
        import pandas as pd

        index = pd.MultiIndex.from_product(
            [range(self.n_rep), self.variables], names=["rep", "variable"]
        )
        if not all(isinstance(result, OLSResult) for result in self.results):
            return pd.DataFrame(
                {
                    "coef": self.coef_trace.reshape(-1),
                },
                index=index,
            )

        coef_ci = self.confidence_intervals(alpha)
        return pd.DataFrame(
            {
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

    def F_test(self, alpha: FloatScalar = 0.05) -> MCResult:
        self._require_ols_results()
        dfn, dfd = _f_test_degrees_of_freedom(self.n, self.k, self.variables)
        return MCResult(
            test_name="F-test",
            dist=ReferenceDistribution.F,
            df=(float64(dfn), float64(dfd)),
            pval_method=PvalMethod.SF,
            alpha=float64(alpha),
            statistic_trace=self.F_stat_trace,
        )

    def to_dict(self) -> dict:
        return {
            "variables": self.variables,
            "coef_trace": self.coef_trace,
            "status_trace": self.status_trace,
            "n_rep": self.n_rep,
            "n": self.n,
            "k": self.k,
        }
