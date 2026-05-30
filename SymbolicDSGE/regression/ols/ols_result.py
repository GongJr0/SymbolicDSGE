from __future__ import annotations

from SymbolicDSGE._diag_tests.distributions import (
    FloatScalar,
    ReferenceDistribution,
    PvalMethod,
)
from SymbolicDSGE.regression.ols.diag_utils import r2, r2_adj, se
from ..._diag_tests.result import MCResult, TestResult

from dataclasses import dataclass, field, asdict

import numpy as np
from numpy import float64, asarray
from numpy.typing import NDArray
from scipy.stats import t
from functools import cached_property

from typing import TYPE_CHECKING, Sequence
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


@dataclass(frozen=True)
class MCRegressionResult:
    variables: list[str]
    results: tuple[OLSResult, ...] = field(repr=False)

    coef_trace: NDF = field(init=False)
    status_trace: tuple[Status, ...] = field(init=False)
    n_rep: int = field(init=False)
    n: int = field(init=False)
    k: int = field(init=False)

    def __post_init__(self) -> None:
        results = tuple(self.results)
        if not results:
            raise ValueError("MCRegressionResult requires at least one OLSResult.")

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
        status_trace: list[Status] = []
        for result in results:
            if result.variables != variables:
                raise ValueError("MC regression results have incompatible variables.")
            if result.n != n or result.k != k:
                raise ValueError("MC regression results have incompatible dimensions.")
            if result.y.shape != (n,):
                raise ValueError("MC regression results require 1D response arrays.")
            if result.x.shape != (n, k):
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
    def from_results(cls, results: Sequence[OLSResult]) -> "MCRegressionResult":
        result_tuple = tuple(results)
        if not result_tuple:
            raise ValueError("MCRegressionResult requires at least one OLSResult.")
        return cls(variables=list(result_tuple[0].variables), results=result_tuple)

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
            np.stack([result.x for result in self.results]),
            dtype=np.float64,
        )

    @cached_property
    def y_hat_trace(self) -> NDF:
        return np.asarray(
            np.einsum("rnk,rk->rn", self.x_trace, self.coef_trace),
            dtype=np.float64,
        )

    @cached_property
    def se_trace(self) -> NDF:
        if all(result._L.shape == (self.k, self.k) for result in self.results):
            eps = self.y_trace - self.y_hat_trace
            sigma2 = (eps**2).sum(axis=1) / (self.n - self.k)
            eye = np.eye(self.k, dtype=np.float64)
            rhs = np.broadcast_to(eye, (self.n_rep, self.k, self.k))
            L_trace = np.stack([result._L for result in self.results])
            L_inv = np.linalg.solve(L_trace, rhs)
            inv_diag = (L_inv * L_inv).sum(axis=1)
            return np.asarray(np.sqrt(inv_diag * sigma2[:, None]), dtype=np.float64)

        return np.asarray(
            np.vstack([result.se for result in self.results]), dtype=np.float64
        )

    @cached_property
    def t_stat_trace(self) -> NDF:
        return np.asarray(self.coef_trace / self.se_trace, dtype=np.float64)

    @cached_property
    def r2_trace(self) -> NDF:
        eps = self.y_trace - self.y_hat_trace
        ssr = (eps**2).sum(axis=1)
        centered = self.y_trace - self.y_trace.mean(axis=1, keepdims=True)
        sst = (centered**2).sum(axis=1)
        out = np.zeros(self.n_rep, dtype=np.float64)
        mask = sst > 0
        out[mask] = 1 - ssr[mask] / sst[mask]
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
        dfn = self.k
        dfd = self.n - self.k - 1
        num = self.r2_trace / dfn
        denom = (1 - self.r2_trace) / dfd
        return np.asarray(num / denom, dtype=np.float64)

    @cached_property
    def F_pval_trace(self) -> NDF:
        frozen = ReferenceDistribution.F.freeze(
            float64(self.k), float64(self.n - self.k - 1)
        )
        return np.asarray(PvalMethod.SF(frozen, self.F_stat_trace), dtype=np.float64)

    def confidence_intervals(self, alpha: FloatScalar = 0.05) -> NDF:
        q = 1 - alpha / 2
        df = self.n - self.k
        t_crit = t.ppf(q, df)
        lower_bound = self.coef_trace - t_crit * self.se_trace
        upper_bound = self.coef_trace + t_crit * self.se_trace
        return np.stack([lower_bound, upper_bound], axis=2).astype(np.float64)

    def summary(self, alpha: FloatScalar = 0.05) -> DataFrame:
        import pandas as pd

        coef_ci = self.confidence_intervals(alpha)
        index = pd.MultiIndex.from_product(
            [range(self.n_rep), self.variables], names=["rep", "variable"]
        )
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
        return MCResult(
            test_name="F-test",
            dist=ReferenceDistribution.F,
            df=(float64(self.k), float64(self.n - self.k - 1)),
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
