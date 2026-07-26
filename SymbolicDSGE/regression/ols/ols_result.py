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
    """Across-replication regression summary, held as traces.

    Carries only the sufficient statistics of each replication's solve, all
    ``O(k)`` or ``O(1)`` per replication: the coefficient, standard-error, SSR,
    SST, and status traces. The per-replication ``RegressionResult`` objects and
    their ``y`` / ``X`` arrays are consumed by
    :class:`MCRegressionAccumulator` and released, so a run's retained memory
    does not scale with the regression sample size. Every reported quantity
    derives from the stored traces.
    """

    variables: list[str]
    coef_trace: NDF
    status_trace: tuple[RegressionStatus, ...]
    ssr_trace: NDF
    sst_trace: NDF
    n: int
    is_ols: bool = False
    #: OLS standard errors, computed per replication while the design was still
    #: live. ``None`` for kinds that do not report them; read via ``se_trace``.
    stored_se_trace: NDF | None = field(default=None, repr=False)

    n_rep: int = field(init=False)
    k: int = field(init=False)

    def __post_init__(self) -> None:
        variables = list(self.variables)
        coef_trace = np.ascontiguousarray(self.coef_trace, dtype=np.float64)
        if coef_trace.ndim != 2:
            raise ValueError("MC regression coef_trace must be a 2D array.")
        n_rep, k = coef_trace.shape
        if n_rep == 0:
            raise ValueError("MCRegressionResult requires at least one result.")
        if len(variables) != k:
            raise ValueError(
                "MC regression variables must match the number of coefficients."
            )
        status_trace = tuple(RegressionStatus(status) for status in self.status_trace)
        if len(status_trace) != n_rep:
            raise ValueError(
                "MC regression status trace must match the coefficient trace length."
            )
        ssr_trace = np.ascontiguousarray(self.ssr_trace, dtype=np.float64)
        sst_trace = np.ascontiguousarray(self.sst_trace, dtype=np.float64)
        if ssr_trace.shape != (n_rep,) or sst_trace.shape != (n_rep,):
            raise ValueError(
                "MC regression SSR and SST traces must be 1D and match the "
                "coefficient trace length."
            )
        se_trace = self.stored_se_trace
        if se_trace is not None:
            se_trace = np.ascontiguousarray(se_trace, dtype=np.float64)
            if se_trace.shape != (n_rep, k):
                raise ValueError(
                    "MC regression standard-error trace must match the coefficient "
                    "trace shape."
                )

        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "coef_trace", coef_trace)
        object.__setattr__(self, "status_trace", status_trace)
        object.__setattr__(self, "ssr_trace", ssr_trace)
        object.__setattr__(self, "sst_trace", sst_trace)
        object.__setattr__(self, "stored_se_trace", se_trace)
        object.__setattr__(self, "n", int(self.n))
        object.__setattr__(self, "is_ols", bool(self.is_ols))
        object.__setattr__(self, "n_rep", int(n_rep))
        object.__setattr__(self, "k", int(k))

    @classmethod
    def accumulator(cls, n_rep: int) -> "MCRegressionAccumulator":
        """A streaming builder that consumes results and releases them."""
        return MCRegressionAccumulator(n_rep)

    @classmethod
    def from_results(cls, results: Sequence[RegressionResult]) -> "MCRegressionResult":
        result_tuple = tuple(results)
        if not result_tuple:
            raise ValueError("MCRegressionResult requires at least one result.")
        accumulator = cls.accumulator(len(result_tuple))
        for result in result_tuple:
            accumulator.push(result)
        return accumulator.finalize()

    def _require_ols(self) -> None:
        if not self.is_ols:
            raise TypeError(
                "OLS-specific MC diagnostics require all results to be OLSResult."
            )

    @property
    def coefficients(self) -> NDF:
        return self.coef_trace

    @property
    def se_trace(self) -> NDF:
        self._require_ols()
        se_trace = self.stored_se_trace
        if se_trace is None:
            raise TypeError(
                "OLS-specific MC diagnostics require all results to be OLSResult."
            )
        return se_trace

    @cached_property
    def mse_trace(self) -> NDF:
        return np.asarray(self.ssr_trace / self.n, dtype=np.float64)

    @cached_property
    def rmse_trace(self) -> NDF:
        return np.asarray(np.sqrt(self.mse_trace), dtype=np.float64)

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
        self._require_ols()
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
        self._require_ols()
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
        if not self.is_ols:
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
        self._require_ols()
        dfn, dfd = _f_test_degrees_of_freedom(self.n, self.k, self.variables)
        return MCResult(
            test_name="F-test",
            dist=ReferenceDistribution.F,
            df=(float64(dfn), float64(dfd)),
            pval_method=PvalMethod.SF,
            alpha=float64(alpha),
            statistic_trace=self.F_stat_trace,
            status_trace=tuple(
                TestStatus.OK if status is RegressionStatus.OK else TestStatus.LINALG
                for status in self.status_trace
            ),
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


class MCRegressionAccumulator:
    """Streaming builder for an :class:`MCRegressionResult` over a replication loop.

    Extracts each replication's sufficient statistics (coefficients, status, SSR,
    SST, and OLS standard errors) at push time and releases the
    :class:`RegressionResult`, so neither the design matrix nor the response
    survives the replication. The standard errors are computed here rather than
    lazily, since the design they need is gone once the object is released.

    Buffers are sized on ``n_rep`` up front and filled at a cursor, so a run with
    skipped replications finalizes to the count actually pushed.
    """

    __slots__ = (
        "_coef",
        "_se",
        "_ssr",
        "_sst",
        "_status",
        "_variables",
        "_n",
        "_k",
        "_is_ols",
        "_cursor",
        "_capacity",
    )

    def __init__(self, n_rep: int) -> None:
        if n_rep <= 0:
            raise ValueError("MCRegressionAccumulator requires a positive n_rep.")
        self._capacity = int(n_rep)
        self._coef: NDF | None = None
        self._se: NDF | None = None
        self._ssr: NDF = np.empty(self._capacity, dtype=np.float64)
        self._sst: NDF = np.empty(self._capacity, dtype=np.float64)
        self._status: list[RegressionStatus] = []
        self._variables: list[str] | None = None
        self._n = -1
        self._k = -1
        self._is_ols = False
        self._cursor = 0

    @property
    def n_pushed(self) -> int:
        """Replications pushed so far."""
        return self._cursor

    def push(self, result: RegressionResult) -> None:
        """Record one replication's result, releasing the object to the caller."""
        if self._variables is None:
            self._variables = list(result.variables)
            self._n = result.n
            self._k = result.k
            self._is_ols = isinstance(result, OLSResult)
            self._coef = np.empty((self._capacity, self._k), dtype=np.float64)
            if self._is_ols:
                self._se = np.empty((self._capacity, self._k), dtype=np.float64)
        else:
            if result.variables != self._variables:
                raise ValueError("MC regression results have incompatible variables.")
            if result.n != self._n or result.k != self._k:
                raise ValueError("MC regression results have incompatible dimensions.")
            if self._is_ols != isinstance(result, OLSResult):
                raise ValueError("MC regression results have incompatible kinds.")
        if self._cursor >= self._capacity:
            raise ValueError(
                "MC regression results exceed the accumulator's capacity of "
                f"{self._capacity} replications."
            )

        assert self._coef is not None
        self._coef[self._cursor] = result.coefficients
        self._ssr[self._cursor] = result.ssr
        self._sst[self._cursor] = result.sst
        self._status.append(result.status)
        if self._se is not None:
            self._se[self._cursor] = cast(OLSResult, result).se
        self._cursor += 1

    def finalize(self) -> MCRegressionResult:
        """Build the summary over the replications pushed so far."""
        if self._variables is None or self._coef is None:
            raise ValueError("MCRegressionResult requires at least one result.")
        cursor = self._cursor
        return MCRegressionResult(
            variables=self._variables,
            coef_trace=self._coef[:cursor].copy(),
            status_trace=tuple(self._status),
            ssr_trace=self._ssr[:cursor].copy(),
            sst_trace=self._sst[:cursor].copy(),
            n=self._n,
            is_ols=self._is_ols,
            stored_se_trace=(None if self._se is None else self._se[:cursor].copy()),
        )
