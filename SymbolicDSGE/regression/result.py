from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from functools import cached_property
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy.stats import norm, t

from .enums import RegressionStatus
from .._diag_tests.distributions import (
    FloatScalar,
    PvalMethod,
    ReferenceDistribution,
)
from .._diag_tests.result import MCTestResult
from .._diag_tests.status import TestStatus
from ..monte_carlo.spec import MCRegressionResultMeta, MCRegressionResultSpec

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


def _clamp_unit(bounds: tuple[NDF, NDF]) -> tuple[NDF, NDF]:
    """Interval bounds held inside ``[0, 1]``, for quantities that live there."""
    low, high = bounds
    return np.clip(low, 0.0, 1.0), np.clip(high, 0.0, 1.0)


def r2(y: NDF, y_hat: NDF) -> float64:
    ssr = ((y - y_hat) ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    return float64(1 - ssr / sst) if sst > 0 else float64(0.0)


def r2_adj(r2_value: float64, n: int, k: int) -> float64:
    if n <= k + 1:
        return float64(0.0)
    return float64(1 - (1 - r2_value) * (n - 1) / (n - k - 1))


@dataclass(frozen=True)
class RegressionResult:
    variables: list[str]
    coefficients: NDF

    # Raw Data
    y: NDF
    X: NDF
    n: int = field(init=False)
    k: int = field(init=False)

    # Meta
    status: RegressionStatus

    def __post_init__(self) -> None:
        y = np.ascontiguousarray(self.y, dtype=np.float64)
        x = np.ascontiguousarray(self.X, dtype=np.float64)
        coefficients = np.ascontiguousarray(self.coefficients, dtype=np.float64)
        variables = list(self.variables)

        if y.ndim != 1:
            raise ValueError("Regression response must be a 1D array.")
        if x.ndim != 2:
            raise ValueError("Regression design matrix must be a 2D array.")
        if coefficients.ndim != 1:
            raise ValueError("Regression coefficients must be a 1D array.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Regression response and design matrix row counts differ.")
        if coefficients.shape != (x.shape[1],):
            raise ValueError("Regression coefficient count must match design columns.")
        if len(variables) != x.shape[1]:
            raise ValueError("Regression variables must match design columns.")

        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "X", x)
        object.__setattr__(self, "n", y.shape[0])
        object.__setattr__(self, "k", x.shape[1])
        object.__setattr__(self, "status", RegressionStatus(self.status))

    @cached_property
    def y_hat(self) -> NDF:
        return np.asarray(self.X @ self.coefficients, dtype=np.float64)

    @property
    def x(self) -> NDF:
        return self.X

    @cached_property
    def residuals(self) -> NDF:
        return np.asarray(self.y - self.y_hat, dtype=np.float64)

    @cached_property
    def ssr(self) -> float64:
        return float64((self.residuals**2).sum())

    @cached_property
    def sst(self) -> float64:
        centered = self.y - self.y.mean()
        return float64((centered**2).sum())

    @cached_property
    def mse(self) -> float64:
        return float64(self.ssr / self.n)

    @cached_property
    def rmse(self) -> float64:
        return float64(np.sqrt(self.mse))

    @cached_property
    def r2(self) -> float64:
        return r2(self.y, self.y_hat)

    @cached_property
    def r2_adj(self) -> float64:
        return r2_adj(self.r2, self.n, self.k)

    def to_dict(self) -> dict:
        return asdict(self)


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

    def _mc_se(self, trace: NDF) -> NDF:
        """Monte Carlo standard error of a trace's mean, per coefficient."""
        if self.n_retained < 2:
            return np.full(self.k, np.nan, dtype=np.float64)
        return np.asarray(
            trace.std(ddof=1, axis=0) / np.sqrt(self.n_retained), dtype=np.float64
        )

    @cached_property
    def mean_coef(self) -> NDF:
        return np.asarray(self.coef_trace.mean(axis=0), dtype=np.float64)

    @cached_property
    def coef_se(self) -> NDF:
        """Spread of :attr:`mean_coef` across replications, not a per-fit
        standard error. The per-fit ones are :attr:`se_trace`."""
        return self._mc_se(self.coef_trace)

    @cached_property
    def mean_t_stat(self) -> NDF:
        return np.asarray(self.t_stat_trace.mean(axis=0), dtype=np.float64)

    @cached_property
    def t_stat_se(self) -> NDF:
        return self._mc_se(self.t_stat_trace)

    @cached_property
    def mean_pval(self) -> NDF:
        return np.asarray(self.pval_trace.mean(axis=0), dtype=np.float64)

    @cached_property
    def pval_se(self) -> NDF:
        return self._mc_se(self.pval_trace)

    def rejection_rate(self, alpha: FloatScalar = 0.05) -> NDF:
        """Share of retained replications rejecting at ``alpha``, per coefficient."""
        return np.asarray((self.pval_trace < alpha).mean(axis=0), dtype=np.float64)

    def rejection_rate_se(self, alpha: FloatScalar = 0.05) -> NDF:
        p = self.rejection_rate(alpha)
        return np.asarray(np.sqrt(p * (1 - p) / self.n_retained), dtype=np.float64)

    def _mean_interval(
        self, mean: NDF, se: NDF, confidence_level: float | float64, t_interval: bool
    ) -> tuple[NDF, NDF]:
        if t_interval:
            crit = t.ppf(1 - (1 - confidence_level) / 2, self.n_retained - 1)
        else:
            crit = norm.ppf(1 - (1 - confidence_level) / 2)
        return mean - crit * se, mean + crit * se

    def _wilson_interval(
        self, p: NDF, confidence_level: float | float64
    ) -> tuple[NDF, NDF]:
        z = norm.ppf(1 - (1 - confidence_level) / 2)
        n = self.n_retained
        center = (p + (z**2) / (2 * n)) / (1 + (z**2) / n)
        spread = z * np.sqrt((p * (1 - p)) / n + (z**2) / (4 * n**2)) / (1 + (z**2) / n)
        return np.maximum(0.0, center - spread), np.minimum(1.0, center + spread)

    def summary(self, alpha: FloatScalar = 0.05) -> DataFrame:
        """One row per coefficient, aggregated over the retained replications.

        ``coef_se`` is the Monte Carlo standard error of ``coef``, and
        ``reject_rate`` the share of replications rejecting at ``alpha``.
        Interval bounds are :meth:`intervals`; the per-replication table is
        :meth:`trace_frame`.
        """
        import pandas as pd

        return pd.DataFrame(
            {
                "coef": self.mean_coef,
                "coef_se": self.coef_se,
                "t_stat": self.mean_t_stat,
                "pval": self.mean_pval,
                "reject_rate": self.rejection_rate(alpha),
            },
            index=pd.Index(list(self.variables), name="variable"),
        )

    def intervals(
        self,
        alpha: FloatScalar = 0.05,
        confidence_level: float | float64 = 0.95,
        t_interval: bool = False,
        wilson: bool = True,
    ) -> DataFrame:
        """Interval bounds for every quantity :meth:`summary` reports.

        ``coef``, ``t_stat``, and ``pval`` come off the trace spread, normal by
        default and Student-t under ``t_interval``. ``reject_rate`` is a
        proportion, so it takes a Wilson interval unless ``wilson`` is off.
        ``pval`` and ``reject_rate`` are bounded, so their bounds clamp to
        ``[0, 1]``.
        """
        import pandas as pd

        rate = self.rejection_rate(alpha)
        blocks = {
            "coef": self._mean_interval(
                self.mean_coef, self.coef_se, confidence_level, t_interval
            ),
            "t_stat": self._mean_interval(
                self.mean_t_stat, self.t_stat_se, confidence_level, t_interval
            ),
            "pval": _clamp_unit(
                self._mean_interval(
                    self.mean_pval, self.pval_se, confidence_level, t_interval
                )
            ),
            "reject_rate": (
                self._wilson_interval(rate, confidence_level)
                if wilson
                else _clamp_unit(
                    self._mean_interval(
                        rate,
                        self.rejection_rate_se(alpha),
                        confidence_level,
                        t_interval,
                    )
                )
            ),
        }

        index = pd.MultiIndex.from_product(
            [list(self.variables), list(blocks)], names=["variable", "quantity"]
        )
        low = np.stack([blocks[q][0] for q in blocks], axis=1).reshape(-1)
        high = np.stack([blocks[q][1] for q in blocks], axis=1).reshape(-1)
        return pd.DataFrame({"ci_low": low, "ci_high": high}, index=index)

    def trace_frame(self, alpha: FloatScalar = 0.05) -> DataFrame:
        """The retained replications in long form, one row per replication and
        coefficient."""
        import pandas as pd

        index = pd.MultiIndex.from_product(
            [range(self.n_retained), self.variables], names=["retained_row", "variable"]
        )

        # Each row's own sampling interval, from the n observations that fit it.
        t_crit = t.ppf(1 - alpha / 2, self.n - self.k)
        half_width = t_crit * self.se_trace
        return pd.DataFrame(
            {
                "rep_idx": np.repeat(self.retained_reps, self.k),
                "coef": self.coef_trace.reshape(-1),
                "std_err": self.se_trace.reshape(-1),
                "coef_ci_low": (self.coef_trace - half_width).reshape(-1),
                "coef_ci_high": (self.coef_trace + half_width).reshape(-1),
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
