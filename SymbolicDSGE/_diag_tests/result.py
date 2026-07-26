from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

import numpy as np
from numpy import float64, sqrt
from numpy.typing import NDArray

from .distributions import (
    DistributionParameter,
    FloatScalar,
    FrozenDistribution,
    PvalMethod,
    ReferenceDistribution,
)
from .status import TestStatus

DfSpec: TypeAlias = (
    DistributionParameter
    | Sequence[DistributionParameter]
    | NDArray[float64]
    | NDArray[np.integer[Any]]
)
NormalizedParameter: TypeAlias = float64 | int
NormalizedDf: TypeAlias = NormalizedParameter | tuple[NormalizedParameter, ...]


def _normalize_distribution_parameter(value: object) -> NormalizedParameter:
    if isinstance(value, bool | np.bool_):
        raise TypeError("df must be numeric")
    if isinstance(value, int | np.integer):
        return int(value)
    try:
        return float64(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise TypeError("df must be numeric") from exc


def _normalize_df(df: DfSpec) -> NormalizedDf:
    if isinstance(df, np.ndarray):
        arr = np.asarray(df)
        if arr.ndim == 0:
            return _normalize_distribution_parameter(arr.item())
        if arr.ndim != 1:
            raise ValueError("df sequence must be 1D")
        if arr.size == 0:
            raise ValueError("df sequence must be non-empty")
        return tuple(_normalize_distribution_parameter(value) for value in arr)

    if isinstance(df, Sequence):
        if isinstance(df, str | bytes):
            raise TypeError("df must be numeric or a numeric sequence")
        if len(df) == 0:
            raise ValueError("df sequence must be non-empty")
        return tuple(_normalize_distribution_parameter(value) for value in df)

    return _normalize_distribution_parameter(df)


def _df_args(df: DfSpec) -> tuple[NormalizedParameter, ...]:
    normalized = _normalize_df(df)
    if isinstance(normalized, tuple):
        return normalized
    return (normalized,)


def _compute_pvalues(
    dist: ReferenceDistribution,
    df: DfSpec,
    pval_method: PvalMethod,
    statistic: FloatScalar | NDArray[float64],
) -> tuple[FrozenDistribution, NDArray[float64]]:
    if not isinstance(dist, ReferenceDistribution):
        raise ValueError(f"Unsupported reference distribution: {dist}")
    frozen_dist = dist.freeze(*_df_args(df))
    pvals = np.asarray(pval_method(frozen_dist, statistic), dtype=np.float64)
    return frozen_dist, pvals


@dataclass(frozen=True)
class TestResult:
    test_name: str
    dist: ReferenceDistribution
    df: DfSpec
    pval_method: PvalMethod
    alpha: float64
    statistic: float64
    status: TestStatus
    _auto_pval: bool = field(default=True, repr=False, compare=False)

    _frozen_dist: FrozenDistribution | None = field(
        init=False, default=None, repr=False, compare=False
    )
    _pval: float64 | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        statistic = float64(self.statistic)
        object.__setattr__(self, "statistic", statistic)
        object.__setattr__(self, "df", _normalize_df(self.df))
        object.__setattr__(self, "status", TestStatus(self.status))

        if self._auto_pval:
            self.compute_pval()

    @property
    def frozen_dist(self) -> FrozenDistribution:
        if self._frozen_dist is None:
            self.compute_pval()
        frozen_dist = self._frozen_dist
        if frozen_dist is None:
            raise RuntimeError("p-value computation failed to freeze distribution")
        return frozen_dist

    @property
    def pval(self) -> float64:
        if self._pval is None:
            return self.compute_pval()
        return self._pval

    def compute_pval(self) -> float64:
        if self._pval is not None:
            return self._pval

        frozen_dist, pval = _compute_pvalues(
            self.dist,
            self.df,
            self.pval_method,
            self.statistic,
        )
        if pval.shape != ():
            raise ValueError("scalar statistic must produce a scalar p-value")

        object.__setattr__(
            self,
            "_frozen_dist",
            frozen_dist,
        )
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
    df: DfSpec
    pval_method: PvalMethod
    alpha: float64
    statistic_trace: NDArray[float64]
    status_trace: tuple[TestStatus, ...]

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

        status_trace = tuple(TestStatus(status) for status in self.status_trace)
        if len(status_trace) != n:
            raise ValueError(
                "status_trace and statistic_trace must have the same length"
            )

        df = _normalize_df(self.df)
        object.__setattr__(self, "df", df)
        frozen_dist, pval_trace = _compute_pvalues(
            self.dist,
            df,
            self.pval_method,
            statistic_trace,
        )
        if statistic_trace.shape != pval_trace.shape:
            raise ValueError("statistic_trace and pval_trace must have the same shape")

        object.__setattr__(
            self,
            "statistic_trace",
            statistic_trace,
        )
        object.__setattr__(
            self,
            "status_trace",
            status_trace,
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


def _df_metadata_matches(a: object, b: object) -> bool:
    """Compare normalized df metadata across replications, treating NaN == NaN
    as a match. Parameter-free reference distributions (CUSUM) carry a NaN df
    placeholder, which a plain ``!=`` would otherwise flag as incompatible."""
    at = a if isinstance(a, tuple) else (a,)
    bt = b if isinstance(b, tuple) else (b,)
    if len(at) != len(bt):
        return False
    for x, y in zip(at, bt):
        if x == y:
            continue
        if (
            isinstance(x, float | np.floating)
            and isinstance(y, float | np.floating)
            and np.isnan(x)
            and np.isnan(y)
        ):
            continue
        return False
    return True


class MCResultAccumulator:
    """Streaming builder for an :class:`MCResult` over a replication loop.

    Pulls the statistic and status off each per-replication :class:`TestResult` at
    push time so the result object itself can be released, and keeps the
    rep-invariant metadata (``dist`` / ``df`` / ``pval_method`` / ``alpha``) from
    the first push, rejecting a later result that disagrees. The statistic buffer
    is sized on ``n_rep`` up front and filled at a cursor, so a run with skipped
    replications finalizes to the count actually pushed.
    """

    __slots__ = ("_statistic", "_status", "_metadata", "_cursor")

    def __init__(self, n_rep: int) -> None:
        if n_rep <= 0:
            raise ValueError("MCResultAccumulator requires a positive n_rep.")
        self._statistic: NDArray[float64] = np.empty(n_rep, dtype=np.float64)
        self._status: list[TestStatus] = []
        self._metadata: tuple[Any, ...] | None = None
        self._cursor = 0

    @property
    def n_pushed(self) -> int:
        """Replications pushed so far."""
        return self._cursor

    def push(self, result: TestResult, *, step_name: str = "") -> None:
        """Record one replication's result, releasing the object to the caller."""
        metadata = (result.dist, result.df, result.pval_method, result.alpha)
        if self._metadata is None:
            self._metadata = metadata
        else:
            dist, df, pval_method, alpha = self._metadata
            if (
                result.dist is not dist
                or result.pval_method is not pval_method
                or not _df_metadata_matches(result.df, df)
                or result.alpha != alpha
            ):
                raise ValueError(
                    f"Test results for step '{step_name}' have incompatible metadata."
                )
        if self._cursor >= self._statistic.size:
            raise ValueError(
                f"Test results for step '{step_name}' exceed the accumulator's "
                f"capacity of {self._statistic.size} replications."
            )
        self._statistic[self._cursor] = result.statistic
        self._status.append(result.status)
        self._cursor += 1

    def finalize(self, test_name: str) -> MCResult:
        """Build the :class:`MCResult` over the replications pushed so far."""
        if self._metadata is None:
            raise ValueError(
                f"Test results for step '{test_name}' are empty; MCResult requires "
                "at least one replication."
            )
        dist, df, pval_method, alpha = self._metadata
        return MCResult(
            test_name=test_name,
            dist=dist,
            df=df,
            pval_method=pval_method,
            alpha=alpha,
            statistic_trace=self._statistic[: self._cursor].copy(),
            status_trace=tuple(self._status),
        )
