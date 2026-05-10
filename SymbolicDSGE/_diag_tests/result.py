from dataclasses import dataclass, asdict, field
from numpy import float64, sqrt
from numpy.typing import NDArray


@dataclass(frozen=True)
class TestResult:
    test_name: str
    alpha: float64
    statistic: float64
    pval: float64

    def is_significant(self, threshold: float | float64 | None = None) -> bool:
        if threshold is None:
            threshold = self.alpha
        return bool(self.pval < threshold)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MCResult:
    test_name: str
    alpha: float64
    n: int
    statistic_trace: NDArray[float64]
    pval_trace: NDArray[float64]

    mean_statistic: float64 = field(init=False)
    mean_pval: float64 = field(init=False)
    rejection_rate: float64 = field(init=False)
    pval_se: float64 = field(init=False)
    statistic_se: float64 = field(init=False)

    def __post_init__(self) -> None:
        if self.statistic_trace.shape != self.pval_trace.shape:
            raise ValueError("statistic_trace and pval_trace must have the same shape")

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
