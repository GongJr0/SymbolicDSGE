from dataclasses import dataclass, asdict
from pandas import DataFrame, Series


@dataclass
class FitResult:
    expressions: DataFrame
    best: Series

    def to_dict(self) -> dict:
        return asdict(self)
