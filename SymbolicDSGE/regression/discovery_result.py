from dataclasses import dataclass, asdict
import pandas as pd


@dataclass(frozen=True)
class DiscoveryResult:
    """Structured representation of a symbolic regression discovery result."""

    top_candidates: pd.DataFrame
    qualified_expressions: pd.DataFrame
    disqualified_expressions: pd.DataFrame

    def to_dict(self) -> dict[str, pd.DataFrame]:
        """Return a dictionary representation of the data class."""
        return asdict(self)
