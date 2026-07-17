from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass(frozen=True)
class P0Config:
    mode: str
    scale: float
    diag: dict[str, float] | None


@dataclass(frozen=True)
class KalmanConfig:
    R: NDArray | None
    P0: P0Config
    R_param_names: list[str] | None = None
    R_std_param_map: dict[str, str] | None = None
    R_corr_param_map: dict[frozenset[str], str | None] | None = None


@dataclass(frozen=True)
class KalmanStateSpace:
    A: NDArray
    B: NDArray
    C: NDArray
    d: NDArray
    Q: NDArray

    y_names: list[str]
    eps_names: list[str]
    x_names: list[str]
