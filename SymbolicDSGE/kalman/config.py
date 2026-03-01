from dataclasses import dataclass
from numpy.typing import NDArray

from sympy import Symbol, Matrix
from numpy import float64, array, eye, outer
from typing import Callable

from ..core.config import PairGetterDict, SymbolGetterDict


@dataclass(frozen=True)
class P0Config:
    mode: str
    scale: float
    diag: dict[str, float] | None


@dataclass(frozen=True)
class KalmanConfig:
    y_names: list[str]
    R: NDArray | None
    jitter: float
    symmetrize: bool
    P0: P0Config
    R_symbolic: Matrix | None = None
    R_param_symbols: list[Symbol] | None = None
    R_param_names: list[str] | None = None
    R_builder: Callable[..., NDArray] | None = None


def make_R(
    y_order: list[Symbol],
    std: SymbolGetterDict[Symbol, float64],
    corr: PairGetterDict[float64],
) -> NDArray:
    n = len(y_order)

    sig_vec = array([std[y] for y in y_order], dtype=float64)

    rho = eye(n, dtype=float64)
    for pair, rho_ij in corr.items():
        a, b = tuple(pair)  # pair is a frozenset[Symbol]
        i = y_order.index(a)
        j = y_order.index(b)
        rho[i, j] = rho_ij
        rho[j, i] = rho_ij

    return outer(sig_vec, sig_vec) * rho


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
