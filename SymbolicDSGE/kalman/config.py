from dataclasses import dataclass
from numpy.typing import NDArray
from numpy import float64, array, eye, outer, ndarray
from sympy import Symbol


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


def make_R(
    y_order: list[Symbol],
    std: dict[Symbol, float64],
    corr: dict[frozenset, float64],
) -> ndarray:
    """Assemble a measurement covariance ``R = outer(sig, sig) * rho``.

    ``std`` maps each observable symbol to its standard deviation; ``corr`` maps
    each specified observable pair (a ``frozenset`` of two symbols) to its
    correlation. Unspecified pairs default to zero and the diagonal to one.
    """
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
