from dataclasses import dataclass
from numpy.typing import NDArray
from numpy import float64, array, eye, zeros, outer, ndarray
from sympy import Symbol


@dataclass(frozen=True)
class KalmanConfig:
    R: NDArray | None
    P0: NDArray
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


def make_diag_P0(
    diag: dict[str, float], ordered_var_names: list[str]
) -> NDArray[float64]:
    n = len(ordered_var_names)
    P0 = zeros((n, n), dtype=float64)

    for i, var_name in enumerate(ordered_var_names):
        P0[i, i] = diag[var_name]
    return P0


def make_P0(
    mode: str, diag: dict[str, float] | None, ordered_var_names: list[str]
) -> NDArray[float64]:
    if mode == "diag" and diag:
        return make_diag_P0(diag, ordered_var_names)
    else:
        n = len(ordered_var_names)
        return eye(n, dtype=float64)


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
