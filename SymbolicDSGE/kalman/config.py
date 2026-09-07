from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from numpy.typing import NDArray
from numpy import float64, array, eye, ix_, outer
from sympy import Symbol

from ..core.config import SymbolGetterDict, PairGetterDict


@dataclass(frozen=True)
class KalmanConfig:
    R: NDArray | None
    R_param_names: list[str] | None = None
    R_std_param_map: SymbolGetterDict[str] | None = None
    R_corr_param_map: PairGetterDict[str | None] | None = None


def make_R(
    y_order: Sequence[Symbol | str],
    std_param_map: Mapping[Any, str],
    corr_param_map: Mapping[Any, str | None] | None,
    params: Mapping[Any, float64],
    *,
    observables: Sequence[str] | None = None,
) -> NDArray[float64]:
    """Assemble a measurement covariance ``R = outer(sig, sig) * rho``.

    The two maps carry parameter *names*; ``params`` resolves each to a value,
    so one call serves both a calibration and an estimation draw. ``y_order``
    fixes the row and column order. A pair mapped to ``None`` stays at zero
    correlation, and the diagonal is one.

    ``observables`` subsets the result. R is assembled over the whole
    ``y_order`` and sliced afterwards, since a correlation pair may name an
    observable the subset leaves out.
    """
    n = len(y_order)
    pos = {str(y): i for i, y in enumerate(y_order)}

    def _param(name: str) -> float64:
        if name not in params:
            raise KeyError(f"Missing R parameter '{name}' in the supplied parameters.")
        return float64(params[name])

    sig_vec = array([_param(std_param_map[y]) for y in y_order], dtype=float64)

    rho = eye(n, dtype=float64)
    for pair, param_name in (corr_param_map or {}).items():
        if param_name is None:
            continue
        i, j = (pos[str(member)] for member in pair)
        rho_ij = _param(param_name)
        rho[i, j] = rho_ij
        rho[j, i] = rho_ij

    R = outer(sig_vec, sig_vec) * rho
    if observables is None:
        return R

    idx = [pos[name] for name in observables]
    return R[ix_(idx, idx)]


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
