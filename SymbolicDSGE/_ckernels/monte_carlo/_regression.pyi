from numpy import float64
from numpy.typing import NDArray

_F64 = NDArray[float64]

def ols_fit(
    X: _F64,  # (n, p)
    y: _F64,  # (n,)
    coef: _F64,  # (p,)
    se: _F64,  # (p,)
    L: _F64,  # (p, p)
    G: _F64,  # (p, p)
    g: _F64,  # (p,)
    work: _F64,  # (p,)
    intercept: bool,
) -> tuple[float64, float64, int]: ...

"""OLS backend for MC replications. Writes into {coef, se, L, G, g} and returns {ssr, sst, status}."""
