from numpy import float64
from numpy.typing import NDArray

from typing import Sequence, TypedDict

_F64 = NDArray[float64]
_Bound = tuple[float | None, float | None]

class OptimResult(TypedDict):
    x: _F64
    fun: float
    nfev: int
    nit: int
    success: bool
    status: int
    message: str

# Native optimizer drivers over the synthetic benchmark objectives
# ("rosenbrock", "quad", "double_well", "rosen_halfplane"). ``bounds`` is None
# (unbounded) or a length-n sequence of (lo, hi) with None for a missing side;
# ``params`` feeds the parametrized objectives (e.g. "quad").

def run_lbfgsb(
    objective: str,
    x0: _F64,
    bounds: Sequence[_Bound] | None = ...,
    params: _F64 | None = ...,
    m: int = ...,
    maxiter: int = ...,
    maxfun: int = ...,
    maxls: int = ...,
    factr: float = ...,
    pgtol: float = ...,
    fd_step: float = ...,
) -> OptimResult: ...
def run_neldermead(
    objective: str,
    x0: _F64,
    bounds: Sequence[_Bound] | None = ...,
    params: _F64 | None = ...,
    maxiter: int = ...,
    maxfun: int = ...,
    xatol: float = ...,
    fatol: float = ...,
) -> OptimResult: ...
