"""Type stubs for the compiled ``_regression`` extension.

The native kernels carry no inspectable type information, so these signatures
exist solely for the LSP / mypy. They must stay in sync with ``_regression.pyx``
/ ``regression.c`` and the numba references in ``SymbolicDSGE.regression``; the
parity tests guard the runtime behavior, not this stub.
"""

from numpy import float64
from numpy.typing import NDArray

_F64 = NDArray[float64]

def chol_solve_L2(
    X: _F64, y: _F64, alpha: float, intercept: bool
) -> tuple[_F64, _F64, float, int]: ...
def ridge_grid_search(
    X: _F64, y: _F64, alphas: _F64, criterion: int, intercept: bool
) -> tuple[float, _F64, float, int]: ...
def lasso_gram_cd(
    G: _F64, g: _F64, alpha: float, max_iter: int, tol: float
) -> tuple[_F64, int]: ...
def lars_lasso_gram(
    G: _F64, c: _F64, max_iter: int, tol: float
) -> tuple[_F64, _F64, int]: ...
def lasso_path_eval(lam_path: _F64, beta_path: _F64, lam_grid: _F64) -> _F64: ...
