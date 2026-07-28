from numpy import float64, int64
from numpy.typing import NDArray

_F64 = NDArray[float64]
_I64 = NDArray[int64]

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
def ridge_fit(
    X: _F64,
    y: _F64,
    coef: _F64,
    alpha: float,
    L: _F64,
    G: _F64,
    G_unpen: _F64,
    g: _F64,
    col: _F64,
    intercept: bool,
) -> tuple[float64, float64, int]: ...
def ridge_gs_fit(
    X: _F64,
    y: _F64,
    alphas: _F64,
    criterion: int,
    coef: _F64,
    G_base: _F64,
    G: _F64,
    L: _F64,
    g: _F64,
    coef_work: _F64,
    col: _F64,
    intercept: bool,
) -> tuple[float64, float64, int]: ...
def lasso_fit(
    X: _F64,
    y: _F64,
    coef: _F64,
    alpha: float,
    max_iter: int,
    tol: float,
    G_base: _F64,
    G: _F64,
    g: _F64,
    Gcoef: _F64,
    intercept: bool,
) -> tuple[float64, float64, int]: ...
def lasso_gs_fit(
    X: _F64,
    y: _F64,
    alphas: _F64,
    coef: _F64,
    max_iter: int,
    tol: float,
    G_base: _F64,
    G: _F64,
    g: _F64,
    lam_path: _F64,
    beta_path: _F64,
    beta_grid: _F64,
    work: _F64,
    intercept: bool,
) -> tuple[float64, float64, int]: ...
def elastic_net_fit(
    X: _F64,
    y: _F64,
    coef: _F64,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    G_base: _F64,
    G: _F64,
    g: _F64,
    Gcoef: _F64,
    intercept: bool,
) -> tuple[float64, float64, int]: ...
def elastic_net_gs_fit(
    X: _F64,
    y: _F64,
    alphas: _F64,
    l1_ratio: float,
    criterion: int,
    coef: _F64,
    max_iter: int,
    tol: float,
    G_base: _F64,
    G: _F64,
    g: _F64,
    beta_grid: _F64,
    statuses: _I64,
    Gcoef: _F64,
    beta: _F64,
    dof_work: _F64,
    intercept: bool,
) -> tuple[float64, float64, int]: ...

"""OLS backend for MC replications. Writes into {coef, se, L, G, g} and returns {ssr, sst, status}."""
