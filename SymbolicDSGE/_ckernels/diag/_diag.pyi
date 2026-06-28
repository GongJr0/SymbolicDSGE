"""Type stubs for the compiled ``_diag`` extension.

The native kernels carry no inspectable type information (the type checker never
parses ``_diag.pyx`` nor introspects the compiled object), so these signatures
exist solely to give the LSP and mypy the shapes of the exported functions. They
must stay in sync with ``_diag.pyx`` / ``diag.c`` and the numba references in
``SymbolicDSGE._diag_tests``; the parity tests guard the runtime behavior, not
this stub.
"""

from numpy import float64
from numpy.typing import NDArray

_F64 = NDArray[float64]

#: Status value meaning "design rank-deficient -- retry the statistic in numba".
FALLBACK: int

def bg_stat(eps: _F64, X: _F64, lags: int) -> tuple[int, float]:
    """Breusch-Godfrey LM statistic. Returns (status, stat)."""

def bp_aux(eps: _F64, X_aug: _F64) -> tuple[int, float, float]:
    """Breusch-Pagan auxiliary regression. Returns (status, rss, tss)."""

def chow_stat(y: _F64, X: _F64, t_break: int) -> tuple[int, float]:
    """Chow break-point F statistic. Returns (status, stat)."""

def recursive_residuals(y: _F64, X: _F64) -> tuple[int, _F64]:
    """Brown-Durbin-Evans recursive residuals. Returns (status, w)."""

def cusum_series(y: _F64, X: _F64) -> tuple[int, _F64]:
    """Standardized CUSUM series. Returns (status, series)."""

def cusum_stat(y: _F64, X: _F64) -> tuple[int, float]:
    """CUSUM statistic. Returns (status, stat)."""

def cusumsq_stat(y: _F64, X: _F64) -> tuple[int, int, float]:
    """CUSUM-of-squares statistic. Returns (status, n, stat)."""

def jb_stat(x: _F64) -> tuple[int, float]:
    """Jarque-Bera normality statistic. Returns (status, stat)."""

def acorr(x: _F64, L: int) -> tuple[int, _F64]:
    """Autocorrelation of x up to lag L. Returns (status, out(L+1))."""

def lb_stat(x: _F64, L: int) -> tuple[int, float]:
    """Ljung-Box statistic for x up to lag L. Returns (status, stat)."""

def fill_mean_ax0(x: _F64) -> _F64:
    """Column means of x over axis 0. Returns mean(p)."""

def fill_centered_ax0(x: _F64, mean: _F64) -> _F64:
    """x with its column means subtracted. Returns centered(n, p)."""

def hac_estimator_matmul(r: _F64, kernel_id: int, L: int) -> _F64:
    """HAC long-run covariance (full estimator). Returns the (p, p) matrix."""

def wald_stat_from_mean_and_cov(
    mean: _F64, target: _F64, omega: _F64, n: int
) -> tuple[int, float]:
    """Wald statistic n * dev^T omega^-1 dev. Returns (status, stat)."""

def symmetric_outer_prod_2dim(x: _F64) -> tuple[int, _F64]:
    """Per-row vech of x_t x_t'. Returns (status, out(n, q))."""

def fill_symmetric_target_vec(
    target: _F64, atol: float, rtol: float
) -> tuple[int, _F64]:
    """Pack the upper triangle of a symmetric target. Returns (status, vec(q))."""
