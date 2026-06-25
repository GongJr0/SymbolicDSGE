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
