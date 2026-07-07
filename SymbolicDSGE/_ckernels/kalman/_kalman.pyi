"""Type stubs for the compiled ``_kalman`` extension.

The native kernel carries no inspectable type information (the type checker never
parses ``_kalman.pyx`` nor introspects the compiled object), so this signature
exists solely to give the LSP and mypy the shape of the exported function. It
must stay in sync with ``_kalman.pyx`` / ``kalman.c`` and the numba reference
``SymbolicDSGE.kalman.filter._kalman_hot_loop`` (the two are interchangeable at
the call site); the parity tests guard the runtime behavior, not this stub.
"""

from numpy import float64
from numpy.typing import NDArray

_F64 = NDArray[float64]

def kalman_hot_loop(
    T: int,
    nmk: tuple[int, int, int],
    A: _F64,
    B: _F64,
    C: _F64,
    d: _F64,
    Q: _F64,
    R: _F64,
    y: _F64,
    x0: _F64,
    P0: _F64,
    symmetrize: bool,
    jitter: float,
    return_shocks: bool = ...,
    store_history: bool = ...,
) -> tuple[
    int,
    tuple[float64, float64, float64],
    tuple[_F64, _F64, _F64, _F64, _F64, _F64, _F64, _F64, _F64, _F64, float64],
]:
    """Run the linear Kalman filter; mirrors numba ``_kalman_hot_loop``."""

def ukf_hot_loop(
    meas_addr: int,
    hx: _F64,
    gx: _F64,
    bx: _F64,
    hxx: _F64,
    gxx: _F64,
    hss: _F64,
    gss: _F64,
    steady_state: _F64,
    params: _F64,
    Q: _F64,
    R: _F64,
    obs: _F64,
    z0: _F64,
    P0: _F64,
    alpha: float,
    beta: float,
    kappa: float,
    jitter: float,
    symmetrize: bool = ...,
    store_history: bool = ...,
) -> tuple[
    int,
    tuple[float64, float64, float64],
    tuple[_F64, _F64, _F64, _F64, _F64, _F64, _F64, _F64, _F64, _F64, _F64, float64],
]:
    """Run the native second-order UKF hot loop."""
