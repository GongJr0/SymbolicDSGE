"""Type stubs for the compiled ``_core`` extension.

The native kernels carry no inspectable type information (the type checker never
parses ``_core.pyx`` nor introspects the compiled object), so these signatures
exist solely to give the LSP and mypy the shapes of the exported functions. They
must stay in sync with ``_core.pyx`` / ``core.c`` and the numba reference in
``SymbolicDSGE.core.simulation``; the parity tests guard the runtime behavior,
not this stub.
"""

from numpy import complex128, float64
from numpy.typing import NDArray

_F64 = NDArray[float64]
_C128 = NDArray[complex128]

def simulate_linear_states_into(
    A: _F64,
    B: _F64,
    x0: _F64,
    shock_mat: _F64,
    out: _F64,
) -> None:
    """out[(T+1, n)] <- linear state recursion. Mirrors the numba kernel."""

def affine_observations_into(
    states: _F64,
    C: _F64,
    d: _F64,
    state_start: int,
    out: _F64,
) -> None:
    """out[(T, m)] <- d + C @ states[state_start + t]. Mirrors the numba kernel."""

def klein_postprocess(
    s: _C128,
    t: _C128,
    z: _C128,
    n_states: int,
) -> tuple[_C128, _C128, int, _C128]:
    """(f, p, stab, eig) from the ordered Schur factors. Mirrors the numba path."""
