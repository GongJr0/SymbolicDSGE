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

def spike_drive(fn_addr: int, a: _C128, b: _C128, out: _C128) -> None:
    """Stage-0 (#248): call a numba @cfunc (by ``.address``) from native C, nogil."""

# --- bicomplex (bc256) primitives -------------------------------------------
# A bc256 crosses the boundary as the 4-tuple (real, i, j, ij).
_BC = tuple[float, float, float, float]

def bc_add(x: _BC, y: _BC) -> _BC: ...
def bc_sub(x: _BC, y: _BC) -> _BC: ...
def bc_neg(x: _BC) -> _BC: ...
def bc_mul(x: _BC, y: _BC) -> _BC: ...
def bc_div(x: _BC, y: _BC) -> _BC: ...
def bc_real_scale(x: _BC, s: float) -> _BC: ...
def bc_i_conj(x: _BC) -> _BC: ...
def bc_j_conj(x: _BC) -> _BC: ...
def bc_conj(x: _BC) -> _BC: ...
def bc_exp(x: _BC) -> _BC: ...
def bc_log(x: _BC) -> _BC: ...
def bc_spow(x: _BC, p: float) -> _BC: ...
def bc_ipow(X: _BC, p: int) -> _BC: ...
def bc_sqrt(x: _BC) -> _BC: ...
def c_sqrt(z: tuple[float, float]) -> tuple[float, float]: ...
def bc_cpow(x: _BC, y: _BC) -> _BC: ...
def bc_accessors(x: _BC) -> _BC: ...
def bc_proj(x: _BC) -> tuple[float, float, float, float]: ...
def bc_reconst(p: tuple[float, float, float, float]) -> _BC: ...
