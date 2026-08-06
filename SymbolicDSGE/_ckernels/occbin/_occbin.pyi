"""Type stubs for the compiled ``_occbin`` extension.

The native kernels carry no inspectable type information (the type checker never
parses ``_occbin.pyx`` nor introspects the compiled object), so these signatures
exist solely to give the LSP and mypy the shapes of the exported functions. They
must stay in sync with ``_occbin.pyx`` / ``occbin.c``; the tests guard the
runtime behavior, not this stub.
"""

from numpy import float64, int8
from numpy.typing import NDArray

_F64 = NDArray[float64]
_I8 = NDArray[int8]

MAX_CONSTRAINTS: int

def constraint_path(
    cond_addr: int,
    path: _F64,
    par: _F64,
    regime_in: _I8,
    n_constraint: int,
    out: _I8 | None = ...,
) -> tuple[_I8, int]:
    """(regime_out, changed) <- latched regime mask over a (T, n_var) level path.

    ``out`` may alias ``regime_in`` to latch in place; ``changed`` counts the
    periods whose mask moved.
    """
