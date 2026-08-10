"""Native OccBin kernels: the regime latch, per-regime pencils, and the solve.

Re-exports the compiled ``_occbin`` extension, which is mandatory: if it is not
built, importing this module (and the library) raises ``ImportError``.
"""

from ._occbin import (
    MAX_CONSTRAINTS,
    constraint_path,
    occbin_recursion,
    regime_pencil,
)

__all__ = [
    "MAX_CONSTRAINTS",
    "constraint_path",
    "occbin_recursion",
    "regime_pencil",
]
