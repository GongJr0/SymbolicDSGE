"""Native OccBin kernels: the regime latch, per-regime pencils, and the solve.

Re-exports the compiled ``_occbin`` extension, which is mandatory: if it is not
built, importing this module (and the library) raises ``ImportError``.
"""

from ._occbin import (
    MAX_CONSTRAINTS,
    constraint_path,
    occbin_forward,
    occbin_recursion,
    occbin_recursion_arena_size,
    occbin_solve,
    occbin_solve_arena_size,
    regime_pencil,
)

__all__ = [
    "MAX_CONSTRAINTS",
    "constraint_path",
    "occbin_forward",
    "occbin_recursion",
    "occbin_recursion_arena_size",
    "occbin_solve",
    "occbin_solve_arena_size",
    "regime_pencil",
]
